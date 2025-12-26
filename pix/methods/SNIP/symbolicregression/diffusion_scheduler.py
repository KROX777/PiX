import torch
import torch.nn.functional as F
import math
import sys

BETA_START = 0.0001
BETA_END = 0.02

PAD_TOKEN_ID = 8
PAD_LOSS_WEIGHT = 0.5
EOS_TOKEN_ID = 0
EOS_LOSS_WEIGHT = 2.0

def linear_beta_schedule(timesteps, beta_start=BETA_START, beta_end=BETA_END):
    return torch.linspace(beta_start, beta_end, timesteps, dtype=torch.float64)

def cosine_beta_schedule(timesteps, s=0.008):
    steps = timesteps + 1
    x = torch.linspace(0, timesteps, steps, dtype=torch.float64)
    alphas_cumprod = torch.cos(((x / timesteps) + s) / (1 + s) * math.pi * 0.5) ** 2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    return torch.clip(betas, 0.0001, 0.9999)

class DiscreteDiffusion:
    def __init__(self, num_timesteps=100, vocab_size=None, device='cuda', schedule_type='linear'):
        self.num_timesteps = num_timesteps
        self.vocab_size = vocab_size
        self.device = device

        if schedule_type == 'linear':
            self.betas = linear_beta_schedule(num_timesteps).to(device)
        elif schedule_type == 'cosine':
            self.betas = cosine_beta_schedule(num_timesteps).to(device)
        else:
            raise ValueError(f"Unknown schedule type: {schedule_type}")

        self.alphas = (1. - self.betas).to(device)
        self.alphas_cumprod = torch.cumprod(self.alphas, axis=0).to(device)
        self.alphas_cumprod_prev = F.pad(self.alphas_cumprod[:-1], (1, 0), value=1.0).to(device)

        # Compact representation to avoid O(V^2)/O(V^3) allocations
        V = int(self.vocab_size)
        eps = 1e-40
        # q(x_t | x_{t-1}) has two values per t: diag and off-diag
        # diag_prob = 1 - beta_t + beta_t / V
        # off_prob  = beta_t / V
        self.log_q_t_x_t_minus_1_diag = torch.log((1.0 - self.betas) + (self.betas / V)).to(device).clamp(min=eps).float()
        self.log_q_t_x_t_minus_1_off = torch.log((self.betas / V)).to(device).clamp(min=eps).float()

        # q(x_t | x_0) also has diag / off-diag per t
        diag = self.alphas_cumprod + (1.0 - self.alphas_cumprod) / V
        off = (1.0 - self.alphas_cumprod) / V
        self.log_q_t_x_0_diag = torch.log(diag.clamp(min=eps)).to(device).float()
        self.log_q_t_x_0_off = torch.log(off.clamp(min=eps)).to(device).float()

    # NOTE: We no longer precompute full (V,V) or (V,V,V) matrices. Instead we
    # keep compact per-t diag/off-diag log probs and compute needed (B,S,V)
    # tensors on the fly. This avoids O(V^2)/O(V^3) memory blowups for large V.

    def q_sample(self, x_start, t):
        # Sample x_t ~ q(x_t | x_0) without materializing full V-dim vectors.
        # We use the fact that q(x_t|x_0=k) is: p_diag for j==k, and p_off for j!=k.
        B, S = x_start.shape
        device = x_start.device
        V = int(self.vocab_size)

        # ensure t is a tensor of shape (B,)
        if t.dim() == 0:
            t = t.view(1).expand(B)
        t = t.to(device)

        # expand per-position
        t_exp = t.unsqueeze(1).expand(B, S).contiguous()
        N = B * S
        k_flat = x_start.view(-1)
        t_flat = t_exp.view(-1)

        # get diag probs for q(x_t | x_0)
        log_diag = self.log_q_t_x_0_diag[t_flat]    # (N,)
        p_diag = torch.exp(log_diag)

        # Bernoulli decide whether to keep k or sample an off-token
        u = torch.rand(N, device=device)
        choose_diag = u < p_diag

        x_t_flat = torch.empty(N, device=device, dtype=torch.long)
        # positions that keep the original token
        if choose_diag.any():
            x_t_flat[choose_diag] = k_flat[choose_diag]

        # positions that sample uniform from other V-1 tokens
        off_idx = (~choose_diag).nonzero(as_tuple=False).squeeze(-1)
        num_off = off_idx.numel()
        if num_off > 0:
            r = torch.randint(0, V - 1, size=(num_off,), device=device)
            k_off = k_flat[off_idx]
            # map r in [0, V-2] to token space excluding k
            mapped = r + (r >= k_off).long()
            x_t_flat[off_idx] = mapped

        return x_t_flat.view(B, S).long()

    def q_posterior_log_probs(self, x_0, x_t, t):
        # Returns normalized log q(x_{t-1}=j | x_t, x_0) for the given x_0/x_t.
        B, S = x_0.shape
        device = x_0.device
        V = int(self.vocab_size)

        # ensure t is tensor shape (B,)
        if t.dim() == 0:
            t = t.view(1).expand(B)
        t = t.to(device)
        t_exp = t.unsqueeze(1).expand(B, S).contiguous()

        N = B * S
        i_flat = x_t.view(-1).clamp(0, V - 1)
        k_flat = x_0.view(-1).clamp(0, V - 1)
        t_flat = t_exp.view(-1)

        # per-position scalars
        log_diag_given = self.log_q_t_x_t_minus_1_diag[t_flat]
        log_off_given = self.log_q_t_x_t_minus_1_off[t_flat]
        log_diag_prev = self.log_q_t_x_0_diag[(t_flat - 1).clamp(min=0)]
        log_off_prev = self.log_q_t_x_0_off[(t_flat - 1).clamp(min=0)]

        # base = log_off_given + log_off_prev
        base = (log_off_given + log_off_prev).unsqueeze(1)          # (N,1)
        result = base.expand(N, V).clone()                          # (N,V)

        # add delta for j == i (given transition)
        delta_given = (log_diag_given - log_off_given).to(device)
        rows = torch.arange(N, device=device)
        result[rows, i_flat] += delta_given

        # add delta for j == k (previous|0 transition)
        delta_prev = (log_diag_prev - log_off_prev).to(device)
        result[rows, k_flat] += delta_prev

        # normalize to log probs per position
        logsum = torch.logsumexp(result, dim=1, keepdim=True)
        result = result - logsum
        result = result.view(B, S, V)
        result = torch.clamp(result, -100.0, 0.0)
        return result

    def p_log_probs(self, model, x_t, t, condition):
        log_pred_x0 = model(x_t, t, condition)
        return F.log_softmax(log_pred_x0, dim=-1)

    def p_sample(self, model, x_t, t, condition):
        batch_size, seq_len = x_t.shape
        device = x_t.device
        # Efficient computation of p(x_{t-1} | x_t, condition) without V^2/V^3.
        # Formula: log p(j) = logsum_{x0} [ log q(j | x_t, x0) + log p_model(x0 | x_t, cond) ]
        # We exploit structure to compute this with O(V) per position.
        log_pred_x0 = self.p_log_probs(model, x_t, t, condition)  # (B,S,V)
        B, S, V = log_pred_x0.shape

        # ensure t is tensor shape (B,)
        if t.dim() == 0:
            t = t.view(1).expand(batch_size)
        t = t.to(device)
        t_exp = t.unsqueeze(1).expand(batch_size, seq_len).contiguous()

        # Flatten positions
        N = B * S
        i_flat = x_t.view(-1).clamp(0, V - 1)
        t_flat = t_exp.view(-1)

        # per-position scalars
        log_diag_given = self.log_q_t_x_t_minus_1_diag[t_flat]
        log_off_given = self.log_q_t_x_t_minus_1_off[t_flat]
        log_diag_prev = self.log_q_t_x_0_diag[(t_flat - 1).clamp(min=0)]
        log_off_prev = self.log_q_t_x_0_off[(t_flat - 1).clamp(min=0)]

        # base vector for all j
        base_given = log_off_given + log_off_prev   # (N,)
        delta_given = (log_diag_given - log_off_given)  # (N,)
        delta_prev = (log_diag_prev - log_off_prev)     # (N,)

        # p_model probabilities from log_pred_x0
        log_pred_flat = log_pred_x0.view(N, V)
        p_model = torch.exp(log_pred_flat)  # (N, V)

        # Compute for each j: s_j = base_j + log( (1 - p_j) + p_j * exp(delta_prev) )
        # base_j equals base_given for all j, except add delta_given when j == i
        # so compute common_term = log( (1 - p_j) + p_j * exp(delta_prev) )
        exp_delta_prev = torch.exp(delta_prev).unsqueeze(1)  # (N,1)
        common = torch.log((1.0 - p_model) + p_model * exp_delta_prev.clamp(min=1e-30))  # (N,V)

        base_vec = base_given.unsqueeze(1).expand(N, V).clone()
        rows = torch.arange(N, device=device)
        base_vec[rows, i_flat] += delta_given

        s = base_vec + common  # (N, V)
        # normalize over j to produce log probabilities
        log_p = F.log_softmax(s, dim=1)

        # sample with Gumbel-max
        gumbel = -torch.log(-torch.log(torch.rand_like(log_p).clamp(min=1e-9)) + 1e-9)
        x_t_minus_1_flat = torch.argmax(log_p + gumbel, dim=1)

        return x_t_minus_1_flat.view(B, S).long()

# AI Modifed loss function
    def compute_loss(
            self,
            model,
            x_start,
            condition,
            *,
            pad_token_id = PAD_TOKEN_ID,
            eos_token_id = EOS_TOKEN_ID,
            eos_weight: float = EOS_LOSS_WEIGHT,
            pad_weight: float = PAD_LOSS_WEIGHT,      
        ):
        """
        Cross-entropy on every position **including** PAD, with:
        • extra reward for predicting the first <eos>
        • mild penalty for any non-PAD emitted *after* the first <eos>
        """
        B, S = x_start.shape
        device = x_start.device

        # 1 pick diffusion timestep and corrupt the input
        t   = torch.randint(0, self.num_timesteps, (B,), device=device).long()
        x_t = self.q_sample(x_start, t)                     # (B,S)

        # 2 model prediction (log-probs over vocab)
        log_pred_x0 = self.p_log_probs(model, x_t, t, condition)  # (B,S,V)

        # 3 plain NLL for every token (no ignore_index!)
        loss_tok = F.nll_loss(
            log_pred_x0.permute(0, 2, 1),   # (B,V,S)
            x_start,                        # (B,S)
            reduction='none'                # keep per-token loss
        )                                   # (B,S)

        # ------------------------------------------------------------------ #
        # 4 build a per-token weight matrix
        weights = torch.ones_like(x_start, dtype=torch.float, device=device)

        # (a) first <eos> in each sequence
        eos_mask = x_start == eos_token_id
        weights  += eos_mask * (eos_weight - 1.0)     # boost eos

        # (b) tokens AFTER the first <eos> are expected to be PAD
        eos_cum   = torch.cumsum(eos_mask.long(), dim=1)          # (B,S)
        should_be_pad = (eos_cum > 0) & ~eos_mask                # after-eos positions
        weights = torch.where(should_be_pad, pad_weight * weights, weights)

        # (c) “real” padding on the right of sequences *before* eos — ignore
        true_pad = (x_start == pad_token_id) & (eos_cum == 0)
        weights  = torch.where(true_pad, torch.zeros_like(weights), weights)

        # ------------------------------------------------------------------ #
        # 5 aggregate
        weighted_loss = loss_tok * weights
        denom = weights.sum().clamp(min=1)       # avoid div-by-zero
        loss  = weighted_loss.sum() / denom

        return loss

    @torch.no_grad()
    def sample(self, model, condition, shape):
        batch_size, seq_len = shape
        device = self.device
        model.eval()
        x_t = torch.randint(1, self.vocab_size, size=shape, device=device).long()

        for t in reversed(range(0, self.num_timesteps)):
            print(f"\rSampling timestep {t+1}/{self.num_timesteps}   ", end="")
            sys.stdout.flush()
            t_tensor = torch.full((batch_size,), t, device=device, dtype=torch.long)
            if t > 0:
                 x_t = self.p_sample(model, x_t, t_tensor, condition)
            else:
                 log_pred_x0 = self.p_log_probs(model, x_t, t_tensor, condition)
                 gumbel_noise = torch.rand_like(log_pred_x0)
                 gumbel_noise = -torch.log(-torch.log(gumbel_noise.clamp(min=1e-9)) + 1e-9)
                 x_t = torch.argmax(log_pred_x0 + gumbel_noise, dim=-1).long()

        print("\nSampling complete.")
        model.train()
        return x_t


class LatentDiffusion:
    """Lightweight latent (Gaussian) diffusion scheduler for z vectors.
    Provides q_sample and helper scalars (sqrt_alphas_cumprod, sqrt_one_minus_alphas_cumprod).
    """
    def __init__(self, num_timesteps=100, latent_dim=None, device='cuda', schedule_type='linear'):
        self.num_timesteps = num_timesteps
        self.latent_dim = latent_dim
        self.device = device

        if schedule_type == 'linear':
            betas = linear_beta_schedule(num_timesteps).to(device)
        elif schedule_type == 'cosine':
            betas = cosine_beta_schedule(num_timesteps).to(device)
        else:
            raise ValueError(f"Unknown schedule type: {schedule_type}")

        self.betas = betas.float()
        self.alphas = (1.0 - self.betas).float()
        self.alphas_cumprod = torch.cumprod(self.alphas, dim=0).to(device)
        self.sqrt_alphas_cumprod = torch.sqrt(self.alphas_cumprod).to(device)
        self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1.0 - self.alphas_cumprod).to(device)

    def q_sample(self, z0, t, noise=None):
        """z0: (B, D), t: (B,) long
        returns z_t and noise used
        """
        device = z0.device
        B = z0.size(0)
        if t.dim() == 0:
            t = t.view(1).expand(B)
        t = t.to(device)

        a = self.sqrt_alphas_cumprod[t].unsqueeze(-1)  # (B,1)
        b = self.sqrt_one_minus_alphas_cumprod[t].unsqueeze(-1)

        if noise is None:
            noise = torch.randn_like(z0, device=device)

        z_t = a * z0 + b * noise
        return z_t, noise

    def get_scalars(self, t):
        if t.dim() == 0:
            t = t.view(1).expand(1)
        return self.sqrt_alphas_cumprod[t], self.sqrt_one_minus_alphas_cumprod[t]

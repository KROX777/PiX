import torch
import torch.nn as nn
import numpy as np
import math
import random

SEQ_LEN = 30
N_POINTS = 200
CONDITION_FEATURE_DIM = 3 # <<< MUST SET THIS BASED ON THEIR DATA SHAPE (e.g., 3 if X_Y_combined has columns [x1, x2, y])
VOCAB_SIZE = 19     #OG Pms, not used

PAD_TOKEN_ID = 8
PAD_LOSS_WEIGHT = 0.5
EOS_TOKEN_ID = 0
EOS_LOSS_WEIGHT = 2.0

# Model Capacity
EMBED_DIM = 512
NUM_HEADS = 8           # Changed from 16 to 8 (paper setting)
NUM_LAYERS = 8          # Changed from 16 to 8 (paper setting)
DIM_FEEDFORWARD = 2048
DROPOUT = 0.15          # Changed from 0.1 to 0.15 (paper setting)

def setup_seed(seed):
     torch.manual_seed(seed)
     torch.cuda.manual_seed_all(seed)
     np.random.seed(seed)
     random.seed(seed)
     torch.backends.cudnn.deterministic = True

setup_seed(42)

def extract(a, t, x_shape):
    batch_size = t.shape[0]
    out = a.gather(-1, t)
    return out.reshape(batch_size, *((1,) * (len(x_shape) - 1)))

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)
    def forward(self, x):
        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)

class TimestepEmbedding(nn.Module):
    def __init__(self, dim, max_period=10000):
        super().__init__()
        self.dim = dim
        self.max_period = max_period
    def forward(self, t):
        device = t.device
        half = self.dim // 2
        freqs = torch.exp(-math.log(self.max_period) * torch.arange(start=0, end=half, dtype=torch.float32) / half).to(device)
        args = t[:, None].float() * freqs[None, :]
        embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        if self.dim % 2:
            embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
        return embedding

class SNIPConditionProjection(nn.Module):
    """
    Project SNIP latent z_rep (B, latent_dim)
    into a condition key/value tensor (B, cond_seq_len, embed_dim) for cross-attention.
    Behavior: linear -> relu -> reshape to (B, cond_seq_len, embed_dim)
    """
    def __init__(self, input_dim, embed_dim, cond_seq_len=16):
        super().__init__()
        self.input_dim = input_dim
        self.embed_dim = embed_dim
        self.cond_seq_len = cond_seq_len

        # projection from latent to flattened (cond_seq_len * embed_dim)
        self.latent_proj = nn.Sequential(
            nn.Linear(self.input_dim, self.cond_seq_len * self.embed_dim),
            nn.ReLU(),
            nn.LayerNorm(self.cond_seq_len * self.embed_dim),
        )

        # per-position projection registry for incoming src_enc with different feature dim
        # keys are input dim sizes (str), values are nn.Linear(in_dim, embed_dim)
        self.src_proj = nn.ModuleDict()
        # adapters for latent vectors with unexpected dims -> map to expected input_dim
        self.latent_adapters = nn.ModuleDict()

    def forward(self, condition):
        """Return (B, cond_seq_len, embed_dim)"""
        # condition can be:
        #  - z_rep : (B, latent_dim)
        #  - mapped src_enc : (B, seq_in, embed_dim) or (B, seq_in, some_dim)
        if condition.dim() == 2:
            # latent vector
            B = condition.size(0)
            d = condition.size(1)
            if d != self.input_dim:
                # create adapter to expected latent input dim
                key = f"latent_{d}"
                if key not in self.latent_adapters:
                    self.latent_adapters[key] = nn.Linear(d, self.input_dim)
                condition = self.latent_adapters[key](condition)
            x = self.latent_proj(condition)  # (B, cond_seq_len * embed_dim)
            x = x.view(B, self.cond_seq_len, self.embed_dim)
            return x
        else:
            raise ValueError(f"Unsupported condition dim: {condition.dim()}")


class ConditionalD3PMTransformer(nn.Module):
    def __init__(self, vocab_size, embed_dim, num_heads, num_layers, dim_feedforward,
                 seq_len,
                 condition_feature_dim,
                 num_timesteps, dropout=0.1,
                 cond_seq_len=16,
                 snip_latent_dim=None,
                 use_external_condition_proj: bool = False,
                 ): 
        super().__init__()
        self.vocab_size = vocab_size
        self.embed_dim = embed_dim
        self.seq_len = seq_len
        self.num_layers = num_layers
        # condition projection settings
        self.cond_seq_len = cond_seq_len
        # use provided snip_latent_dim if given, otherwise fallback to condition_feature_dim
        self.snip_latent_dim = snip_latent_dim if snip_latent_dim is not None else condition_feature_dim

        # Embeddings
        self.token_embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=PAD_TOKEN_ID)
        self.positional_encoding = PositionalEncoding(embed_dim, dropout, max_len=seq_len + 1)
        self.timestep_embedding = nn.Sequential(
            TimestepEmbedding(embed_dim),
            nn.Linear(embed_dim, embed_dim), nn.SiLU(), nn.Linear(embed_dim, embed_dim)
        )
        # self.condition_encoder = PointCloudEncoder(input_dim=condition_feature_dim, embed_dim=embed_dim)

        # SNIP condition projector: maps latent vectors or mapped src_enc to (B, cond_seq_len, embed_dim)
        # If an external projector will be provided by the caller (build_modules/Trainer),
        # we can disable internal projector creation to avoid duplicated parameters.
        self.use_external_condition_proj = bool(use_external_condition_proj)
        if not self.use_external_condition_proj:
            self.condition_projector = SNIPConditionProjection(self.snip_latent_dim, embed_dim, cond_seq_len=self.cond_seq_len)
        else:
            self.condition_projector = None

        self.condition_dropout_prob = 0.05 # Set to 0 to disable

        #Transformer Block Components
        self.encoder_self_attn_layers = nn.ModuleList()
        self.encoder_cross_attn_layers = nn.ModuleList()
        self.encoder_ffn_layers = nn.ModuleList()
        self.encoder_norm1_layers = nn.ModuleList()
        self.encoder_norm2_layers = nn.ModuleList()
        self.encoder_norm3_layers = nn.ModuleList()
        self.dropout_layers = nn.ModuleList()

        for _ in range(num_layers):
            self.encoder_self_attn_layers.append(nn.MultiheadAttention(embed_dim, num_heads, dropout=dropout, batch_first=True))
            self.encoder_cross_attn_layers.append(nn.MultiheadAttention(embed_dim, num_heads, dropout=dropout, batch_first=True))
            self.encoder_ffn_layers.append(nn.Sequential(
                nn.Linear(embed_dim, dim_feedforward), nn.GELU(), nn.Dropout(dropout),
                nn.Linear(dim_feedforward, embed_dim)
            ))
            self.encoder_norm1_layers.append(nn.LayerNorm(embed_dim))
            self.encoder_norm2_layers.append(nn.LayerNorm(embed_dim))
            self.encoder_norm3_layers.append(nn.LayerNorm(embed_dim))
            self.dropout_layers.append(nn.Dropout(dropout))

        self.output_layer = nn.Linear(embed_dim, vocab_size)
        self.init_weights()

    def init_weights(self):
        initrange = 0.1
        self.token_embedding.weight.data.uniform_(-initrange, initrange)
        if self.token_embedding.padding_idx is not None:
             self.token_embedding.weight.data[self.token_embedding.padding_idx].zero_()
        self.output_layer.bias.data.zero_()
        self.output_layer.weight.data.uniform_(-initrange, initrange)
        if self.condition_projector is not None:
            for layer in self.condition_projector.modules():
                if isinstance(layer, (nn.Linear,)):
                    # follow project's convention: small normal init
                    layer.weight.data.normal_(mean=0.0, std=0.02)
                    if layer.bias is not None:
                        layer.bias.data.zero_()
                elif isinstance(layer, (nn.LayerNorm,)):
                    try:
                        layer.weight.data.fill_(1.0)
                        layer.bias.data.zero_()
                    except Exception:
                        pass

    def forward(self, x, t, condition):
        # CONDITION INPUT SHAPE: Expects (B, N_POINTS, CONDITION_FEATURE_DIM)
        batch_size, seq_len = x.shape
        device = x.device

        token_emb = self.token_embedding(x) * math.sqrt(self.embed_dim)
        token_emb_permuted = token_emb.transpose(0, 1)
        pos_emb_permuted = self.positional_encoding(token_emb_permuted)
        pos_emb = pos_emb_permuted.transpose(0, 1)
        time_emb = self.timestep_embedding(t)
        time_emb = time_emb.unsqueeze(1).expand(-1, seq_len, -1)

        # Condition handling:
        # - If `condition` is already a projected KV tensor (B, cond_seq_len, embed_dim), accept it.
        # - If `condition` is a 2D SNIP latent (B, latent_dim) and this module has an internal
        #   projector, use it. If no internal projector exists, raise an error (caller must project).
        if condition.dim() == 3:
            # already projected (B, cond_seq_len, embed_dim)
            cond_kv = condition
        elif condition.dim() == 2:
            if self.condition_projector is None:
                raise ValueError(
                    "ConditionalD3PMTransformer was constructed without an internal condition projector; "
                    "please provide a projected condition (B, cond_seq_len, embed_dim)"
                )
            cond_kv = self.condition_projector(condition)
        else:
            raise ValueError(f"Unsupported condition dim: {condition.dim()}")
        if self.training and self.condition_dropout_prob > 0:
            # mask shape should broadcast across (cond_seq_len, embed_dim): use (B,1,1)
            mask = (torch.rand(cond_kv.shape[0], 1, 1, device=cond_kv.device) > self.condition_dropout_prob).float()
            cond_kv = cond_kv * mask

        current_input = pos_emb + time_emb
        padding_mask = (x == PAD_TOKEN_ID)

        for i in range(self.num_layers):
            sa_norm_input = self.encoder_norm1_layers[i](current_input)
            sa_output, _ = self.encoder_self_attn_layers[i](query=sa_norm_input, key=sa_norm_input, value=sa_norm_input, key_padding_mask=padding_mask)
            x = current_input + self.dropout_layers[i](sa_output)
            ca_norm_input = self.encoder_norm3_layers[i](x)
            ca_output, _ = self.encoder_cross_attn_layers[i](query=ca_norm_input, key=cond_kv, value=cond_kv)
            x = x + self.dropout_layers[i](ca_output)
            ffn_norm_input = self.encoder_norm2_layers[i](x)
            ffn_output = self.encoder_ffn_layers[i](ffn_norm_input)
            x = x + ffn_output
            current_input = x

        transformer_output = current_input
        output_logits = self.output_layer(transformer_output)
        return output_logits


class StableLatentDenoiser(nn.Module):
    """
    Simple latent-space denoiser that predicts additive noise on z_t.
    It projects the latent to an embedding, adds a time embedding, performs
    cross-attention with `cond_kv` (B, cond_seq_len, embed_dim) and returns
    a predicted noise vector of shape (B, latent_dim).
    """
    def __init__(self, latent_dim, embed_dim, num_heads=8, num_layers=2, dropout=0.1, cond_seq_len=16):
        super().__init__()
        self.latent_dim = latent_dim
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.num_layers = num_layers
        self.cond_seq_len = cond_seq_len

        # time embedding -> embed_dim
        self.timestep_embedding = nn.Sequential(
            TimestepEmbedding(embed_dim),
            nn.Linear(embed_dim, embed_dim),
            nn.SiLU(),
            nn.Linear(embed_dim, embed_dim),
        )

        # project latent -> embed_dim
        self.latent_in = nn.Linear(latent_dim, embed_dim)
        self.latent_out = nn.Linear(embed_dim, latent_dim)

        # single-token self-attention (trivial for single token but kept for extensibility)
        self.self_attn_layers = nn.ModuleList()
        self.cross_attn_layers = nn.ModuleList()
        self.ffn_layers = nn.ModuleList()
        self.norm1 = nn.ModuleList()
        self.norm2 = nn.ModuleList()
        self.dropout_layers = nn.ModuleList()

        for _ in range(num_layers):
            self.self_attn_layers.append(nn.MultiheadAttention(embed_dim, num_heads, dropout=dropout, batch_first=True))
            self.cross_attn_layers.append(nn.MultiheadAttention(embed_dim, num_heads, dropout=dropout, batch_first=True))
            self.ffn_layers.append(nn.Sequential(nn.Linear(embed_dim, embed_dim * 4), nn.GELU(), nn.Dropout(dropout), nn.Linear(embed_dim * 4, embed_dim)))
            self.norm1.append(nn.LayerNorm(embed_dim))
            self.norm2.append(nn.LayerNorm(embed_dim))
            self.dropout_layers.append(nn.Dropout(dropout))

        # initialization
        try:
            nn.init.normal_(self.latent_in.weight, mean=0.0, std=0.02)
            nn.init.normal_(self.latent_out.weight, mean=0.0, std=0.02)
        except Exception:
            pass

    def forward(self, z_t, t, cond_kv):
        """z_t: (B, latent_dim), t: (B,) long, cond_kv: (B, cond_seq_len, embed_dim)
        returns predicted_noise (B, latent_dim)
        """
        B = z_t.size(0)
        device = z_t.device

        h = self.latent_in(z_t)  # (B, embed_dim)
        time_emb = self.timestep_embedding(t)  # (B, embed_dim)
        h = h + time_emb

        # make sequence dim for attention: (B, 1, embed_dim)
        h = h.unsqueeze(1)

        padding_mask = None

        for i in range(self.num_layers):
            # self-attention (single token)
            h_norm = self.norm1[i](h.squeeze(1)).unsqueeze(1)
            sa_out, _ = self.self_attn_layers[i](query=h_norm, key=h_norm, value=h_norm)
            h = h + self.dropout_layers[i](sa_out)

            # cross-attention with condition KV
            ca_norm = self.norm2[i](h.squeeze(1)).unsqueeze(1)
            ca_out, _ = self.cross_attn_layers[i](query=ca_norm, key=cond_kv, value=cond_kv)
            h = h + self.dropout_layers[i](ca_out)

            # FFN
            ffn_in = h.squeeze(1)
            ffn_out = self.ffn_layers[i](ffn_in)
            h = h + ffn_out.unsqueeze(1)

        h = h.squeeze(1)  # (B, embed_dim)
        pred_noise = self.latent_out(h)
        return pred_noise
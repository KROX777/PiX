# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

from logging import getLogger
import os
import torch
from .embedders import LinearPointEmbedder
from .transformer import SNIP_TransformerModel, TransformerModel, SNIP_E2E_MAP
from .diffusion import SNIPConditionProjection, ConditionalD3PMTransformer, StableLatentDenoiser


logger = getLogger()


def check_model_params(params):
    """
    Check models parameters.
    """
    # model dimensions
    assert params.enc_emb_dim % params.n_enc_heads == 0
    assert params.dec_emb_dim % params.n_dec_heads == 0

    # reload a pretrained model
    if params.reload_model != "":
        print("Reloading model from ", params.reload_model)
        assert os.path.isfile(params.reload_model)


def build_modules(env, params):
    """
    Build modules.
    """
    modules = {}
    # Diffusion-mode: construct snip_encoder, condition_projector and conditional_transformer
    if getattr(params, 'use_diffusion', False):
        # Build a SNIP encoder wrapper that contains the embedder + encoder_y and manages caching.
        class SNIPEncoderWrapper(torch.nn.Module):
            def __init__(self, params, env):
                super().__init__()
                self.embedder = LinearPointEmbedder(params, env)
                self.encoder_y = SNIP_TransformerModel(
                    params,
                    env.float_id2word,
                    is_encoder=True,
                    with_output=False,
                    use_prior_embeddings=True,
                    positional_embeddings=params.enc_positional_embeddings,
                )
                self.use_zrep_cache = getattr(params, 'use_zrep_cache', True)
                # cpu cache mapping key -> cpu tensor
                self.zrep_cache = {}

            def freeze(self):
                for p in self.embedder.parameters():
                    p.requires_grad = False
                for p in self.encoder_y.parameters():
                    p.requires_grad = False

            def encode_from_samples(self, samples, env, device=None):
                # samples is the same dict returned by env.create_train_iterator
                x_to_fit = samples['x_to_fit']
                y_to_fit = samples['y_to_fit']
                # build nested list expected by embedder
                x1 = []
                for seq_id in range(len(x_to_fit)):
                    x1.append([])
                    for seq_l in range(len(x_to_fit[seq_id])):
                        x1[seq_id].append([x_to_fit[seq_id][seq_l], y_to_fit[seq_id][seq_l]])

                # embedder returns (encoded, lengths)
                x1_enc, len1 = self.embedder(x1)

                # determine device
                try:
                    dev = len1.device if isinstance(len1, torch.Tensor) else (device if device is not None else torch.device('cuda' if torch.cuda.is_available() else 'cpu'))
                except Exception:
                    dev = device if device is not None else torch.device('cuda' if torch.cuda.is_available() else 'cpu')

                # Build cache keys from samples if available
                tree_encoded_batch = samples.get('tree_encoded', None)
                keys = []
                if tree_encoded_batch is not None:
                    for t in tree_encoded_batch:
                        try:
                            k = tuple(t) if isinstance(t, (list, tuple)) else str(t)
                        except Exception:
                            k = str(t)
                        keys.append(k)
                else:
                    for i in range(len(x1)):
                        keys.append(f'idx_{i}')

                # compute or fetch per-sample z_reps
                if self.use_zrep_cache:
                    missing_idx = [i for i, k in enumerate(keys) if k not in self.zrep_cache]
                    if len(missing_idx) == 0:
                        z_list = [self.zrep_cache[k].to(dev) for k in keys]
                        encoded_y = torch.stack(z_list, dim=0)
                    else:
                        if len(missing_idx) < len(x1):
                            x1_missing = [x1[i] for i in missing_idx]
                            x1_missing_enc, len1_missing = self.embedder(x1_missing)
                            z_missing = self.encoder_y('fwd', x=x1_missing_enc, lengths=len1_missing, causal=False)
                            for idx_pos, i in enumerate(missing_idx):
                                k = keys[i]
                                self.zrep_cache[k] = z_missing[idx_pos].detach().cpu()
                            z_list = [self.zrep_cache[k].to(dev) for k in keys]
                            encoded_y = torch.stack(z_list, dim=0)
                        else:
                            encoded_y = self.encoder_y('fwd', x=x1_enc, lengths=len1, causal=False)
                            for i, k in enumerate(keys):
                                self.zrep_cache[k] = encoded_y[i].detach().cpu()
                else:
                    encoded_y = self.encoder_y('fwd', x=x1_enc, lengths=len1, causal=False)

                return encoded_y

        # instantiate wrapper
        modules['snip_encoder'] = SNIPEncoderWrapper(params, env)
        # optionally freeze encoder weights
        if getattr(params, 'freeze_encoder', True):
            modules['snip_encoder'].freeze()

        # condition projector (separate module)
        snip_latent_dim = getattr(params, 'latent_dim', None)
        embed_dim = getattr(params, 'dec_emb_dim', getattr(params, 'enc_emb_dim', 512))
        cond_seq_len = getattr(params, 'cond_seq_len', 16)
        modules['condition_projector'] = SNIPConditionProjection(snip_latent_dim if snip_latent_dim is not None else getattr(params, 'condition_feature_dim', 3), embed_dim, cond_seq_len=cond_seq_len)

        # conditional transformer (denoiser). It will expect a projected condition (B, cond_seq_len, embed_dim)
        num_timesteps = getattr(params, 'diffusion_num_timesteps', 1000)
        modules['conditional_transformer'] = ConditionalD3PMTransformer(
            vocab_size=env.n_words,
            embed_dim=embed_dim,
            num_heads=getattr(params, 'n_dec_heads', 8),
            num_layers=getattr(params, 'n_dec_layers', 8),
            dim_feedforward=getattr(params, 'dim_feedforward', embed_dim * 4),
            seq_len=getattr(params, 'max_target_len', 200),
            condition_feature_dim=getattr(params, 'condition_feature_dim', 3),
            num_timesteps=num_timesteps,
            dropout=getattr(params, 'dropout', 0.1),
            cond_seq_len=cond_seq_len,
            snip_latent_dim=snip_latent_dim,
            use_external_condition_proj=True,
        )
        # move modules to GPU if requested
        if not params.cpu:
            for k in ['snip_encoder', 'condition_projector', 'conditional_transformer']:
                modules[k].cuda()

        # Expose legacy keys so Trainer.reload_model can find embedder/encoder_y
        # (they point into the SNIPEncoderWrapper submodules)
        modules['embedder'] = modules['snip_encoder'].embedder
        modules['encoder_y'] = modules['snip_encoder'].encoder_y

        # If requested, register stable-diffusion latent pipeline modules
        if getattr(params, 'use_stable_diffusion', False):
            # SNIP encoder_f: encodes whole expression to a latent vector
            modules['encoder_f'] = SNIP_TransformerModel(
                params,
                env.equation_id2word,
                is_encoder=True,
                with_output=False,
                use_prior_embeddings=False,
                positional_embeddings=params.enc_positional_embeddings,
            )
            # Freeze encoder_f by default
            if getattr(params, 'freeze_encoder', True):
                for p in modules['encoder_f'].parameters():
                    p.requires_grad = False

            # latent denoiser (operates on z vectors and cross-attends to projected cond)
            snip_latent_dim = getattr(params, 'latent_dim', None)
            denoiser_embed = params.denoiser_embed_dim if getattr(params, 'denoiser_embed_dim', None) is not None else getattr(params, 'enc_emb_dim', 512)
            modules['stable_denoiser'] = StableLatentDenoiser(
                latent_dim=snip_latent_dim,
                embed_dim=denoiser_embed,
                num_heads=getattr(params, 'n_dec_heads', 8),
                num_layers=getattr(params, 'stable_denoiser_layers', 2),
                dropout=getattr(params, 'dropout', 0.1),
                cond_seq_len=getattr(params, 'cond_seq_len', 16),
            )

            # Ensure a mapper (SNIP_E2E_MAP) and a decoder exist for mapping latent->decoder
            modules['mapper'] = SNIP_E2E_MAP(params)
            # Reuse the common 'decoder' module name so Trainer can always
            # refer to self.modules['decoder'] in both stable and non-stable paths.
            modules['decoder'] = TransformerModel(
                params,
                env.equation_id2word,
                is_encoder=False,
                with_output=True,
                use_prior_embeddings=False,
                positional_embeddings=params.dec_positional_embeddings,
            )

            # Freeze decoder by default when using stable diffusion
            if getattr(params, 'freeze_decoder', True):
                for p in modules['decoder'].parameters():
                    p.requires_grad = False

    # --- legacy encoder-decoder build (unchanged) ---
    modules["embedder"] = LinearPointEmbedder(params, env)
    env.get_length_after_batching = modules["embedder"].get_length_after_batching

    modules["encoder_y"] = SNIP_TransformerModel(
        params,
        env.float_id2word,
        is_encoder=True,
        with_output=False,
        use_prior_embeddings=True,
        positional_embeddings=params.enc_positional_embeddings,
    )
    # modules["encoder_f"] = SNIP_TransformerModel(
    #     params,
    #     env.equation_id2word,
    #     is_encoder=True,
    #     with_output=False,
    #     use_prior_embeddings=False,
    #     positional_embeddings=params.enc_positional_embeddings,
    # )
    # If stable diffusion already created a decoder / mapper above, avoid
    # overwriting them here.
    if "decoder" not in modules:
        modules["decoder"] = TransformerModel(
            params,
            env.equation_id2word,
            is_encoder=False,
            with_output=True,
            use_prior_embeddings=False,
            positional_embeddings=params.dec_positional_embeddings,
        )

    if "mapper" not in modules:
        modules["mapper"] = SNIP_E2E_MAP(
            params,
        )

    # reload pretrained modules
    if params.reload_model != "":
        logger.info(f"Reloading modules from {params.reload_model} ...")
        reloaded = torch.load(params.reload_model)
        for k, v in modules.items():
            assert k in reloaded
            if all([k2.startswith("module.") for k2 in reloaded[k].keys()]):
                reloaded[k] = {
                    k2[len("module.") :]: v2 for k2, v2 in reloaded[k].items()
                }
            v.load_state_dict(reloaded[k])

    # log
    for k, v in modules.items():
        logger.debug(f"{v}: {v}")
    for k, v in modules.items():
        logger.info(
            f"Number of parameters ({k}): {sum([p.numel() for p in v.parameters() if p.requires_grad])}"
        )

    # cuda
    if not params.cpu:
        for v in modules.values():
            v.cuda()

    return modules

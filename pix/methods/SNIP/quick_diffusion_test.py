import os
import json
import torch
import numpy as np
from parsers import get_parser
from symbolicregression.envs import build_env
from symbolicregression.model import build_modules


def reload_modules_from_checkpoint(modules, path):
    assert os.path.isfile(path), f"Checkpoint not found: {path}"
    data = torch.load(path, map_location=torch.device('cpu'))
    for k, v in modules.items():
        if k in data:
            try:
                v.load_state_dict(data[k])
                print(f"Loaded weights for module: {k}")
            except RuntimeError:
                # try stripping 'module.' prefix
                stripped = {name.partition('.')[-1]: w for name, w in data[k].items()}
                v.load_state_dict(stripped)
                print(f"Loaded (stripped) weights for module: {k}")
        else:
            print(f"Warning: module {k} not found in checkpoint. Skipping.")
    return modules


def make_xy_sequence_from_linear(a, b, n_points=200):
    xs = np.linspace(-3.0, 3.0, n_points)
    seq = []
    for x in xs:
        x_arr = np.array([x], dtype=float)
        y_arr = np.array([a * x * x + b], dtype=float)
        seq.append((x_arr, y_arr))
    return seq


def main():
    parser = get_parser()
    params = parser.parse_args([])
    # diffusion config
    params.use_diffusion = True
    params.cpu = True
    params.batch_size = 1
    params.eval_only = True
    params.reload_model = ""  # not used for diffusion reload
    params.reload_model_snipenc = "./weights/snip-10dmax.pth"
    params.reload_model_e2edec = "./weights/e2e.pth"
    params.max_input_dimension = 10
    params.diffusion_num_timesteps = 100
    params.diffusion_schedule_type = 'cosine'
    params.env_name = 'functions'
    params.latent_dim = 512
    params.n_prediction_points = 30
    params.max_target_len = 200
    params.float_precision = 3
    params.mantissa_len = 1

    # build env and modules
    env = build_env(params)
    print('Env n_words:', env.n_words)
    modules = build_modules(env, params)
    print('Modules:', list(modules.keys()))

    # load checkpoint containing diffusion modules
    ckpt_path = '/sfs/xcy/discdiff/run-test/periodic-110.pth'
    assert os.path.isfile(ckpt_path), f'Checkpoint not found: {ckpt_path}'
    data = torch.load(ckpt_path, map_location=torch.device('cpu'))

    # Try to load module state dicts into matching modules when present
    for name, module in modules.items():
        if name in data:
            try:
                module.load_state_dict(data[name])
                print(f'Loaded {name} from checkpoint')
            except Exception as e:
                try:
                    stripped = {k.partition('.')[-1]: v for k, v in data[name].items()}
                    module.load_state_dict(stripped)
                    print(f'Loaded (stripped) {name} from checkpoint')
                except Exception as e2:
                    print(f'Failed loading {name}:', e2)
        else:
            print(f'Checkpoint does not contain module: {name} (skipping)')

    # move modules to device and set eval
    device = torch.device('cpu')
    for m in modules.values():
        try:
            m.to(device)
            m.eval()
        except Exception:
            pass

    # required modules
    snip_encoder = modules.get('snip_encoder')
    cond_proj = modules.get('condition_projector')
    denoiser = modules.get('conditional_transformer')
    if snip_encoder is None or cond_proj is None or denoiser is None:
        raise RuntimeError('Missing diffusion modules in built modules')

    # build synthetic samples dict for SNIP encoder
    n_points = getattr(params, 'n_prediction_points', None) or 200
    seq = make_xy_sequence_from_linear(a=3.0, b=8.0, n_points=n_points)
    samples = {
        'x_to_fit': [ [p[0] for p in seq] ],
        'y_to_fit': [ [p[1] for p in seq] ],
        # leave tree_encoded absent; encoder wrapper will key by idx
    }

    with torch.no_grad():
        encoded_y = snip_encoder.encode_from_samples(samples, env, device=device)  # (B, latent_dim)
        cond_kv = cond_proj(encoded_y)  # (B, cond_seq_len, embed_dim)

        batch_size = 1
        seq_len = getattr(params, 'max_target_len', 200)
        diffusion = modules.get('conditional_transformer')
        # conditional_transformer is the denoiser model; use diffusion scheduler to sample
        from symbolicregression.diffusion_scheduler import DiscreteDiffusion
        diff = DiscreteDiffusion(num_timesteps=params.diffusion_num_timesteps, vocab_size=env.n_words, device='cpu', schedule_type=params.diffusion_schedule_type)

        print('Starting sampling...')
        samples_tokens = diff.sample(denoiser, cond_kv, (batch_size, seq_len))

    # convert tokens to infix strings
    toks = samples_tokens.cpu().numpy()
    outs = []

    def ids_to_words_collapsed(ids, env):
        """Convert a list of token ids to word tokens, collapsing float-token sequences
        (sign + N... + E...) into a single numeric string using env.float_encoder.decode.
        """
        words = []
        i = 0
        mantissa_len = env.float_encoder.mantissa_len
        chunk_len = 2 + mantissa_len
        while i < len(ids):
            wid = int(ids[i])
            w = env.equation_id2word.get(wid, str(wid))
            # try to decode float sequence starting with '+' or '-'
            if w in ['+', '-'] and i + chunk_len <= len(ids):
                slice_words = [env.equation_id2word.get(int(ids[j]), str(ids[j])) for j in range(i, i + chunk_len)]
                try:
                    val_list = env.float_encoder.decode(slice_words)
                except Exception:
                    val_list = None
                if val_list is not None and len(val_list) > 0 and (not (val_list[0] is None)):
                    # format float in a compact scientific notation
                    try:
                        words.append("{:.{}e}".format(float(val_list[0]), env.float_precision))
                    except Exception:
                        words.append(str(val_list[0]))
                    i += chunk_len
                    continue
            # otherwise append single token word
            words.append(w)
            i += 1
        return words

    for b in range(toks.shape[0]):
        seq_list = toks[b].tolist()
        # trim at first EOS if present
        if 0 in seq_list:
            eos_idx = seq_list.index(0)
            seq_trim = seq_list[:eos_idx+1]
        else:
            seq_trim = seq_list

        # build readable token words, collapsing float encodings
        words = ids_to_words_collapsed(seq_trim, env)

        # try to decode to infix using the environment; fall back to words list
        try:
            infix = env.word_to_infix(words, is_float=False, str_array=False)
        except Exception:
            infix = None

        outs.append({'tokens': words, 'infix': infix})

    out_path = '/sfs/xcy/quick_diffusion_out.jsonl'
    with open(out_path, 'w', encoding='utf-8') as fh:
        for o in outs:
            fh.write(json.dumps(o, ensure_ascii=False) + '\n')

    print('Wrote outputs to', out_path)
    print('Results:')
    for o in outs:
        print(o)


if __name__ == '__main__':
    main()

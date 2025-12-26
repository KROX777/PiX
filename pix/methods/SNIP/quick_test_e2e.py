import os
import torch
import numpy as np
from parsers import get_parser
from symbolicregression.envs import build_env
from symbolicregression.model import build_modules
from symbolicregression.model.model_wrapper import ModelWrapper


def reload_modules_from_checkpoint(modules, path):
    assert os.path.isfile(path), f"Checkpoint not found: {path}"
    data = torch.load(path, map_location=torch.device('cpu'))
    # Expect checkpoint to be a dict with module state_dicts under module names
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


def make_xy_sequence_from_linear(a=3.0, b=2.0, n_points=30):
    xs = np.linspace(-3.0, 3.0, n_points)
    seq = []
    for x in xs:
        # embedder expects per-point arrays (possibly multidim)
        x_arr = np.array([x], dtype=float)
        y_arr = np.array([a * x * x + b], dtype=float)
        seq.append((x_arr, y_arr))
    return seq


def main():
    parser = get_parser()
    params = parser.parse_args([])
    # minimal config for non-diffusion E2E test
    params.use_diffusion = False
    params.cpu = True  # run on CPU for quick test
    params.batch_size = 1
    params.eval_only = True
    params.reload_model = "./weights/snip-e2e-sr.pth"  # path to your checkpoint
    params.reload_model_snipenc = ""  # avoid double-loading
    params.reload_model_e2edec = ""
    # Match training hyperparameters used to produce the checkpoint
    params.max_input_dimension = 10

    # build env and modules
    env = build_env(params)
    print('Env n_words:', env.n_words)
    modules = build_modules(env, params)
    print('Modules:', list(modules.keys()))

    # reload checkpoint modules into our modules
    ckpt_path = params.reload_model
    modules = reload_modules_from_checkpoint(modules, ckpt_path)

    # move modules to CPU device explicitly
    for m in modules.values():
        m.to(torch.device('cpu'))
        m.eval()

    # prepare modules
    embedder = modules.get('embedder')
    encoder = modules.get('encoder_y')
    decoder = modules.get('decoder')
    mapper = modules.get('mapper')
    if embedder is None or encoder is None or decoder is None or mapper is None:
        raise RuntimeError('Missing embedder/encoder/decoder/mapper modules for E2E test')

    # create synthetic data y = 3*x + 2
    seq = make_xy_sequence_from_linear(a=3.0, b=8.0, n_points=30)
    batch_input = [seq]  # batch of one

    # embed and encode
    with torch.no_grad():
        x_enc, x_len = embedder(batch_input)
        # encoder returns latent z_rep (B, latent_dim)
        z_rep = encoder('fwd', x=x_enc, lengths=x_len, causal=False)
        # map latent to decoder src_enc (B, seq_len, dec_emb_dim)
        src_enc = mapper(z_rep)

        # generate using decoder from latent-derived src_enc
        generations, gen_len = decoder.generate_from_latent(src_enc, max_len=200)

    # convert generated token ids to human-readable infix strings
    generations = generations.unsqueeze(-1).view(generations.shape[0], generations.shape[1], 1)
    generations = generations.transpose(0, 1).transpose(1, 2).cpu().tolist()
    outputs = [
        list(
            filter(
                lambda x: x is not None,
                [env.idx_to_infix(hyp[1:-1], is_float=False, str_array=False) for hyp in generations[i]],
            )
        )
        for i in range(len(generations))
    ]

    print('Generated candidates:')
    for i, cand_list in enumerate(outputs):
        print(f'Example {i}:')
        for cand in cand_list:
            print('  ', cand)


if __name__ == '__main__':
    main()

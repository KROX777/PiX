#!/usr/bin/env python3
"""
Compute SNIP `z_rep` latents for a dataset using the exact SNIP preprocessing
pipeline (embedder -> encoder_y("fwd")).

Saves a single `.npz` file containing `z_reps` shaped (N, latent_dim).

Usage example:
  python examples/compute_zrep.py --data data.jsonl --out zrep.npz --batch_size 32 --device cpu

This script mirrors the steps in `symbolicregression/trainer.py:enc_dec_step`.
"""

import argparse
import json
import os
from pathlib import Path
import numpy as np
import torch

from parsers import get_parser
from symbolicregression.envs import build_env
from symbolicregression.model import build_modules
from symbolicregression.utils import to_cuda, CUDA


def str_list_to_float_array(lst):
    for i in range(len(lst)):
        for j in range(len(lst[i])):
            lst[i][j] = float(lst[i][j])
    return lst


def read_jsonl(path):
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            if not line.strip():
                continue
            yield json.loads(line.rstrip())


def main():
    parser = argparse.ArgumentParser("compute_zrep", add_help=True)
    base = get_parser()
    # inherit all known args from base parser (for model/env params)
    for action in base._actions:
        # skip help duplicate
        if action.option_strings:
            parser.add_argument(*action.option_strings, **{k: v for k, v in vars(action).items() if k in ['dest','default','type','help','choices'] and v is not None})

    parser.add_argument('--data', type=str, required=True, help='Path to input jsonl (same format as dataset)')
    parser.add_argument('--out', type=str, required=True, help='Output .npz path to save z_reps')
    parser.add_argument('--device', type=str, default='cuda', help='Device to run on, e.g. cpu or cuda:0')

    args = parser.parse_args()
    params = args

    # device / CUDA flag
    if args.device.startswith('cuda') or args.device == 'cuda':
        params.cpu = False
        CUDA = True
    else:
        params.cpu = True
        CUDA = False

    # build environment and modules
    env = build_env(params)
    modules = build_modules(env, params)

    embedder = modules['embedder']
    encoder_y = modules['encoder_y']

    # optionally reload pretrained encoder/embedder weights
    if args.reload_model_snipenc:
        ckpt = torch.load(args.reload_model_snipenc, map_location='cpu')
        for k in ['embedder', 'encoder_y']:
            if k in ckpt:
                try:
                    modules[k].load_state_dict(ckpt[k])
                    print(f"Loaded weights for {k} from {args.reload_model_snipenc}")
                except Exception:
                    # try to strip 'module.' prefix
                    stripped = {name.partition('module.')[2]: v for name, v in ckpt[k].items()}
                    modules[k].load_state_dict(stripped)
                    print(f"Loaded (stripped) weights for {k} from {args.reload_model_snipenc}")

    # move to device
    device = torch.device(args.device)
    if not params.cpu:
        for m in modules.values():
            try:
                m.to(device)
            except Exception:
                pass

    # read dataset lines
    samples = list(read_jsonl(args.data))
    n = len(samples)
    print(f"Found {n} samples in {args.data}")

    z_list = []

    batch_size = args.batch_size
    for start in range(0, n, batch_size):
        batch = samples[start:start+batch_size]
        x1_list = []
        for x in batch:
            # Convert strings to floats when necessary (same as EnvDataset.read_sample)
            x_to_fit = x.get('x_to_fit', [])
            y_to_fit = x.get('y_to_fit', [])
            x_to_fit = [list(map(float, row)) for row in x_to_fit]
            y_to_fit = [list(map(float, row)) for row in y_to_fit]

            seq = []
            for i in range(len(x_to_fit)):
                seq.append([x_to_fit[i], y_to_fit[i]])
            x1_list.append(seq)

        # embedder expects list-of-seqs -> returns (tensor, lengths)
        x1_tensor, len1 = embedder(x1_list)

        if not params.cpu:
            x1_tensor, len1 = to_cuda(x1_tensor, len1)

        # encoder forward (canonical SNIP preprocessing)
        with torch.no_grad():
            encoded_y = encoder_y('fwd', x=x1_tensor, lengths=len1, causal=False)

        z_np = encoded_y.detach().cpu().numpy()
        z_list.append(z_np)

    z_reps = np.concatenate(z_list, axis=0)
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out_path, z_reps=z_reps)
    print(f"Saved {z_reps.shape} z_reps to {out_path}")


if __name__ == '__main__':
    main()

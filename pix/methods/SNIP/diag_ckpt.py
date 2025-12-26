import torch
from parsers import get_parser
import symbolicregression
from symbolicregression.envs import build_env
from symbolicregression.model import build_modules

ckpt_path = "/home/ma-user/work/srbench/algorithms/snip/weights/snip-10dmax.pth"
print("Checkpoint path:", ckpt_path)
ckpt = torch.load(ckpt_path, map_location='cpu')
print("Top-level keys:", list(ckpt.keys()))

if 'embedder' in ckpt:
    print('\n-- checkpoint embedder keys/shapes --')
    for k,v in ckpt['embedder'].items():
        try:
            print(k, tuple(v.shape))
        except Exception:
            print(k, type(v))
else:
    print('\n-- no embedder top-key in checkpoint --')

# instantiate current embedder
parser = get_parser()
args = parser.parse_args([])
args.cpu = True
args.reload_data = ""
args.batch_size = 1
args.max_src_len = 200
args.max_target_len = 200
env = build_env(args)
modules = build_modules(env, args)
embedder = modules['embedder']
print('\nCurrent embedder class:', embedder.__class__)
print('-- current embedder state_dict keys/shapes --')
for k,v in embedder.state_dict().items():
    print(k, tuple(v.shape))

# comparisons
ck_keys = set(ckpt.get('embedder', {}).keys())
cur_keys = set(embedder.state_dict().keys())
print('\nMissing in checkpoint (present in model):', sorted(list(cur_keys-ck_keys)) )
print('Unexpected in checkpoint (not in model):', sorted(list(ck_keys-cur_keys)) )

print('\nShape mismatches:')
for k in sorted(list(cur_keys & ck_keys)):
    s_cur = tuple(embedder.state_dict()[k].shape)
    s_ck = tuple(ckpt['embedder'][k].shape)
    if s_cur != s_ck:
        print(k, 'ckpt', s_ck, 'current', s_cur)

# print some current parser values
print('\nCurrent parser embedder args:')
for a in ['emb_emb_dim','n_emb_layers','emb_expansion_factor','enc_emb_dim','latent_dim','emb_emb_dim']:
    print(' ', a, getattr(args,a,None))

# if checkpoint has saved params print them
for candidate in ['params','args','config']:
    if candidate in ckpt:
        print(f"\nCheckpoint had top-level '{candidate}'; sample keys/values:")
        try:
            d=ckpt[candidate]
            if isinstance(d, dict):
                for key in ['emb_emb_dim','n_emb_layers','emb_expansion_factor','enc_emb_dim','latent_dim']:
                    if key in d:
                        print(' ', key, d[key])
                print('  (first 20 keys):', list(d.keys())[:20])
            else:
                print(' ', type(d), d)
        except Exception as e:
            print('  error reading', e)

print('\nDone')

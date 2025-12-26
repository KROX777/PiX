import torch
pth = "/home/ma-user/work/srbench/algorithms/snip/weights/snip-10dmax.pth"
ckpt = torch.load(pth, map_location="cpu")
print("checkpoint type:", type(ckpt))
if isinstance(ckpt, dict):
    if "encoder_f" in ckpt:
        sd = ckpt["encoder_f"]
        if isinstance(sd, dict):
            total = sum(v.numel() for v in sd.values())
            print("encoder_f: state_dict entries =", len(sd))
            print("encoder_f: total parameters =", total)
            print("sample keys:", list(sd.keys())[:20])
        else:
            print("encoder_f exists but is not a dict (type:", type(sd), ")")
    else:
        print("'encoder_f' not found in checkpoint. Available keys:\n", list(ckpt.keys())[:200])
else:
    # checkpoint saved as object
    attrs = list(getattr(ckpt, '__dict__', {}).keys())
    print('checkpoint is object; attrs:', attrs[:200])
    if hasattr(ckpt, 'encoder_f'):
        obj = getattr(ckpt, 'encoder_f')
        try:
            sd = obj.state_dict() if hasattr(obj, 'state_dict') else obj
            if isinstance(sd, dict):
                total = sum(v.numel() for v in sd.values())
                print("encoder_f (object).state_dict entries =", len(sd))
                print("encoder_f (object).total parameters =", total)
        except Exception as e:
            print('Could not get encoder_f params from object:', e)

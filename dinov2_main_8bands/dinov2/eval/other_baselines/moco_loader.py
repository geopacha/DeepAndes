import os
import torch
import torch.nn as nn

def _inflate_conv_weight(w3, new_in_chans, mode="mean"):
    out, old_in, k1, k2 = w3.shape
    assert old_in == 3
    if mode == "mean":
        w_mean = w3.mean(dim=1, keepdim=True)
        w_new = w_mean.repeat(1, new_in_chans, 1, 1)
    elif mode == "repeat":
        reps = (new_in_chans + old_in - 1) // old_in
        w_new = w3.repeat(1, reps, 1, 1)[:, :new_in_chans, :, :]
        w_new *= old_in / float(new_in_chans)
    return w_new

def _adapt_first_layer(state_dict, model, new_in_chans=8, mode="mean"):
    conv_keys = []
    if hasattr(model, "conv1") and isinstance(getattr(model, "conv1"), nn.Conv2d):
        conv_keys.append("conv1.weight")
    if hasattr(model, "patch_embed") and hasattr(model.patch_embed, "proj"):
        conv_keys.append("patch_embed.proj.weight")
    for k in conv_keys:
        if k in state_dict:
            w = state_dict[k]
            if w.ndim == 4 and w.shape[1] == 3 and new_in_chans != 3:
                state_dict[k] = _inflate_conv_weight(w, new_in_chans, mode)
    return state_dict

def load_moco_backbone(model, ckpt_path, in_chans=8, mode="mean"):
    if not (ckpt_path and os.path.isfile(ckpt_path)):
        print(f"=> no checkpoint found at '{ckpt_path}'")
        return model
    print(f"=> loading checkpoint '{ckpt_path}'")
    checkpoint = torch.load(ckpt_path, map_location="cpu")
    if isinstance(checkpoint, dict):
        print("Top-level keys:", list(checkpoint.keys())[:10])
    sd = checkpoint.get("state_dict", checkpoint)
    new_sd = {}
    for k, v in sd.items():
        if k.startswith("module.encoder_q") and not k.startswith("module.encoder_q.fc"):
            nk = k[len("module.encoder_q."):]
            new_sd[nk] = v
    new_sd = _adapt_first_layer(new_sd, model, in_chans, mode)
    msg = model.load_state_dict(new_sd, strict=False)
    print("Missing keys:", msg.missing_keys)
    print("Unexpected keys:", msg.unexpected_keys)
    return model

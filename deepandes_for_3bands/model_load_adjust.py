"""
Load DeepAndes model backbone and adapt for 3-band input.

Adapts patch_embedding from 8 channels to 3 channels after loading pretrained weights.
Note: Model was designed for 8-band Worldview imagery; 3-band adaptation is experimental.
"""


import torch 
import torch.nn as nn 


def adapt_patch_embed_in_chans(model, new_in_ch=3, mode="repeat_scaled"):
    """
    Make model.patch_embed.proj accept `new_in_ch` by inflating pretrained RGB weights.
    mode:
      - 'repeat_scaled': repeat RGB across bands and scale by (3 / new_in_ch)
      - 'mean': average RGB, copy to all bands
    """
    pe = model.patch_embed.proj                      # Conv2d [E, old_in , p, p]
    W = pe.weight.data
    E, old_in, p1, p2 = W.shape
    assert old_in == 8, f"Expected 8 input channels, got {old_in}"

    if mode == "mean":
        W_new = W.mean(dim=1, keepdim=True).repeat(1, new_in_ch, 1, 1)
    elif mode == "repeat_scaled":
        reps = (new_in_ch + old_in - 1) // old_in
        W_new = W.repeat(1, reps, 1, 1)[:, :new_in_ch, :, :]
        W_new *= (old_in / float(new_in_ch))  # preserve variance roughly
    else:
        raise ValueError("mode must be 'repeat_scaled' or 'mean'")

    new_conv = nn.Conv2d(
        in_channels=new_in_ch, out_channels=E,
        kernel_size=(p1, p2), stride=pe.stride,
        padding=pe.padding, dilation=pe.dilation,
        groups=pe.groups, bias=(pe.bias is not None)
    )
    new_conv.weight = nn.Parameter(W_new)
    if pe.bias is not None:
        new_conv.bias = nn.Parameter(pe.bias.data.clone())

    model.patch_embed.proj = new_conv
    if hasattr(model.patch_embed, "in_chans"):
        model.patch_embed.in_chans = new_in_ch
    return model



pretrained_weight = '/home/guoj5/checkpoints/DeepAndes/teacher_checkpoint.pth'  

# 1. Load deepandes using torch hub method  
model = torch.hub.load('facebookresearch/dinov2', 'dinov2_vitl14')
teacher_checkpoint = pretrained_weight # path to .pth 
pretrained_dict = torch.load(teacher_checkpoint)
checkpoint_key = 'teacher'
new_state_dict = {}
for k, v in pretrained_dict[checkpoint_key].items():
    if 'dino_head' in k:
        print(f'{k} not used')

    elif 'ibot_head' in k:
        print(f'{k} not used')
    else:
        new_key = k.replace('backbone.', '')
        new_state_dict[new_key] = v
#change shape of pos_embed, shape depending on vits or vitg, or  vitl
pos_embed = nn.Parameter(torch.zeros(1, 257, 1024))
model.pos_embed = pos_embed

new_patch_embed = model.patch_embed
new_patch_embed.proj = nn.Conv2d(
    in_channels=8,  # Updated for 8 input bands
    out_channels=new_patch_embed.proj.out_channels,
    kernel_size=new_patch_embed.proj.kernel_size,
    stride=new_patch_embed.proj.stride,
    padding=new_patch_embed.proj.padding,
)
model.patch_embed = new_patch_embed
model.load_state_dict(new_state_dict, strict=True)


# 2. Make model (weights loaded) for 3 band inputs. 
# mode = 'mean': average weights 
# mode = 'repeat_scaled' : repeatly copy weights 
adapt_patch_embed_in_chans(model, new_in_ch=3, mode="mean")
model.eval()

# validate modified deepandes model (patch_embedding)
print(f" [embed_dim, in_ch, vit_patch_size, vit_patch_size] : {model.patch_embed.proj.weight.shape}")
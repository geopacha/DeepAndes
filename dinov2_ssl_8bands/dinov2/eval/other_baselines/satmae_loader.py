import torch
import torch.nn as nn
from .satmae_model import MaskedAutoencoderViT


class ClassifierHead(nn.Module):
    def __init__(self, in_dim=1024, hidden_dim=256, num_classes=2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, num_classes)
        )
    def forward(self, x):
        return self.net(x)

class SatMAEForClassification(nn.Module):
    def __init__(self, base_model, num_classes=2, hidden_dim=256, freeze_decoder=True, base_embed_dim=1024):
        super().__init__()
        self.base = base_model
        in_dim = getattr(self.base, "embed_dim", base_embed_dim)
        self.head = ClassifierHead(in_dim, hidden_dim, num_classes)

        if freeze_decoder and hasattr(self.base, "decoder"):
            for p in self.base.decoder.parameters():
                p.requires_grad = False

    def forward(self, x):
        # encoder only for classification
        tokens = self.base.forward_encoder(x, mask_ratio=0.0)[0]   # (B, N+1, D)
        cls = tokens[:, 0, :]                                      # (B, D)
        logits = self.head(cls)                                    # (B, num_classes)
        return logits

class backbone_builder:
    def __init__(self):
        self.model = MaskedAutoencoderViT.from_pretrained("MVRL/satmaepp_ViT-L_pretrain_fmow_rgb")


    def adapt_patch_embed_in_chans(self, new_in_ch=8, mode="repeat_scaled"):
        """
        Make model.patch_embed.proj accept `new_in_ch` by inflating pretrained RGB weights.
        mode:
        - 'repeat_scaled': repeat RGB across bands and scale by (3 / new_in_ch)
        - 'mean': average RGB, copy to all bands
        """
        pe = self.model.patch_embed.proj                      # Conv2d [E, 3, p, p]
        W = pe.weight.data
        E, old_in, p1, p2 = W.shape
        assert old_in == 3, f"Expected 3 input channels, got {old_in}"

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

        self.model.patch_embed.proj = new_conv
        if hasattr(self.model.patch_embed, "in_chans"):
            self.model.patch_embed.in_chans = new_in_ch

# if __name__ == "__main__":
#     satmae_model = backbone_builder()
#     satmae_model.adapt_patch_embed_in_chans(new_in_ch=8, mode="mean")
#     print(satmae_model.model.patch_embed.proj)

#     satmae_classifier = SatMAEForClassification(satmae_model.model, num_classes=2, hidden_dim=256, freeze_decoder=True, base_embed_dim=1024)
#     print(satmae_classifier)
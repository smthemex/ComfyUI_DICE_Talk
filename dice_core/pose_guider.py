"""PoseGuider: per-frame 2D conv stack producing the 320-channel spatial condition
that the DICE UNet adds right after conv_in.

Ported from DICE-Talk `src/models/audio_adapter/pose_guider.py`; the only change is
that `inflated` convs are applied with a plain reshape (no einops) and all
convs run through comfy `operations` so they follow the model's dtype/offload.
"""
import torch.nn as nn
import torch.nn.functional as F


class InflatedConv3d(nn.Conv2d):
    """Conv2d applied independently to every frame of a [b,c,f,h,w] tensor.

    Mirrors the official einops `rearrange("b c f h w -> (b f) c h w")`: the frame
    axis has to be permuted next to the batch axis BEFORE the reshape, otherwise
    the channels and frames get interleaved. (With f == 1 the two orders happen to
    coincide, which is why this is easy to miss.)
    """

    def forward(self, x):
        b, c, f, h, w = x.shape
        x = x.permute(0, 2, 1, 3, 4).reshape(b * f, c, h, w)   # [(b f), c, h, w]
        x = super().forward(x)
        # stride-2 convs shrink h,w -> rebuild from the ACTUAL output shape
        out = x.reshape(b, f, *x.shape[1:])                     # [b, f, c_out, h', w']
        return out.permute(0, 2, 1, 3, 4).contiguous()          # [b, c_out, f, h', w']


def zero_module(module):
    for p in module.parameters():
        p.detach().zero_()
    return module


class PoseGuider(nn.Module):
    def __init__(self, conditioning_embedding_channels: int, conditioning_channels: int = 3,
                 block_out_channels=(16, 32, 96, 256), operations=None):
        super().__init__()
        if operations is None:
            import torch.nn as operations
        self.conv_in = InflatedConv3d(conditioning_channels, block_out_channels[0], kernel_size=3, padding=1)

        self.blocks = nn.ModuleList([])
        for i in range(len(block_out_channels) - 1):
            channel_in = block_out_channels[i]
            channel_out = block_out_channels[i + 1]
            self.blocks.append(InflatedConv3d(channel_in, channel_in, kernel_size=3, padding=1))
            self.blocks.append(InflatedConv3d(channel_in, channel_out, kernel_size=3, padding=1, stride=2))

        self.conv_out = zero_module(
            InflatedConv3d(block_out_channels[-1], conditioning_embedding_channels, kernel_size=3, padding=1)
        )

    def forward(self, conditioning):
        embedding = self.conv_in(conditioning)
        embedding = F.silu(embedding)
        for block in self.blocks:
            embedding = block(embedding)
            embedding = F.silu(embedding)
        return self.conv_out(embedding)

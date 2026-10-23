import torch
import torch.nn.functional as F
from torch import nn, Tensor
from typing import Union, Tuple


class Fold3d(nn.Module):
    def __init__(
        self,
        output_size: Tuple[int, int, int],
        kernel_size: Union[int, Tuple[int, int, int]],
        channels: int,
        dilation: Union[int, Tuple[int, int, int]] = 1,
        padding: Union[int, Tuple[int, int, int]] = 0,
        stride: Union[int, Tuple[int, int, int]] = 1,
    ):
        super().__init__()
        self.output_size = output_size
        self.kernel_size = kernel_size if isinstance(kernel_size, tuple) else (kernel_size,) * 3
        self.dilation = dilation if isinstance(dilation, tuple) else (dilation,) * 3
        self.padding = padding if isinstance(padding, tuple) else (padding,) * 3
        self.stride = stride if isinstance(stride, tuple) else (stride,) * 3
        self.channels = channels
        self._scatter_idx = None

    def _scatter_index(self, device):
        """Flat destination index for every (patch_voxel, patch) pair.

        Replicates conv_transpose3d accumulation exactly: patch l at grid
        position (i, j, k) scatters its voxel offset (a, b, c) to global voxel
        (i*s+a - p, j*s+b - p, k*s+c).
        """
        if self._scatter_idx is not None and self._scatter_idx.device == device:
            return self._scatter_idx
        kD, kH, kW = self.kernel_size
        sD, sH, sW = self.stride
        dD, dH, dW = self.dilation
        pD, pH, pW = self.padding
        D, H, W = self.output_size
        Dp = (D + 2 * pD - dD * (kD - 1) - 1) // sD + 1
        Hp = (H + 2 * pH - dH * (kH - 1) - 1) // sH + 1
        Wp = (W + 2 * pW - dW * (kW - 1) - 1) // sW + 1
        # zz[a,i] / yy[b,j] / xx[c,k]: destination coordinate of patch-voxel
        # offset (a,b,c) inside patch grid cell (i,j,k)
        zz = torch.arange(Dp).view(1, -1) * sD + torch.arange(kD).view(-1, 1) * dD - pD  # [kD, Dp]
        yy = torch.arange(Hp).view(1, -1) * sH + torch.arange(kH).view(-1, 1) * dH - pH  # [kH, Hp]
        xx = torch.arange(Wp).view(1, -1) * sW + torch.arange(kW).view(-1, 1) * dW - pW  # [kW, Wp]
        idx = (zz.view(kD, 1, 1, Dp, 1, 1) * (H * W)
               + yy.view(1, kH, 1, 1, Hp, 1) * W
               + xx.view(1, 1, kW, 1, 1, Wp))                 # [kD,kH,kW,Dp,Hp,Wp]
        idx = idx.reshape(kD * kH * kW, Dp * Hp * Wp)          # [patch_vol, L]
        self._scatter_idx = idx.reshape(-1).to(device)
        return self._scatter_idx

    def forward(self, input: Tensor) -> Tensor:
        """
        Args:
            input: Tensor of shape [N, C * kD * kH * kW, L]
        Returns:
            Tensor of shape [N, C, D_out, H_out, W_out]
        """
        N, CK, L = input.shape
        C = self.channels
        kD, kH, kW = self.kernel_size
        patch_vol = kD * kH * kW
        assert CK == C * patch_vol, f"Expected {C * patch_vol}, got {CK}"

        D_out, H_out, W_out = self.output_size

        if input.device.type == 'mps':
            # conv_transpose3d is unsupported on MPS; scatter-add is identical.
            idx = self._scatter_index(input.device)                       # [patch_vol*L]
            vals = input.view(N * C, patch_vol, L).reshape(N * C, -1)     # [N*C, patch_vol*L]
            out = torch.zeros(N * C, D_out * H_out * W_out,
                              dtype=input.dtype, device=input.device)
            out = out.index_add(1, idx, vals)
            return out.view(N, C, D_out, H_out, W_out)

        Dp = (D_out + 2 * self.padding[0] - self.dilation[0] * (kD - 1) - 1) // self.stride[0] + 1
        Hp = (H_out + 2 * self.padding[1] - self.dilation[1] * (kH - 1) - 1) // self.stride[1] + 1
        Wp = (W_out + 2 * self.padding[2] - self.dilation[2] * (kW - 1) - 1) // self.stride[2] + 1
        assert Dp * Hp * Wp == L, f"L ({L}) != expected number of patches ({Dp * Hp * Wp})"

        # Reshape to [N * C, patch_vol, Dp, Hp, Wp]
        input = input.view(N, C, patch_vol, Dp, Hp, Wp).reshape(N * C, patch_vol, Dp, Hp, Wp)

        # Construct weight: [patch_vol, 1, kD, kH, kW]
        weight = torch.eye(patch_vol, device=input.device).view(patch_vol, 1, kD, kH, kW)

        # Apply transpose conv with groups=1
        out = F.conv_transpose3d(
            input,
            weight,
            bias=None,
            stride=self.stride,
            padding=self.padding,
            dilation=self.dilation,
            groups=1
        )  # Output: [N * C, 1, D_out, H_out, W_out]

        # Reshape back to [N, C, D_out, H_out, W_out]
        out = out.view(N, C, D_out, H_out, W_out)
        return out

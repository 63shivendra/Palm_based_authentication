"""
indipalm_edge/models/rep_msfnet.py

Rep-MSFNet: Multi-Scale Spatial-Frequency Backbone for Contactless Multi-Spectral Palm Recognition.
Engineered for real-time edge inference on NXP i.MX8M Plus VeriSilicon VIP8000 NPU (2.3 TOPS).

Architecture:
  - Spatial Path: Structurally reparameterizable convolutions (RepConv2d). Multi-branch at training
    (3x3 conv, 1x1 conv, identity/BatchNorm), algebraically collapsed to a single 3x3 INT8 conv at inference.
  - Frequency Path: 2D Discrete Wavelet Transform (Haar DWT) isolating high-frequency directional subbands
    (LH: horizontal friction ridges, HL: vertical flexion creases, HH: diagonal junctions), attenuating
    smooth illumination variations in low-frequency LL.
  - Dual-Stream Multi-Spectral: Parallel RGB (surface texture) and NIR (deep vascular tree) backbones.
"""

from typing import Tuple, Optional
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from common.constants import (
    PALM_ROI_SIZE,
    EMBEDDING_DIM,
)


def conv_bn(in_channels: int, out_channels: int, kernel_size: int, stride: int = 1, padding: int = 0, groups: int = 1) -> nn.Sequential:
    """Helper to construct Conv2d + BatchNorm2d sequence."""
    return nn.Sequential(
        nn.Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            groups=groups,
            bias=False,
        ),
        nn.BatchNorm2d(num_features=out_channels),
    )


class RepConv2d(nn.Module):
    """
    Structurally Reparameterizable 2D Convolutional Block.

    During training:
        Output = Conv3x3(x) + Conv1x1(x) + Identity/BatchNorm(x)
    During inference (after calling .reparameterize()):
        Output = Conv3x3_fused(x)  # Single 3x3 convolution layer
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        stride: int = 1,
        padding: int = 1,
        dilation: int = 1,
        groups: int = 1,
        deploy: bool = False,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.groups = groups
        self.deploy = deploy

        self.act = nn.SiLU()

        if deploy:
            self.rbr_reparam = nn.Conv2d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=3,
                stride=stride,
                padding=padding,
                dilation=dilation,
                groups=groups,
                bias=True,
            )
        else:
            self.rbr_dense = conv_bn(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=3,
                stride=stride,
                padding=padding,
                groups=groups,
            )
            self.rbr_1x1 = conv_bn(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=1,
                stride=stride,
                padding=0,
                groups=groups,
            )
            self.rbr_identity = (
                nn.BatchNorm2d(num_features=in_channels)
                if out_channels == in_channels and stride == 1
                else None
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.deploy:
            return self.act(self.rbr_reparam(x))

        if self.rbr_identity is None:
            id_out = 0.0
        else:
            id_out = self.rbr_identity(x)

        return self.act(self.rbr_dense(x) + self.rbr_1x1(x) + id_out)

    def get_equivalent_kernel_bias(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Algebraically collapses Conv3x3, Conv1x1, and BatchNorm branches into
        a single 3x3 kernel and bias vector.
        """
        kernel3x3, bias3x3 = self._fuse_bn_tensor(self.rbr_dense)
        kernel1x1, bias1x1 = self._fuse_bn_tensor(self.rbr_1x1)
        kernelid, biasid = self._fuse_bn_tensor(self.rbr_identity)

        kernel = kernel3x3 + self._pad_1x1_to_3x3_tensor(kernel1x1) + kernelid
        bias = bias3x3 + bias1x1 + biasid
        return kernel, bias

    def _pad_1x1_to_3x3_tensor(self, kernel1x1: torch.Tensor) -> torch.Tensor:
        if kernel1x1 is None:
            return 0
        return F.pad(kernel1x1, [1, 1, 1, 1])

    def _fuse_bn_tensor(self, branch) -> Tuple[torch.Tensor, torch.Tensor]:
        if branch is None:
            return 0, 0
        if isinstance(branch, nn.Sequential):
            conv = branch[0]
            bn = branch[1]
            w = conv.weight
            mean = bn.running_mean
            var_sqrt = torch.sqrt(bn.running_var + bn.eps)
            gamma = bn.weight
            beta = bn.bias

            if gamma is not None and beta is not None:
                w_fused = w * (gamma / var_sqrt).reshape(-1, 1, 1, 1)
                b_fused = beta - mean * gamma / var_sqrt
            else:
                w_fused = w / var_sqrt.reshape(-1, 1, 1, 1)
                b_fused = -mean / var_sqrt
            return w_fused, b_fused

        if isinstance(branch, nn.BatchNorm2d):
            input_dim = self.in_channels // self.groups
            kernel_value = torch.zeros(
                (self.in_channels, input_dim, 3, 3),
                dtype=branch.weight.dtype,
                device=branch.weight.device,
            )
            for i in range(self.in_channels):
                kernel_value[i, i % input_dim, 1, 1] = 1.0

            mean = branch.running_mean
            var_sqrt = torch.sqrt(branch.running_var + branch.eps)
            gamma = branch.weight
            beta = branch.bias
            w_fused = kernel_value * (gamma / var_sqrt).reshape(-1, 1, 1, 1)
            b_fused = beta - mean * gamma / var_sqrt
            return w_fused, b_fused

        return 0, 0

    def reparameterize(self) -> None:
        """Transforms the module in-place to single-conv deployment mode."""
        if self.deploy:
            return
        kernel, bias = self.get_equivalent_kernel_bias()
        self.rbr_reparam = nn.Conv2d(
            in_channels=self.in_channels,
            out_channels=self.out_channels,
            kernel_size=3,
            stride=self.stride,
            padding=self.padding,
            dilation=self.dilation,
            groups=self.groups,
            bias=True,
        )
        self.rbr_reparam.weight.data = kernel
        self.rbr_reparam.bias.data = bias
        self.__delattr__("rbr_dense")
        self.__delattr__("rbr_1x1")
        if hasattr(self, "rbr_identity"):
            self.__delattr__("rbr_identity")
        self.deploy = True


class HaarDWT2d(nn.Module):
    """
    Differentiable 2D Haar Discrete Wavelet Transform.
    Decomposes an input into four sub-bands:
      - LL: Low-frequency approximation
      - LH: Horizontal high frequencies (friction ridges)
      - HL: Vertical high frequencies (flexion creases)
      - HH: Diagonal high frequencies (ridge junctions)
    """

    def __init__(self, in_channels: int):
        super().__init__()
        self.in_channels = in_channels

        # Haar wavelet basis filters (2x2 with stride 2)
        ll = torch.tensor([[0.5, 0.5], [0.5, 0.5]], dtype=torch.float32)
        lh = torch.tensor([[-0.5, -0.5], [0.5, 0.5]], dtype=torch.float32)
        hl = torch.tensor([[-0.5, 0.5], [-0.5, 0.5]], dtype=torch.float32)
        hh = torch.tensor([[0.5, -0.5], [-0.5, 0.5]], dtype=torch.float32)

        filters = torch.stack([ll, lh, hl, hh], dim=0)  # (4, 2, 2)
        # Repeat filters across input channels for depthwise convolution
        filters = filters.repeat(in_channels, 1, 1).unsqueeze(1)  # (4*C, 1, 2, 2)

        self.register_buffer("filters", filters)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        b, c, h, w = x.shape
        # Depthwise conv with stride 2
        subbands = F.conv2d(x, self.filters, stride=2, padding=0, groups=c)
        # Reshape to (B, C, 4, H/2, W/2)
        subbands = subbands.view(b, c, 4, h // 2, w // 2)

        ll = subbands[:, :, 0, :, :]
        lh = subbands[:, :, 1, :, :]
        hl = subbands[:, :, 2, :, :]
        hh = subbands[:, :, 3, :, :]
        return ll, lh, hl, hh


class SpatialFrequencyBlock(nn.Module):
    """
    Dual-branch Spatial-Frequency processing block.
      - Spatial branch: RepConv2d extracting local spatial geometric features.
      - Frequency branch: 2D Haar DWT isolating high-frequency directional subbands [LH, HL, HH],
        upsampled and projected via 1x1 Conv to match spatial dimensions.
    """

    def __init__(self, in_channels: int, out_channels: int, stride: int = 1):
        super().__init__()
        self.stride = stride

        # Spatial branch
        self.spatial_conv = RepConv2d(in_channels, out_channels, stride=stride)

        # Frequency branch
        self.dwt = HaarDWT2d(in_channels)
        # Concatenate 3 high-frequency subbands: 3 * in_channels
        self.freq_proj = nn.Sequential(
            nn.Conv2d(in_channels * 3, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.SiLU(),
        )

        # Feature mixer
        self.fusion_conv = nn.Conv2d(out_channels * 2, out_channels, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Spatial path
        f_spatial = self.spatial_conv(x)

        # Frequency path
        ll, lh, hl, hh = self.dwt(x)
        hf_concat = torch.cat([lh, hl, hh], dim=1)  # (B, 3*C, H/2, W/2)
        f_freq = self.freq_proj(hf_concat)

        # Match spatial resolution if needed
        if f_freq.shape[-2:] != f_spatial.shape[-2:]:
            f_freq = F.interpolate(f_freq, size=f_spatial.shape[-2:], mode="bilinear", align_corners=False)

        # Fuse spatial and directional frequency representations
        f_merged = torch.cat([f_spatial, f_freq], dim=1)
        out = self.fusion_conv(f_merged)
        return out


class RepMSFNetStream(nn.Module):
    """Single modality stream (RGB or NIR) of Rep-MSFNet."""

    def __init__(self, in_channels: int = 3, base_channels: int = 32):
        super().__init__()
        # Stem: 128x128 -> 64x64
        self.stem = nn.Sequential(
            RepConv2d(in_channels, base_channels, stride=2),
            RepConv2d(base_channels, base_channels, stride=1),
        )

        # Stage 1: 64x64 -> 32x32
        self.stage1 = SpatialFrequencyBlock(base_channels, base_channels * 2, stride=2)

        # Stage 2: 32x32 -> 32x32 (Preserves 32x32 resolution for Kinematic Flow Field)
        self.stage2 = nn.Sequential(
            RepConv2d(base_channels * 2, base_channels * 2, stride=1),
            RepConv2d(base_channels * 2, base_channels * 2, stride=1),
        )

        # Stage 3: 32x32 -> 16x16
        self.stage3 = SpatialFrequencyBlock(base_channels * 2, base_channels * 4, stride=2)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Returns:
            f_stage2: 32x32 intermediate feature map for Kinematic Flow alignment.
            f_stage3: 16x16 deep feature map for Cross-Spectral Attention Fusion.
        """
        x = self.stem(x)
        f1 = self.stage1(x)
        f2 = self.stage2(f1)  # (B, 64, 32, 32)
        f3 = self.stage3(f2)  # (B, 128, 16, 16)
        return f2, f3

    def reparameterize(self) -> None:
        """Collapses all RepConv2d modules in this stream for INT8 edge deployment."""
        for module in self.modules():
            if isinstance(module, RepConv2d):
                module.reparameterize()


class RepMSFNet(nn.Module):
    """
    Coupled Dual-Stream Rep-MSFNet Edge Backbone.
    Processes synchronized RGB and NIR images, providing aligned feature maps
    for dense cross-attention fusion and deep biometric metric embeddings.
    """

    def __init__(self, base_channels: int = 32, embedding_dim: int = EMBEDDING_DIM):
        super().__init__()
        self.rgb_stream = RepMSFNetStream(in_channels=3, base_channels=base_channels)
        self.nir_stream = RepMSFNetStream(in_channels=1, base_channels=base_channels)

        # Global pooling and embedding projection
        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))
        # After MCA-FF fusion of 16x16 maps (channels = base_channels * 4 = 128)
        self.head = nn.Sequential(
            nn.Linear(base_channels * 4, base_channels * 4),
            nn.BatchNorm1d(base_channels * 4),
            nn.SiLU(),
            nn.Linear(base_channels * 4, embedding_dim, bias=False),
        )

    def forward_features(self, rgb: torch.Tensor, nir: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Extracts multi-scale features from both spectral streams.
        Returns:
            rgb_f2 (32x32), rgb_f3 (16x16), nir_f2 (32x32), nir_f3 (16x16)
        """
        rgb_f2, rgb_f3 = self.rgb_stream(rgb)
        nir_f2, nir_f3 = self.nir_stream(nir)
        return rgb_f2, rgb_f3, nir_f2, nir_f3

    def project_embedding(self, fused_features: torch.Tensor) -> torch.Tensor:
        """Projects fused feature map (B, C, H, W) to L2-normalized embedding on S^(D-1)."""
        pooled = self.global_pool(fused_features).flatten(1)
        raw_emb = self.head(pooled)
        # Project onto unit hypersphere S^(D-1)
        normalized_emb = F.normalize(raw_emb, p=2, dim=1)
        return normalized_emb

    def reparameterize(self) -> None:
        """Reparameterizes both RGB and NIR streams for zero-latency deployment."""
        self.rgb_stream.reparameterize()
        self.nir_stream.reparameterize()

"""DFMNet: a lightweight SWH forecaster distilled from MICN.

Design philosophy (see analysis):
  - KEEP MICN's irreducible core: multi-scale moving-average decomposition
    (trend / seasonal split) + a cheap linear trend regression. These two
    pieces carry most of MICN's long-horizon advantage and are nearly
    parameter-free.
  - DROP the three always-on parallel conv branches (the weight dump shows the
    large-kernel branch is the most parameter-heavy AND the most diluted, i.e.
    redundant), and DROP RevIN / explicit channel-gate (both proved harmful:
    RevIN double-normalizes on top of decomposition; the channel-gate learned
    scales ~= 1.0, i.e. it is absorbed by the linear embedding).
  - REPLACE the redundant parallel branches with an *input-conditional*
    frequency Mixture-of-Experts (FreqMoE): a handful of tiny complex spectral
    filters, softly selected per window by a gate. Conditional selection gives
    the same multi-scale coverage with far fewer *effective* active parameters,
    which is exactly the lesson the weight analysis taught.

Maps each data property (paper Ch. III) to one and only one mechanism:
  A Non-stationarity      -> series decomposition (no extra normalization)
  B Long-range dependency -> linear trend regression (L -> H) + global FFT mixing
  C Multi-scale coupling  -> multi-kernel decomposition + FreqMoE bands
  D Multivariate heterog. -> single linear channel embedding (seasonal) and a
                             tiny channel mixer (trend); no separate gate.

Rename `DFMNet` to your paper's model name freely.
"""
from __future__ import annotations

from typing import Dict, List

import torch
import torch.nn as nn
import torch.nn.functional as F


# --------------------------------------------------------------------------- #
# Multi-scale moving-average decomposition (kept verbatim in spirit from MICN) #
# --------------------------------------------------------------------------- #
class MovingAverage(nn.Module):
    def __init__(self, kernel_size: int):
        super().__init__()
        self.kernel_size = int(kernel_size)
        self.avg = nn.AvgPool1d(kernel_size=self.kernel_size, stride=1, padding=0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # x: [B, T, C]
        pad_left = (self.kernel_size - 1) // 2
        pad_right = self.kernel_size - 1 - pad_left
        front = x[:, 0:1, :].repeat(1, pad_left, 1)
        end = x[:, -1:, :].repeat(1, pad_right, 1)
        x_pad = torch.cat([front, x, end], dim=1)
        return self.avg(x_pad.permute(0, 2, 1)).permute(0, 2, 1)


class SeriesDecompMulti(nn.Module):
    """Average several moving-average trends -> robust multi-scale trend."""

    def __init__(self, kernel_sizes: List[int]):
        super().__init__()
        ks = [k + 1 if k % 2 == 0 else k for k in kernel_sizes]
        self.mas = nn.ModuleList([MovingAverage(k) for k in ks])

    def forward(self, x: torch.Tensor):
        trends = [ma(x) for ma in self.mas]
        trend = torch.stack(trends, dim=0).mean(dim=0)
        seasonal = x - trend
        return seasonal, trend


# --------------------------------------------------------------------------- #
# Frequency Mixture-of-Experts (replaces MICN's parallel conv branches)        #
# --------------------------------------------------------------------------- #
class FreqMoE(nn.Module):
    """Input-conditional spectral filtering.

    Each expert owns a learnable complex filter over rFFT frequency bins
    (shared across the feature dimension to stay cheap). A small gate, fed by
    the per-window band-energy profile, softly mixes the experts. Because only
    the relevant expert(s) dominate for a given window, this covers multiple
    scales without the always-on redundancy that bloated MICN's conv branches.
    """

    def __init__(self, seq_len: int, d_model: int, n_experts: int = 3,
                 keep_modes: int | None = None, dropout: float = 0.1):
        super().__init__()
        self.seq_len = seq_len
        self.n_freq = seq_len // 2 + 1
        self.keep = self.n_freq if keep_modes is None else min(keep_modes, self.n_freq)
        self.n_experts = n_experts

        # complex filters: [E, keep] stored as separate real / imag parameters
        scale = 0.02
        self.w_real = nn.Parameter(torch.randn(n_experts, self.keep) * scale + 1.0)
        self.w_imag = nn.Parameter(torch.randn(n_experts, self.keep) * scale)

        # gate reads the band-energy profile (keep dims) -> expert weights
        self.gate = nn.Sequential(
            nn.Linear(self.keep, n_experts * 2),
            nn.GELU(),
            nn.Linear(n_experts * 2, n_experts),
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # x: [B, L, d]
        B, L, d = x.shape
        Xf = torch.fft.rfft(x, dim=1)                 # [B, n_freq, d] complex
        Xf = Xf[:, : self.keep, :]                    # keep low modes

        # band-energy profile for the gate (per window): [B, keep]
        energy = Xf.abs().mean(dim=-1)
        energy = energy / (energy.sum(dim=-1, keepdim=True) + 1e-8)
        g = torch.softmax(self.gate(energy), dim=-1)  # [B, E]

        # apply each expert's complex filter, then mix by gate
        w = torch.complex(self.w_real, self.w_imag)   # [E, keep]
        # [B, E, keep, 1] * [1, E, keep, 1] ... broadcast over feature dim
        Xf_e = Xf.unsqueeze(1) * w.unsqueeze(0).unsqueeze(-1)  # [B, E, keep, d]
        mixed = (Xf_e * g.unsqueeze(-1).unsqueeze(-1)).sum(dim=1)  # [B, keep, d]

        # pad truncated modes back with zeros, inverse FFT
        if self.keep < self.n_freq:
            pad = torch.zeros(B, self.n_freq - self.keep, d,
                              dtype=mixed.dtype, device=mixed.device)
            mixed = torch.cat([mixed, pad], dim=1)
        out = torch.fft.irfft(mixed, n=L, dim=1)      # [B, L, d]
        return self.dropout(out)


# --------------------------------------------------------------------------- #
# Seasonal block: embed -> FreqMoE (residual+norm) -> small FFN -> time proj    #
# --------------------------------------------------------------------------- #
class SeasonalBlock(nn.Module):
    def __init__(self, c_in: int, seq_len: int, pred_len: int, d_model: int,
                 n_experts: int, keep_modes: int | None, ffn_ratio: int,
                 use_ffn: bool, dropout: float):
        super().__init__()
        self.embed = nn.Linear(c_in, d_model)          # also handles channel heterogeneity (D)
        self.freq_moe = FreqMoE(seq_len, d_model, n_experts, keep_modes, dropout)
        self.norm1 = nn.LayerNorm(d_model)             # weight dump: LayerNorm is high-value
        self.use_ffn = use_ffn
        if use_ffn:
            self.ffn = nn.Sequential(
                nn.Linear(d_model, d_model * ffn_ratio),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(d_model * ffn_ratio, d_model),
            )
            self.norm2 = nn.LayerNorm(d_model)
        self.time_proj = nn.Linear(seq_len, pred_len)  # DLinear-style horizon map (B)
        self.head = nn.Linear(d_model, 1)              # target-only output
        self.dropout = nn.Dropout(dropout)

    def forward(self, seasonal: torch.Tensor) -> torch.Tensor:  # [B, L, C]
        h = self.embed(seasonal)                       # [B, L, d]
        h = self.norm1(h + self.dropout(self.freq_moe(h)))
        if self.use_ffn:
            h = self.norm2(h + self.dropout(self.ffn(h)))
        h = self.time_proj(h.transpose(1, 2)).transpose(1, 2)   # [B, H, d]
        return self.head(h)                            # [B, H, 1]


# --------------------------------------------------------------------------- #
# Trend block: cheap channel mix -> linear regression L -> H                    #
# --------------------------------------------------------------------------- #
class TrendBlock(nn.Module):
    def __init__(self, c_in: int, seq_len: int, pred_len: int):
        super().__init__()
        self.channel_mix = nn.Linear(c_in, 1)          # uses strong low-freq cross-corr (D)
        self.regression = nn.Linear(seq_len, pred_len)
        nn.init.constant_(self.regression.weight, 1.0 / max(1, seq_len))
        nn.init.zeros_(self.regression.bias)

    def forward(self, trend: torch.Tensor) -> torch.Tensor:  # [B, L, C]
        t = self.channel_mix(trend)                    # [B, L, 1]
        t = self.regression(t.transpose(1, 2)).transpose(1, 2)  # [B, H, 1]
        return t


# --------------------------------------------------------------------------- #
class DFMNet(nn.Module):
    def __init__(self, configs):
        super().__init__()
        self.seq_len = configs.seq_len
        self.pred_len = configs.pred_len
        self.enc_in = configs.enc_in
        self.target_idx = configs.target_idx

        self.decomp = SeriesDecompMulti(list(configs.decomp_kernel))
        self.trend_block = TrendBlock(self.enc_in, self.seq_len, self.pred_len)
        self.seasonal_block = SeasonalBlock(
            c_in=self.enc_in,
            seq_len=self.seq_len,
            pred_len=self.pred_len,
            d_model=configs.d_model,
            n_experts=configs.n_experts,
            keep_modes=configs.keep_modes,
            ffn_ratio=configs.ffn_ratio,
            use_ffn=configs.use_ffn,
            dropout=configs.dropout,
        )

    def forward(self, x_enc: torch.Tensor) -> Dict[str, torch.Tensor]:
        seasonal, trend = self.decomp(x_enc)           # [B, L, C] each
        trend_out = self.trend_block(trend)            # [B, H, 1]
        seasonal_out = self.seasonal_block(seasonal)   # [B, H, 1]
        point = seasonal_out + trend_out               # [B, H, 1]
        return {"point": point, "full": point,
                "trend": trend_out, "seasonal": seasonal_out}


Model = DFMNet

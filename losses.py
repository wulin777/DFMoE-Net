"""Loss functions for baseline models."""
from __future__ import annotations

from typing import Dict
import torch
import torch.nn as nn


class ForecastMSELoss(nn.Module):
    """保持与你的 WaveSpec / PatchTST 训练循环一致：输入 outputs 字典，返回 loss 字典。"""

    def __init__(self, config=None):
        super().__init__()
        self.mse = nn.MSELoss()

    def forward(self, outputs: Dict[str, torch.Tensor], target: torch.Tensor) -> Dict[str, torch.Tensor]:
        pred = outputs["point"] if isinstance(outputs, dict) else outputs
        loss = self.mse(pred, target)
        return {"total": loss, "mse": loss}

"""Shared training / evaluation utilities."""
from __future__ import annotations

import random
from pathlib import Path
from typing import Dict, Union

import numpy as np
import torch


def set_seed(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = True


def ensure_dir(path: Union[str, Path]) -> Path:
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def load_checkpoint_state_dict(checkpoint):
    """兼容 state_dict 或包含 state_dict/model_state_dict/model 的 checkpoint。"""
    if isinstance(checkpoint, dict):
        for key in ("state_dict", "model_state_dict", "model"):
            if key in checkpoint and isinstance(checkpoint[key], dict):
                checkpoint = checkpoint[key]
                break

    if not isinstance(checkpoint, dict):
        raise TypeError("无法识别权重文件格式：checkpoint 不是 state_dict 或包含 state_dict 的字典。")

    new_state_dict = {}
    for k, v in checkpoint.items():
        new_key = k.replace("module.", "", 1) if k.startswith("module.") else k
        new_state_dict[new_key] = v
    return new_state_dict


def regression_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    # 与 PatchTST.zip 一致：对整个预测序列 H 展平后计算总体指标。
    yt = y_true.reshape(-1).astype(np.float64)
    yp = y_pred.reshape(-1).astype(np.float64)
    err = yp - yt
    mae = float(np.mean(np.abs(err)))
    mse = float(np.mean(err ** 2))
    rmse = float(np.sqrt(mse))

    denom = float(np.sum((yt - np.mean(yt)) ** 2))
    r2 = float(1.0 - np.sum(err ** 2) / denom) if denom > 0 else float("nan")
    corr = float(np.corrcoef(yt, yp)[0, 1]) if np.std(yt) > 0 and np.std(yp) > 0 else float("nan")
    return {"MAE": mae, "MSE": mse, "RMSE": rmse, "R2": r2, "Corr": corr}


def write_lines(lines, output_file: Union[str, Path]) -> None:
    output_file = Path(output_file)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    text = "\n".join(lines)
    print(text)
    output_file.write_text(text + "\n", encoding="utf-8")

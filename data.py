"""Australian wave dataset loader for baseline comparison.

默认读取 Excel 中的 6 个变量：
Hmax, Tp, Tz, Drip, Tsea, 有效波高
其中 Drip 会从角度制转换为弧度。目标列默认是最后一列：有效波高。
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Sequence

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset


DEFAULT_COLUMNS =["Hmax", "Tp", "Tz", "Drip", "Tsea", "有效波高"]#["Tp", "Drip", "Tz", "T1", "Tc", "有效波高"] #["WDIR", "WSPD", "GST", "DPD", "APD", "有效波高"]#


@dataclass
class StandardScaler:
    mean: np.ndarray
    std: np.ndarray

    def transform(self, arr: np.ndarray) -> np.ndarray:
        return (arr - self.mean) / self.std

    def inverse_transform_target(self, arr: np.ndarray, target_idx: int) -> np.ndarray:
        return arr * self.std[target_idx] + self.mean[target_idx]


def _select_numeric_columns(df: pd.DataFrame, columns: Optional[Sequence[str]] = None) -> pd.DataFrame:
    if columns is None:
        if all(col in df.columns for col in DEFAULT_COLUMNS):
            columns = DEFAULT_COLUMNS
        else:
            numeric_cols = list(df.select_dtypes(include=[np.number]).columns)
            if len(numeric_cols) < 6:
                raise ValueError(
                    "Excel 中没有找到默认列名，也没有至少 6 个数值列。"
                    f"当前列名: {list(df.columns)}"
                )
            columns = numeric_cols[:6]

    missing = [c for c in columns if c not in df.columns]
    if missing:
        raise ValueError(f"Excel 缺少这些列: {missing}; 当前列名: {list(df.columns)}")

    return df.loc[:, list(columns)].copy()


def load_wave_array(file_path: str, columns: Optional[Sequence[str]] = None) -> np.ndarray:
    df = pd.read_excel(file_path)
    df = _select_numeric_columns(df, columns)

    # 处理缺失值：线性插值 + 前后填充，避免训练时出现 NaN。
    df = df.interpolate(method="linear", limit_direction="both").ffill().bfill()

    arr = df.values.astype(np.float32)

    # 如果存在 Drip，按你的原始代码将角度转换为弧度。
    col_names = list(df.columns)
    if "Drip" in col_names:
        drip_idx = col_names.index("Drip")
        arr[:, drip_idx] = np.deg2rad(arr[:, drip_idx])

    if not np.isfinite(arr).all():
        raise ValueError("数据中仍存在 NaN 或 Inf，请检查 Excel 文件。")
    return arr


def split_array(
    arr: np.ndarray,
    mode: str = "train",
    train_ratio: float = 0.8,
    val_ratio: float = 0.2,
) -> np.ndarray:
    """按时间顺序切分，默认 train=前80%，val=后20%。

    如果需要独立 test，可把 train_ratio 调小，例如 0.7, val_ratio=0.15，
    此时 test 为最后 15%。默认 8:2 时 test.py 会复用 val 区间。
    """
    mode = mode.lower()
    n = len(arr)
    train_end = int(n * train_ratio)
    val_end = int(n * (train_ratio + val_ratio))

    if mode == "train":
        part = arr[:train_end]
    elif mode in ("val", "valid", "eval"):
        part = arr[train_end:val_end]
    elif mode == "test":
        part = arr[val_end:]
        if len(part) == 0:
            part = arr[train_end:]
    elif mode == "all":
        part = arr
    else:
        raise ValueError("mode 必须是 train/val/test/all 之一")

    return part.astype(np.float32)


class AustralianWaveDataset(Dataset):
    """Sliding-window supervised dataset.

    返回：
      x: [L, C]
      y: [H, 1]，目标为 target_idx 对应列，默认有效波高。
    """

    def __init__(
        self,
        file_path: str,
        mode: str = "train",
        L: int = 48,
        H: int = 96,
        target_idx: int = 5,
        columns: Optional[Sequence[str]] = None,
        train_ratio: float = 0.8,
        val_ratio: float = 0.2,
    ):
        super().__init__()
        self.L = int(L)
        self.H = int(H)
        self.target_idx = int(target_idx)

        full = load_wave_array(file_path, columns=columns)
        self.data = split_array(full, mode=mode, train_ratio=train_ratio, val_ratio=val_ratio)

        if self.data.shape[1] <= self.target_idx:
            raise ValueError(f"target_idx={self.target_idx} 超出数据列数 {self.data.shape[1]}")
        if len(self.data) < self.L + self.H + 1:
            raise ValueError(
                f"{mode} 数据太短：len={len(self.data)}, 但需要至少 L+H+1={self.L + self.H + 1}。"
            )

    def __len__(self) -> int:
        return len(self.data) - self.L - self.H + 1

    def __getitem__(self, i: int):
        x = self.data[i : i + self.L]
        y = self.data[i + self.L : i + self.L + self.H, self.target_idx : self.target_idx + 1]
        return torch.from_numpy(x).float(), torch.from_numpy(y).float()

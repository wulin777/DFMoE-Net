"""
DFMNet 多种子评估脚本（直接在编辑器里点 Run，无需命令行参数）。

读取 train.py 训练得到的 3 个 seed 权重，逐个在 test 上评估，然后：
  - 窗口实时打印每个 seed 的指标；
  - 写出 results/evaluate_DFMNet_L{L}_H{H}_{mode}.txt，内容包含：
      1) 3 个种子各自的 MAE / MSE / RMSE / R2 / Corr；
      2) 最终汇总指标 mean(±std)。

评估在 eval 模式下进行（dropout 关闭、权重固定），因此给定权重后结果确定，
与 DFMNet_abla full 在同一配置（同 L/H/数据/seed）下的结果一致。

注意：默认 8:2 划分时 data.py 中 test 区间为空，会自动回退到验证集区间，
即 test 与 val 评估的是同一段数据。
"""
from __future__ import annotations

import statistics as stats
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from config import DFMConfig
from data import AustralianWaveDataset
from models.dfmnet import DFMNet
from utils import ensure_dir, load_checkpoint_state_dict, regression_metrics, write_lines


# ===================== 关键参数：按需修改这里 =====================
L = 144              # 回看窗口
H = 192                 # 预测步长（与训练保持一致）
SEEDS = [42, 1, 2]      # 与训练保持一致
MODE = "test"           # val / test / all
BATCH_SIZE = 32
DATA_PATH = None        # None -> 用 config.data_path
DEVICE = None           # None -> 自动（有 GPU 用 cuda，否则 cpu）
# ================================================================

METRIC_KEYS = ["MAE", "MSE", "RMSE", "R2", "Corr"]


def seed_ckpt_path(L: int, H: int, seed: int) -> Path:
    """与 train.py 中的权重保存路径保持一致。"""
    return Path("weights") / f"H{H}" / f"seed{seed}" / f"best_L{L}_H{H}.pt"


def make_config() -> DFMConfig:
    cfg = DFMConfig()
    cfg.L = L
    cfg.H = H
    cfg.batch_size = BATCH_SIZE
    if DATA_PATH is not None:
        cfg.data_path = DATA_PATH
    return cfg


@torch.no_grad()
def eval_one(cfg: DFMConfig, device, ckpt_path: Path, mode: str) -> dict:
    ds = AustralianWaveDataset(
        cfg.data_path, mode=mode, L=cfg.L, H=cfg.H, target_idx=cfg.target_idx,
        columns=cfg.columns, train_ratio=cfg.train_ratio, val_ratio=cfg.val_ratio,
    )
    loader = DataLoader(ds, batch_size=cfg.batch_size, shuffle=False, drop_last=False)

    model = DFMNet(cfg).to(device)
    checkpoint = torch.load(ckpt_path, map_location=device)
    state_dict = load_checkpoint_state_dict(checkpoint)
    model.load_state_dict(state_dict, strict=True)
    model.eval()

    preds, tgts = [], []
    for x, y in loader:
        x = x.to(device)
        out = model(x)
        pred = out["point"] if isinstance(out, dict) else out
        preds.append(pred.detach().cpu().numpy())
        tgts.append(y.numpy())

    if len(preds) == 0:
        raise RuntimeError("数据集为空，请检查数据量是否大于 L + H。")

    preds = np.concatenate(preds, axis=0)
    tgts = np.concatenate(tgts, axis=0)
    return regression_metrics(tgts, preds), preds.shape, tgts.shape


def mean_std(values):
    mean_value = stats.mean(values)
    std_value = stats.pstdev(values) if len(values) > 1 else 0.0
    return mean_value, std_value


def main():
    device = torch.device(
        "cuda" if (DEVICE in (None, "cuda") and torch.cuda.is_available()) else "cpu"
    )
    cfg = make_config()
    results_dir = ensure_dir(Path(cfg.results_dir))
    output_file = results_dir / f"evaluate_DFMNet_L{L}_H{H}_{MODE}.txt"
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    print("=" * 70)
    print(f"DFMNet 多种子评估 | L={L} H={H} | seeds={SEEDS} | mode={MODE}")
    print(f"设备: {device} | 数据: {cfg.data_path}")
    print("=" * 70)

    per_seed = {k: [] for k in METRIC_KEYS}
    seed_metrics = {}     # seed -> metrics dict
    pred_shape = tgt_shape = None
    n_params = None

    for seed in SEEDS:
        ckpt = seed_ckpt_path(L, H, seed)
        if not ckpt.exists():
            raise FileNotFoundError(
                f"找不到 seed={seed} 的权重: {ckpt}\n请先运行 train.py 完成该 seed 的训练。"
            )
        m, pred_shape, tgt_shape = eval_one(cfg, device, ckpt, MODE)
        seed_metrics[seed] = m
        for k in METRIC_KEYS:
            per_seed[k].append(m[k])
        if n_params is None:
            tmp = DFMNet(cfg)
            n_params = sum(p.numel() for p in tmp.parameters() if p.requires_grad)
        print(f"[seed={seed}] MAE={m['MAE']:.6f} MSE={m['MSE']:.6f} "
              f"RMSE={m['RMSE']:.6f} R2={m['R2']:.6f} Corr={m['Corr']:.6f}")

    agg = {k: mean_std(per_seed[k]) for k in METRIC_KEYS}

    # ---------------- 写出 TXT ----------------
    lines = [
        "========== DFMNet 多种子评价结果 ==========",
        f"运行时间: {now}",
        f"设备: {device}",
        f"参数量: {n_params:,}",
        "",
        "---------- 关键参数 ----------",
        f"L: {L}",
        f"H: {H}",
        f"seeds: {SEEDS}",
        f"batch_size: {BATCH_SIZE}",
        f"mode: {MODE}",
        f"数据路径: {cfg.data_path}",
        "",
        "---------- 数据形状 ----------",
        f"预测结果形状: {pred_shape}",
        f"真实值形状: {tgt_shape}",
        "",
        "---------- 各 seed 结果 ----------",
    ]
    for seed in SEEDS:
        m = seed_metrics[seed]
        lines += [
            f"[seed={seed}]  权重: {seed_ckpt_path(L, H, seed)}",
            f"    MAE:  {m['MAE']:.6f}",
            f"    MSE:  {m['MSE']:.6f}",
            f"    RMSE: {m['RMSE']:.6f}",
            f"    R²:   {m['R2']:.6f}",
            f"    Corr: {m['Corr']:.6f}",
        ]
    lines += [
        "",
        f"---------- 最终汇总指标（{len(SEEDS)} 个 seed，mean ± std） ----------",
        f"MAE:  {agg['MAE'][0]:.6f} ± {agg['MAE'][1]:.6f}",
        f"MSE:  {agg['MSE'][0]:.6f} ± {agg['MSE'][1]:.6f}",
        f"RMSE: {agg['RMSE'][0]:.6f} ± {agg['RMSE'][1]:.6f}",
        f"R²:   {agg['R2'][0]:.6f} ± {agg['R2'][1]:.6f}",
        f"Corr: {agg['Corr'][0]:.6f} ± {agg['Corr'][1]:.6f}",
        "===========================================",
    ]

    write_lines(lines, output_file)
    print(f"\n评价指标已保存至: {output_file}")


if __name__ == "__main__":
    main()

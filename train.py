"""
DFMNet 多种子训练脚本（直接在编辑器里点 Run，无需命令行参数）。

与 DFMNet_abla 的 full 配置在同一设定下完全对齐：
  - 每个 seed 先 set_seed(seed)，再依次构建数据集 -> 模型 -> 训练；
  - 训练循环、优化器、调度器、按验证损失保存 best 的逻辑，与 run_ablation.py
    的 train_one 完全一致（仅额外增加了每个 epoch 的窗口打印）。
  因此只要 L / H / epochs / 数据 一致，用相同 seed 训练得到的权重与 abla 中 full
  对应 seed 的权重逐字节相同，评估结果自然一致。

运行后：
  - 窗口（编辑器输出区）实时打印每个 seed、每个 epoch 的训练/验证损失；
  - 每个 seed 的最佳权重分别保存到 weights/H{H}/seed{seed}/best_L{L}_H{H}.pt。

训练完成后再运行 evaluate.py 即可得到 3 个种子 + 汇总指标的 TXT。
"""
from __future__ import annotations

from pathlib import Path

import torch
from torch.utils.data import DataLoader

from config import DFMConfig
from data import AustralianWaveDataset
from losses import ForecastMSELoss
from models.dfmnet import DFMNet
from utils import ensure_dir, set_seed


# ===================== 关键参数：按需修改这里 =====================
L = 144              # 回看窗口
H = 192             # 预测步长：2 / 12 / 24 / 48 / 96 / 144 / 192 / 336
SEEDS = [42, 1, 2]      # 多种子；单种子就写 [42]
EPOCHS = 60           # None -> 用 config 默认 (30)
BATCH_SIZE = None       # None -> 用 config 默认 (32)
LR = None               # None -> 用 config 默认 (1e-3)
DATA_PATH = None        # None -> 用 config.data_path
DEVICE = None           # None -> 自动（有 GPU 用 cuda，否则 cpu）
# ================================================================


def seed_ckpt_path(L: int, H: int, seed: int) -> Path:
    """每个 seed 的权重保存路径。"""
    return Path("weights") / f"H{H}" / f"seed{seed}" / f"best_L{L}_H{H}.pt"


def make_config() -> DFMConfig:
    cfg = DFMConfig()
    cfg.L = L
    cfg.H = H
    if EPOCHS is not None:
        cfg.n_epochs = EPOCHS
    if BATCH_SIZE is not None:
        cfg.batch_size = BATCH_SIZE
    if LR is not None:
        cfg.lr = LR
    if DATA_PATH is not None:
        cfg.data_path = DATA_PATH
    return cfg


def train_one(cfg: DFMConfig, device, save_path: Path, seed: int) -> float:
    """训练单个 seed，返回最佳验证损失；逻辑与 abla full 的 train_one 对齐。"""
    train_ds = AustralianWaveDataset(
        cfg.data_path, mode="train", L=cfg.L, H=cfg.H, target_idx=cfg.target_idx,
        columns=cfg.columns, train_ratio=cfg.train_ratio, val_ratio=cfg.val_ratio,
    )
    val_ds = AustralianWaveDataset(
        cfg.data_path, mode="val", L=cfg.L, H=cfg.H, target_idx=cfg.target_idx,
        columns=cfg.columns, train_ratio=cfg.train_ratio, val_ratio=cfg.val_ratio,
    )

    train_loader = DataLoader(train_ds, batch_size=cfg.batch_size, shuffle=True, drop_last=False)
    val_loader = DataLoader(val_ds, batch_size=cfg.batch_size, shuffle=False, drop_last=False)

    model = DFMNet(cfg).to(device)
    criterion = ForecastMSELoss(cfg).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=cfg.n_epochs)

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  模型: DFMNet | 参数量: {n_params:,} | 保存到: {save_path}")

    best_val = float("inf")
    for epoch in range(cfg.n_epochs):
        model.train()
        train_loss = 0.0
        for x, y in train_loader:
            x = x.to(device)
            y = y.to(device)
            optimizer.zero_grad(set_to_none=True)
            loss = criterion(model(x), y)["total"]
            loss.backward()
            if cfg.grad_clip is not None and cfg.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
            optimizer.step()
            train_loss += loss.item()

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for x, y in val_loader:
                x = x.to(device)
                y = y.to(device)
                val_loss += criterion(model(x), y)["total"].item()

        avg_train = train_loss / max(1, len(train_loader))
        avg_val = val_loss / max(1, len(val_loader))

        flag = ""
        if avg_val < best_val:
            best_val = avg_val
            ensure_dir(save_path.parent)
            torch.save(
                {"model_state_dict": model.state_dict(), "config": cfg.__dict__,
                 "best_val_loss": best_val, "seed": seed},
                save_path,
            )
            flag = "  --> best，已保存"

        # 每个 epoch 都在窗口打印，方便观察进程
        print(f"  [seed={seed}] Epoch [{epoch + 1}/{cfg.n_epochs}] | "
              f"Train Loss: {avg_train:.6f} | Val Loss: {avg_val:.6f}{flag}")

        scheduler.step()

    print(f"  [seed={seed}] 训练完成，最佳验证损失: {best_val:.6f}\n")
    return best_val


def main():
    device = torch.device(
        "cuda" if (DEVICE in (None, "cuda") and torch.cuda.is_available()) else "cpu"
    )
    cfg_preview = make_config()
    print("=" * 70)
    print(f"DFMNet 多种子训练 | L={L} H={H} | seeds={SEEDS} | epochs={cfg_preview.n_epochs}")
    print(f"设备: {device} | 数据: {cfg_preview.data_path}")
    print("=" * 70)

    for seed in SEEDS:
        print(f"\n>>> 开始训练 seed={seed}")
        set_seed(seed)
        cfg = make_config()
        ckpt = seed_ckpt_path(cfg.L, cfg.H, seed)
        train_one(cfg, device, ckpt, seed)

    print("=" * 70)
    print("全部 seed 训练完成。各 seed 权重：")
    for seed in SEEDS:
        print(f"  seed={seed}: {seed_ckpt_path(L, H, seed)}")
    print("接下来运行 evaluate.py 生成 3 个种子 + 汇总指标的 TXT。")
    print("=" * 70)


if __name__ == "__main__":
    main()

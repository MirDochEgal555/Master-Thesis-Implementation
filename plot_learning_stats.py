import argparse
import math
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


LOSS_COLUMNS = [
    "total_loss",
    "policy_loss",
    "value_loss",
    "kf_loss",
    "dyn_loss",
    "sim_policy_loss",
    "sim_value_loss",
]
PHASE_ORDER = ["warmup_kf", "warmup_dyn", "warmup_crit", "train"]


def _min_max_normalize(series: pd.Series) -> pd.Series:
    min_val = series.min()
    max_val = series.max()
    if math.isclose(min_val, max_val, rel_tol=0.0, abs_tol=1e-12):
        return pd.Series(np.zeros(len(series)), index=series.index)
    return (series - min_val) / (max_val - min_val)


def plot_validation_sharpe(df: pd.DataFrame, out_dir: Path | None) -> None:
    val_df = df[df["val_sharpe"].notna()].copy()
    if val_df.empty:
        print("No validation Sharpe values found in CSV.")
        return
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(val_df["epoch"], val_df["val_sharpe"], label="val_sharpe")
    ax.set_title("Validation Sharpe")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Sharpe")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    if out_dir:
        fig.savefig(out_dir / "validation_sharpe.png", dpi=150)
    else:
        plt.show()


def plot_losses_normalized(df: pd.DataFrame, out_dir: Path | None) -> None:
    loss_df = df.copy()
    fig, ax = plt.subplots(figsize=(10, 4))
    phase_offsets = {}
    offset = 0
    for phase in PHASE_ORDER:
        phase_max = loss_df.loc[loss_df["phase"] == phase, "epoch"].max()
        if pd.isna(phase_max):
            continue
        phase_offsets[phase] = offset
        offset += int(phase_max) + 1
    if not phase_offsets:
        phase_offsets = {None: 0}
    loss_df["global_step"] = loss_df.apply(
        lambda row: row["epoch"] + phase_offsets.get(row.get("phase"), 0), axis=1
    )
    for col in LOSS_COLUMNS:
        if col not in loss_df.columns:
            continue
        series = loss_df[col].astype(float)
        if series.isna().all():
            continue
        ax.plot(loss_df["global_step"], _min_max_normalize(series), label=col)
    ax.set_title("Losses (Min-Max Normalized, Warmups Included)")
    ax.set_xlabel("Global Step")
    ax.set_ylabel("Normalized loss")
    ax.grid(True, alpha=0.3)
    ax.legend(ncol=2, fontsize=8)
    fig.tight_layout()
    if out_dir:
        fig.savefig(out_dir / "losses_normalized.png", dpi=150)
    else:
        plt.show()


def plot_discounted_reward(df: pd.DataFrame, out_dir: Path | None) -> None:
    reward_df = df[df["discounted_reward"].notna()].copy()
    if reward_df.empty:
        print("No discounted reward values found in CSV.")
        return
    phase_offsets = {}
    offset = 0
    for phase in PHASE_ORDER:
        phase_max = reward_df.loc[reward_df["phase"] == phase, "epoch"].max()
        if pd.isna(phase_max):
            continue
        phase_offsets[phase] = offset
        offset += int(phase_max) + 1
    if not phase_offsets:
        phase_offsets = {None: 0}
    reward_df["global_step"] = reward_df.apply(
        lambda row: row["epoch"] + phase_offsets.get(row.get("phase"), 0), axis=1
    )
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(reward_df["global_step"], reward_df["discounted_reward"], label="discounted_reward")
    ax.set_title("Discounted Reward (Warmups Included)")
    ax.set_xlabel("Global Step")
    ax.set_ylabel("Reward")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    if out_dir:
        fig.savefig(out_dir / "discounted_reward.png", dpi=150)
    else:
        plt.show()


def plot_average_weights(df: pd.DataFrame, out_dir: Path | None) -> None:
    weight_cols = [col for col in df.columns if col.startswith("avg_weight_")]
    if not weight_cols:
        print("No avg_weight_* columns found in CSV.")
        return
    weight_df = df.copy()
    weight_df[weight_cols] = weight_df[weight_cols].apply(pd.to_numeric, errors="coerce")
    weight_df = weight_df.dropna(subset=weight_cols, how="all")
    if weight_df.empty:
        print("Average weight columns are all NaN.")
        return

    phase_offsets = {}
    offset = 0
    for phase in PHASE_ORDER:
        phase_max = weight_df.loc[weight_df["phase"] == phase, "epoch"].max()
        if pd.isna(phase_max):
            continue
        phase_offsets[phase] = offset
        offset += int(phase_max) + 1
    if not phase_offsets:
        phase_offsets = {None: 0}
    weight_df["global_step"] = weight_df.apply(
        lambda row: row["epoch"] + phase_offsets.get(row.get("phase"), 0), axis=1
    )

    fig, ax = plt.subplots(figsize=(10, 4))
    for col in weight_cols:
        series = weight_df[col].astype(float)
        if series.isna().all():
            continue
        label = col.replace("avg_weight_", "")
        ax.plot(weight_df["global_step"], series, label=label)
    ax.set_title("Average Weights (Warmups Included)")
    ax.set_xlabel("Global Step")
    ax.set_ylabel("Average weight")
    ax.grid(True, alpha=0.3)
    ax.legend(ncol=2, fontsize=8)
    fig.tight_layout()
    if out_dir:
        fig.savefig(out_dir / "average_weights.png", dpi=150)
    else:
        plt.show()


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot learning stats from CSV.")
    parser.add_argument("csv_path", help="Path to learning_stats.csv")
    parser.add_argument(
        "--out-dir",
        default=None,
        help="Directory to save PNGs. If omitted, plots are shown interactively.",
    )
    args = parser.parse_args()

    csv_path = Path(args.csv_path)
    df = pd.read_csv(csv_path)
    out_dir = Path(args.out_dir) if args.out_dir else None
    if out_dir:
        out_dir.mkdir(parents=True, exist_ok=True)

    plot_validation_sharpe(df, out_dir)
    plot_losses_normalized(df, out_dir)
    plot_discounted_reward(df, out_dir)
    plot_average_weights(df, out_dir)


if __name__ == "__main__":
    main()

import argparse
import math
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


DEFAULT_PHASE_ORDER = ["warmup_kf", "warmup_dyn", "warmup_crit", "train"]


def _infer_loss_columns(df: pd.DataFrame) -> list[str]:
    return [col for col in df.columns if "loss" in col]


def _min_max_normalize(
    series: pd.Series,
    mask: pd.Series | None = None,
    eps: float = 1e-12,
) -> pd.Series:
    if mask is None:
        valid = series.dropna()
    else:
        valid = series[mask].dropna()
    if valid.empty:
        return series
    min_val = valid.min()
    max_val = valid.max()
    if math.isclose(min_val, max_val, rel_tol=0.0, abs_tol=eps):
        return pd.Series(np.zeros(len(series)), index=series.index)
    norm = (series - min_val) / (max_val - min_val)
    return norm.clip(lower=0.0, upper=1.0 - eps)


def _build_global_step(df: pd.DataFrame, phase_order: list[str]) -> pd.Series:
    if "epoch" not in df.columns:
        return pd.Series(np.arange(len(df)), index=df.index)
    if "phase" not in df.columns:
        return df["epoch"].astype(float)
    phase_offsets: dict[str | None, int] = {}
    offset = 0
    for phase in phase_order:
        phase_max = df.loc[df["phase"] == phase, "epoch"].max()
        if pd.isna(phase_max):
            continue
        phase_offsets[phase] = offset
        offset += int(phase_max) + 1
    if not phase_offsets:
        phase_offsets = {None: 0}
    return df.apply(
        lambda row: row["epoch"] + phase_offsets.get(row.get("phase"), 0), axis=1
    ).astype(float)


def plot_nonzero_losses(
    csv_path: Path,
    out_path: Path | None,
    vline: float | None,
) -> None:
    df = pd.read_csv(csv_path)
    loss_cols = _infer_loss_columns(df)
    if not loss_cols:
        print("No loss columns found in CSV.")
        return

    x = _build_global_step(df, DEFAULT_PHASE_ORDER)

    fig, ax = plt.subplots(figsize=(10, 4))
    plotted_any = False
    for col in loss_cols:
        series = pd.to_numeric(df[col], errors="coerce")
        mask = series.notna() & (~np.isclose(series, 0.0, atol=1e-12))
        normalized = _min_max_normalize(series, mask=mask)
        if not mask.any():
            continue
        ax.plot(x[mask], normalized[mask], label=col)
        plotted_any = True

    if not plotted_any:
        print("All loss columns are zero or empty.")
        return

    title = f"Non-zero losses: {csv_path.name}"
    ax.set_title(title)
    ax.set_xlabel("Global Step" if "phase" in df.columns else "Epoch")
    ax.set_ylabel("Normalized loss (min-max)")
    ax.grid(True, alpha=0.3)
    if vline is not None:
        ax.axvline(vline, color="red", linestyle=":", linewidth=1.5)
    ax.legend(ncol=2, fontsize=8)
    fig.tight_layout()

    if out_path:
        out_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out_path, dpi=150)
    else:
        plt.show()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Plot all non-zero loss columns from a grid search losses CSV."
    )
    parser.add_argument(
        "csv_path",
        help="Path to losses CSV (e.g. grid_search_results/losses/losses_combo1_seed4.csv)",
    )
    parser.add_argument(
        "--out",
        default=None,
        help="Output PNG path. If omitted, the plot is shown interactively.",
    )
    parser.add_argument(
        "--vline",
        type=float,
        default=None,
        help="X-position for a red dotted vertical line (global step or epoch).",
    )
    args = parser.parse_args()

    csv_path = Path(args.csv_path)
    out_path = Path(args.out) if args.out else None
    plot_nonzero_losses(csv_path, out_path, args.vline)


if __name__ == "__main__":
    main()

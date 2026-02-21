from __future__ import annotations

import argparse
import csv
import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import matplotlib.pyplot as plt
import numpy as np
import torch
from scipy.io import savemat

from portfolio_rl.backtest_seed import (
    DEFAULT_END_DATE,
    DEFAULT_PRICE_FIELD,
    DEFAULT_START_DATE,
    DEFAULT_TICKERS,
    DEFAULT_TRAIN_END,
    DEFAULT_VAL_END,
)
from portfolio_rl.config import TrainConfig
from portfolio_rl.data import YahooConfig, YahooReturnsDataset
from portfolio_rl.kalman import LearnableKalman
from portfolio_rl.models import PolicyNet
from portfolio_rl.trainer import Trainer


@dataclass
class PolicyEvalOutput:
    checkpoint: Path
    sharpe: float
    total_return: float
    mean_return: float
    std_return: float
    max_drawdown: float
    weights_csv: Path
    weights_mat: Path
    plot_png: Path


class PolicyCheckpointMatPlotter:
    """
    Load policy checkpoints (*.pt), evaluate them on the test split,
    and save weights as CSV, MAT, and matplotlib PNG plots.
    """

    def __init__(
        self,
        *,
        out_dir: str | Path = "policy_weight_outputs",
        cache_path: str | None = "returns.parquet",
        tickers: Iterable[str] = DEFAULT_TICKERS,
        start_date: str = DEFAULT_START_DATE,
        end_date: str = DEFAULT_END_DATE,
        price_field: str = DEFAULT_PRICE_FIELD,
        train_end: str = DEFAULT_TRAIN_END,
        val_end: str = DEFAULT_VAL_END,
        default_window_size: int = 10,
        device: str = "cpu",
    ):
        self.out_dir = Path(out_dir)
        self.cache_path = cache_path
        self.tickers = tuple(tickers)
        self.start_date = start_date
        self.end_date = end_date
        self.price_field = price_field
        self.train_end = train_end
        self.val_end = val_end
        self.default_window_size = int(default_window_size)
        self.device = torch.device(device)

        self.out_dir.mkdir(parents=True, exist_ok=True)
        ycfg = YahooConfig(
            tickers=list(self.tickers),
            start_date=self.start_date,
            end_date=self.end_date,
            price_field=self.price_field,
            cache_path=self.cache_path,
        )
        dataset = YahooReturnsDataset(ycfg)
        _, _, self.test_view = dataset.split_by_date(
            train_end=self.train_end,
            val_end=self.val_end,
        )

    def _load_checkpoint(self, checkpoint_path: Path) -> dict:
        try:
            return torch.load(checkpoint_path, map_location=self.device)
        except pickle.UnpicklingError:
            from torch.serialization import safe_globals

            with safe_globals([TrainConfig]):
                return torch.load(
                    checkpoint_path,
                    map_location=self.device,
                    weights_only=False,
                )

    @staticmethod
    def _extract_model_sizes(policy_state_dict: dict) -> tuple[int, int]:
        first_layer = policy_state_dict.get("net.0.weight")
        if first_layer is None:
            raise ValueError("Checkpoint is missing policy net.0.weight.")
        hidden_size = int(first_layer.shape[0])
        n_assets = int(first_layer.shape[1])
        return hidden_size, n_assets

    def _build_models(self, ckpt: dict) -> tuple[PolicyNet, LearnableKalman | None]:
        hidden_size, n_assets = self._extract_model_sizes(ckpt["policy_state_dict"])
        policy = PolicyNet(K=n_assets, hidden=hidden_size).to(self.device)
        policy.load_state_dict(ckpt["policy_state_dict"])

        cfg = ckpt.get("cfg", {})
        use_kf = bool(cfg.get("use_kf", True)) if isinstance(cfg, dict) else True
        kf_state = ckpt.get("kf_state_dict")

        if not use_kf or kf_state is None:
            return policy, None

        kf = LearnableKalman(dim=n_assets).to(self.device)
        kf.load_state_dict(kf_state)
        return policy, kf

    def _resolve_window_size(self, ckpt: dict) -> int:
        cfg = ckpt.get("cfg", {})
        if isinstance(cfg, dict) and cfg.get("window_size") is not None:
            return int(cfg["window_size"])
        return self.default_window_size

    @staticmethod
    def _weights_to_numpy(weights_list: list[torch.Tensor] | None) -> np.ndarray:
        if not weights_list:
            raise ValueError("Evaluation did not return weights.")
        return np.vstack([w.detach().cpu().numpy() for w in weights_list]).astype(np.float32)

    @staticmethod
    def _dates_from_view(test_view, n_rows: int) -> list[str]:
        if hasattr(test_view, "parent") and hasattr(test_view, "row_idx"):
            idx = test_view.parent.returns_df.index[test_view.row_idx]
            dates = [d.strftime("%Y-%m-%d") for d in idx]
            if len(dates) == n_rows:
                return dates
        return [str(i) for i in range(n_rows)]

    @staticmethod
    def _tickers_from_view(test_view, n_assets: int) -> list[str]:
        if hasattr(test_view, "parent") and hasattr(test_view.parent, "returns_df"):
            cols = list(test_view.parent.returns_df.columns)
            if len(cols) == n_assets:
                return [str(c) for c in cols]
        return [f"asset_{i}" for i in range(n_assets)]

    @staticmethod
    def _write_weights_csv(path: Path, dates: list[str], tickers: list[str], weights: np.ndarray) -> None:
        with path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["date", *tickers])
            for i in range(weights.shape[0]):
                writer.writerow([dates[i], *[f"{float(v):.8f}" for v in weights[i]]])

    @staticmethod
    def _write_weights_mat(
        path: Path,
        *,
        dates: list[str],
        tickers: list[str],
        weights: np.ndarray,
        metrics: dict,
    ) -> None:
        savemat(
            path,
            {
                "weights": weights.astype(np.float32),
                "dates": np.asarray(dates, dtype=object),
                "tickers": np.asarray(tickers, dtype=object),
                "metric_names": np.asarray(list(metrics.keys()), dtype=object),
                "metric_values": np.asarray(list(metrics.values()), dtype=np.float32),
            },
        )

    @staticmethod
    def _plot_weights(path: Path, dates: list[str], tickers: list[str], weights: np.ndarray) -> None:
        fig, ax = plt.subplots(figsize=(12, 4))
        x = np.arange(weights.shape[0], dtype=np.int32)
        for i, ticker in enumerate(tickers):
            ax.plot(x, weights[:, i], label=ticker, linewidth=1.2)

        if len(x) > 0:
            tick_step = max(1, len(x) // 8)
            xticks = x[::tick_step]
            ax.set_xticks(xticks)
            ax.set_xticklabels([dates[i] for i in xticks], rotation=45, ha="right")

        ax.set_title("Policy Weights on Test Split")
        ax.set_xlabel("Test Date")
        ax.set_ylabel("Weight")
        ax.set_ylim(0.0, 1.0)
        ax.grid(alpha=0.3)
        ax.legend(ncol=max(1, min(5, len(tickers))), fontsize=8)
        fig.tight_layout()
        fig.savefig(path, dpi=150)
        plt.close(fig)

    def evaluate_checkpoint(self, checkpoint_path: str | Path) -> PolicyEvalOutput:
        checkpoint = Path(checkpoint_path)
        ckpt = self._load_checkpoint(checkpoint)

        policy, kf = self._build_models(ckpt)
        window_size = self._resolve_window_size(ckpt)

        metrics = Trainer.evaluate_full_run(
            policy,
            kf,
            self.test_view,
            window_size=window_size,
            device=str(self.device),
            sample_policy=False,
            return_weights=True,
        )
        weights = self._weights_to_numpy(metrics["weights"])
        dates = self._dates_from_view(self.test_view, weights.shape[0])
        tickers = self._tickers_from_view(self.test_view, weights.shape[1])

        stem = checkpoint.stem
        csv_path = self.out_dir / f"{stem}_weights.csv"
        mat_path = self.out_dir / f"{stem}_weights.mat"
        plot_path = self.out_dir / f"{stem}_weights.png"

        self._write_weights_csv(csv_path, dates, tickers, weights)
        self._write_weights_mat(
            mat_path,
            dates=dates,
            tickers=tickers,
            weights=weights,
            metrics={
                "sharpe": float(metrics["sharpe"]),
                "total_return": float(metrics["total_reward"]),
                "mean_return": float(metrics["mean_return"]),
                "std_return": float(metrics["std_return"]),
                "max_drawdown": float(metrics["max_drawdown"]),
            },
        )
        self._plot_weights(plot_path, dates, tickers, weights)

        return PolicyEvalOutput(
            checkpoint=checkpoint,
            sharpe=float(metrics["sharpe"]),
            total_return=float(metrics["total_reward"]),
            mean_return=float(metrics["mean_return"]),
            std_return=float(metrics["std_return"]),
            max_drawdown=float(metrics["max_drawdown"]),
            weights_csv=csv_path,
            weights_mat=mat_path,
            plot_png=plot_path,
        )

    def evaluate_many(self, checkpoints: Iterable[str | Path]) -> list[PolicyEvalOutput]:
        return [self.evaluate_checkpoint(path) for path in checkpoints]


def _collect_checkpoints(
    *,
    checkpoint_paths: list[str],
    checkpoint_dir: str | None,
    glob_pattern: str,
) -> list[Path]:
    paths = [Path(p) for p in checkpoint_paths]
    if checkpoint_dir:
        paths.extend(sorted(Path(checkpoint_dir).glob(glob_pattern)))

    dedup = []
    seen = set()
    for path in paths:
        resolved = path.resolve()
        if resolved in seen:
            continue
        seen.add(resolved)
        dedup.append(path)

    missing = [p for p in dedup if not p.exists()]
    if missing:
        missing_str = ", ".join(str(p) for p in missing)
        raise FileNotFoundError(f"Checkpoint path(s) not found: {missing_str}")
    if not dedup:
        raise ValueError("No checkpoints found. Use --checkpoint or --checkpoint-dir.")
    return dedup


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate policy checkpoints on test data and save weights + matplotlib plots."
    )
    parser.add_argument(
        "--checkpoint",
        action="append",
        default=[],
        help="Path to a checkpoint .pt file. Can be passed multiple times.",
    )
    parser.add_argument(
        "--checkpoint-dir",
        default=None,
        help="Optional directory to scan for checkpoint files.",
    )
    parser.add_argument(
        "--glob",
        default="policy_combo*_seed*.pt",
        help="Glob used with --checkpoint-dir.",
    )
    parser.add_argument("--out-dir", default="policy_weight_outputs")
    parser.add_argument("--cache-path", default="returns.parquet")
    parser.add_argument(
        "--no-cache",
        action="store_true",
        help="Disable parquet cache and always download fresh data from the API.",
    )
    parser.add_argument("--tickers", nargs="+", default=list(DEFAULT_TICKERS))
    parser.add_argument("--start-date", default=DEFAULT_START_DATE)
    parser.add_argument("--end-date", default=DEFAULT_END_DATE)
    parser.add_argument("--price-field", default=DEFAULT_PRICE_FIELD)
    parser.add_argument("--train-end", default=DEFAULT_TRAIN_END)
    parser.add_argument("--val-end", default=DEFAULT_VAL_END)
    parser.add_argument("--default-window-size", type=int, default=10)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()

    checkpoints = _collect_checkpoints(
        checkpoint_paths=args.checkpoint,
        checkpoint_dir=args.checkpoint_dir,
        glob_pattern=args.glob,
    )
    evaluator = PolicyCheckpointMatPlotter(
        out_dir=args.out_dir,
        cache_path=None if args.no_cache else args.cache_path,
        tickers=args.tickers,
        start_date=args.start_date,
        end_date=args.end_date,
        price_field=args.price_field,
        train_end=args.train_end,
        val_end=args.val_end,
        default_window_size=args.default_window_size,
        device=args.device,
    )

    for result in evaluator.evaluate_many(checkpoints):
        print(
            f"{result.checkpoint}: "
            f"sharpe={result.sharpe:.4f}, "
            f"total_return={result.total_return:.4f}, "
            f"weights_csv={result.weights_csv}, "
            f"weights_mat={result.weights_mat}, "
            f"plot_png={result.plot_png}"
        )


if __name__ == "__main__":
    main()

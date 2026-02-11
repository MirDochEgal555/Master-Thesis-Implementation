from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, Optional, Tuple

import numpy as np
import torch


@dataclass
class BaselineMetrics:
    total_reward: float
    sharpe: float
    mean_return: float
    std_return: float
    pure_returns: np.ndarray
    weights: Optional[Iterable[np.ndarray]] = None
    max_drawdown: float = 0.0


class BaselinePortfolioEvaluator:
    """
    Evaluate equal-weight, inverse-volatility, and minimum-variance baselines
    on validation and test datasets in one call.
    """
    def __init__(
        self,
        *,
        annualization: int = 252,
        eps: float = 1e-6,
        long_only: bool = True,
    ):
        self.annualization = int(annualization)
        self.eps = float(eps)
        self.long_only = bool(long_only)

    def evaluate(
        self,
        val_view,
        test_view,
        *,
        include_weights: bool = False,
    ) -> Dict[str, Dict[str, BaselineMetrics]]:
        return {
            "val": self._evaluate_split(val_view, include_weights=include_weights),
            "test": self._evaluate_split(test_view, include_weights=include_weights),
        }

    def _evaluate_split(
        self,
        data_view,
        *,
        include_weights: bool = False,
    ) -> Dict[str, BaselineMetrics]:
        returns, covs = self._returns_and_covs(data_view)
        return self._evaluate_returns(returns, covs, include_weights=include_weights)

    def _returns_and_covs(self, data_view) -> Tuple[np.ndarray, np.ndarray]:
        if hasattr(data_view, "as_numpy"):
            returns = data_view.as_numpy()
            covs = data_view.precompute_expanding_cov(diag=False, eps=self.eps)
            return returns, covs

        if torch.is_tensor(data_view):
            returns = data_view.detach().cpu().numpy()
        else:
            returns = np.asarray(data_view, dtype=np.float32)

        covs = self._precompute_expanding_cov(returns, eps=self.eps)
        return returns, covs

    def _evaluate_returns(
        self,
        returns: np.ndarray,
        covs: np.ndarray,
        *,
        include_weights: bool = False,
    ) -> Dict[str, BaselineMetrics]:
        returns = np.asarray(returns, dtype=np.float32)
        N, K = returns.shape

        w_eq_prev = np.full(K, 1.0 / K, dtype=np.float32)
        w_inv_prev = w_eq_prev.copy()
        w_min_prev = w_eq_prev.copy()

        eq_pure = np.empty(N, dtype=np.float32)
        inv_pure = np.empty(N, dtype=np.float32)
        min_pure = np.empty(N, dtype=np.float32)

        eq_weights = [] if include_weights else None
        inv_weights = [] if include_weights else None
        min_weights = [] if include_weights else None

        ones = np.ones(K, dtype=np.float32)

        for t in range(N):
            z_t = returns[t]
            cov_t = covs[t]

            w_eq = w_eq_prev

            diag = np.diag(cov_t).astype(np.float32, copy=False)
            vol = np.sqrt(np.maximum(diag, self.eps))
            inv_vol = 1.0 / vol
            w_inv = inv_vol / (inv_vol.sum() + self.eps)

            w_min = self._min_variance_weights(cov_t, ones)

            eq_pure[t] = float(np.dot(w_eq_prev, z_t))
            inv_pure[t] = float(np.dot(w_inv_prev, z_t))
            min_pure[t] = float(np.dot(w_min_prev, z_t))

            if include_weights:
                eq_weights.append(w_eq.copy())
                inv_weights.append(w_inv.copy())
                min_weights.append(w_min.copy())

            w_eq_prev = w_eq
            w_inv_prev = w_inv
            w_min_prev = w_min

        return {
            "equal_weight": self._metrics_from_pure(eq_pure, eq_weights),
            "inverse_volatility": self._metrics_from_pure(inv_pure, inv_weights),
            "minimum_variance": self._metrics_from_pure(min_pure, min_weights),
        }

    def _min_variance_weights(self, cov: np.ndarray, ones: np.ndarray) -> np.ndarray:
        cov = np.asarray(cov, dtype=np.float32)
        cov = 0.5 * (cov + cov.T)
        cov = cov + self.eps * np.eye(cov.shape[0], dtype=np.float32)

        try:
            raw = np.linalg.solve(cov, ones)
        except np.linalg.LinAlgError:
            raw = np.linalg.pinv(cov) @ ones

        denom = float(ones @ raw)
        if denom <= 0.0 or not np.isfinite(denom):
            w = np.full_like(ones, 1.0 / len(ones))
        else:
            w = raw / denom

        if self.long_only:
            w = np.maximum(w, 0.0)
            total = float(w.sum())
            if total <= 0.0 or not np.isfinite(total):
                w = np.full_like(ones, 1.0 / len(ones))
            else:
                w = w / total

        return w.astype(np.float32, copy=False)

    def _metrics_from_pure(
        self,
        pure: np.ndarray,
        weights: Optional[Iterable[np.ndarray]],
    ) -> BaselineMetrics:
        pure_np = np.asarray(pure, dtype=np.float32)
        pure_t = torch.as_tensor(pure, dtype=torch.float32)
        mean = float(pure_t.mean())
        std = float(pure_t.std() + 1e-8)
        sharpe = mean / std * np.sqrt(self.annualization)
        total_reward = float(1.0 + pure_t.sum())
        equity_curve = np.cumprod(1.0 + pure_np)
        running_peak = np.maximum.accumulate(equity_curve)
        drawdowns = equity_curve / (running_peak + 1e-12) - 1.0
        max_drawdown = float(-drawdowns.min()) if drawdowns.size else 0.0

        return BaselineMetrics(
            total_reward=total_reward,
            sharpe=float(sharpe),
            mean_return=mean,
            std_return=float(std),
            pure_returns=pure_np,
            weights=weights,
            max_drawdown=max_drawdown,
        )

    @staticmethod
    def _precompute_expanding_cov(returns: np.ndarray, eps: float = 1e-6) -> np.ndarray:
        returns = np.asarray(returns, dtype=np.float32)
        N, K = returns.shape
        covs = np.zeros((N, K, K), dtype=np.float32)

        mean = np.zeros(K, dtype=np.float32)
        M2 = np.zeros((K, K), dtype=np.float32)
        n = 0
        eye = np.eye(K, dtype=np.float32)

        for t in range(N):
            if n >= 2:
                cov = M2 / (n - 1)
            else:
                cov = eye.copy()
            covs[t] = cov + eps * eye

            x = returns[t]
            n += 1
            delta = x - mean
            mean += delta / n
            delta2 = x - mean
            M2 += np.outer(delta, delta2)

        return covs


if __name__ == "__main__":
    import csv
    from pathlib import Path

    from portfolio_rl.data import YahooConfig, YahooReturnsDataset

    tickers = ["JPM", "JNJ", "XOM", "PG", "MSFT"]
    ycfg = YahooConfig(
        tickers=tickers,
        start_date="2022-01-01",
        end_date="2024-12-31",
        price_field="Close",
        cache_path="returns.parquet",
    )

    dataset = YahooReturnsDataset(ycfg)
    _, val_view, test_view = dataset.split_by_date(
        train_end="2023-03-24",
        val_end="2023-09-30",
    )

    evaluator = BaselinePortfolioEvaluator()
    results = evaluator.evaluate(val_view, test_view, include_weights=True)

    def _write_metrics_csv(path: Path, split_results):
        with path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["strategy", "sharpe", "total_return", "mean_return", "std_return", "max_drawdown"])
            for key, metrics in split_results.items():
                writer.writerow([
                    key,
                    f"{metrics.sharpe:.6f}",
                    f"{metrics.total_reward:.6f}",
                    f"{metrics.mean_return:.10f}",
                    f"{metrics.std_return:.10f}",
                    f"{metrics.max_drawdown:.10f}",
                ])

    def _write_weights_csv(path: Path, dates, weights, tickers_list):
        if weights is None:
            return
        with path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            header = ["date"] + list(tickers_list)
            writer.writerow(header)
            for i, w in enumerate(weights):
                date_val = dates[i] if dates is not None else i
                if hasattr(date_val, "strftime"):
                    date_val = date_val.strftime("%Y-%m-%d")
                writer.writerow([date_val] + [f"{float(x):.8f}" for x in w])

    def _print_split(name, split_results):
        print(f"\n=== {name.upper()} ===")
        for key, metrics in split_results.items():
            print(
                f"{key}: sharpe={metrics.sharpe:.3f}, "
                f"total_return={metrics.total_reward:.3f}, "
                f"mean={metrics.mean_return:.6f}, "
                f"std={metrics.std_return:.6f}, "
                f"maxdd={metrics.max_drawdown:.6f}"
            )

    _print_split("val", results["val"])
    _print_split("test", results["test"])

    out_dir = Path(".")
    _write_metrics_csv(out_dir / "baseline_metrics_val.csv", results["val"])
    _write_metrics_csv(out_dir / "baseline_metrics_test.csv", results["test"])

    val_dates = None
    test_dates = None
    if hasattr(val_view, "parent") and hasattr(val_view, "row_idx"):
        val_dates = list(val_view.parent.returns_df.index[val_view.row_idx])
    if hasattr(test_view, "parent") and hasattr(test_view, "row_idx"):
        test_dates = list(test_view.parent.returns_df.index[test_view.row_idx])

    _write_weights_csv(
        out_dir / "weights_inverse_vol_val.csv",
        val_dates,
        results["val"]["inverse_volatility"].weights,
        tickers,
    )
    _write_weights_csv(
        out_dir / "weights_min_variance_val.csv",
        val_dates,
        results["val"]["minimum_variance"].weights,
        tickers,
    )
    _write_weights_csv(
        out_dir / "weights_inverse_vol_test.csv",
        test_dates,
        results["test"]["inverse_volatility"].weights,
        tickers,
    )
    _write_weights_csv(
        out_dir / "weights_min_variance_test.csv",
        test_dates,
        results["test"]["minimum_variance"].weights,
        tickers,
    )

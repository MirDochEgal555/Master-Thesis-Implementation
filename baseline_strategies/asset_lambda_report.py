import argparse
from typing import Iterable, Dict

import numpy as np

from portfolio_rl.data import YahooConfig, YahooReturnsDataset


class AssetLambdaReport:
    """
    Report per-asset mean score: return - lambda * variance.
    Variance is computed as an expanding estimate using past returns only.
    """

    def __init__(self, returns: np.ndarray, tickers: Iterable[str]):
        self.returns = returns.astype(np.float32, copy=False)
        self.tickers = list(tickers)

    @classmethod
    def from_yahoo(
        cls,
        tickers: Iterable[str],
        start_date: str,
        end_date: str,
        train_end: str | None = "2023-03-24",
        val_end: str | None = "2023-09-30",
        price_field: str = "Close",
        cache_path: str | None = "returns.parquet",
    ) -> "AssetLambdaReport":
        cfg = YahooConfig(
            tickers=list(tickers),
            start_date=start_date,
            end_date=end_date,
            price_field=price_field,
            cache_path=cache_path,
        )
        dataset = YahooReturnsDataset(cfg)
        if train_end is None or val_end is None:
            returns = dataset.returns_df.to_numpy(dtype=np.float32)
        else:
            train_view, val_view, test_view = dataset.split_by_date(train_end=train_end, val_end=val_end)
            returns = train_view.as_numpy()
        return cls(returns=returns, tickers=dataset.returns_df.columns)

    @staticmethod
    def _expanding_variance(returns: np.ndarray, eps: float = 1e-6) -> np.ndarray:
        n_rows, n_assets = returns.shape
        var = np.zeros((n_rows, n_assets), dtype=np.float32)
        mean = np.zeros(n_assets, dtype=np.float32)
        m2 = np.zeros(n_assets, dtype=np.float32)
        n = 0
        for t in range(n_rows):
            if n >= 2:
                var[t] = m2 / (n - 1)
            else:
                var[t] = 1.0
            x = returns[t]
            n += 1
            delta = x - mean
            mean += delta / n
            delta2 = x - mean
            m2 += delta * delta2
        return var + eps

    def mean_scores(self, lambdas: Iterable[float]) -> Dict[float, np.ndarray]:
        variances = self._expanding_variance(self.returns)
        results: Dict[float, np.ndarray] = {}
        for lam in lambdas:
            scores = self.returns - float(lam) * variances
            results[float(lam)] = scores.mean(axis=0)
        return results

    def print_report(self, lambdas: Iterable[float]) -> None:
        results = self.mean_scores(lambdas)
        lambda_list = list(results.keys())
        header = ["asset"] + [f"lambda_{lam:g}" for lam in lambda_list]
        print(",".join(header))
        for i, ticker in enumerate(self.tickers):
            row = [ticker] + [f"{results[lam][i]:.6f}" for lam in lambda_list]
            print(",".join(row))


def _parse_lambdas(raw: str) -> list[float]:
    return [float(x) for x in raw.split(",") if x.strip()]


def main() -> None:
    parser = argparse.ArgumentParser(description="Per-asset mean (return - lambda * variance) report.")
    parser.add_argument(
        "--tickers",
        nargs="+",
        default=["JPM", "JNJ", "XOM", "PG", "MSFT"],
        help="List of tickers.",
    )
    parser.add_argument(
        "--lambdas",
        type=_parse_lambdas,
        default="0.001,0.1,1.0,10.0",
        help="Comma-separated lambda values.",
    )
    parser.add_argument("--start-date", default="2022-01-01")
    parser.add_argument("--end-date", default="2024-12-31")
    parser.add_argument("--train-end", default="2023-03-24")
    parser.add_argument("--val-end", default="2023-09-30")
    parser.add_argument("--price-field", default="Close")
    parser.add_argument("--cache-path", default="returns.parquet")
    args = parser.parse_args()

    report = AssetLambdaReport.from_yahoo(
        tickers=args.tickers,
        start_date=args.start_date,
        end_date=args.end_date,
        train_end=args.train_end,
        val_end=args.val_end,
        price_field=args.price_field,
        cache_path=args.cache_path,
    )
    report.print_report(args.lambdas)


if __name__ == "__main__":
    main()

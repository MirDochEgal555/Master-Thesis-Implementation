# portfolio_rl/data.py
from dataclasses import dataclass
from typing import List, Tuple, Optional
import numpy as np
import pandas as pd
import torch

import yfinance as yf


@dataclass
class YahooConfig:
    tickers: List[str]
    start_date: str
    end_date: str
    price_field: str = "Close"     # "Adj Close" also common
    cache_path: Optional[str] = None  # e.g. "data/returns.parquet"


class YahooReturnsDataset:
    """
    Downloads prices from yfinance, computes daily returns, and provides
    random contiguous windows of length T as torch tensors [T, K].
    """
    def __init__(self, cfg: YahooConfig):
        self.cfg = cfg
        self.returns_df = self._load_or_download_returns(cfg)
        self.returns_np = self.returns_df.to_numpy(dtype=np.float32)  # [N, K]
        self.K = self.returns_np.shape[1]
        self.N = self.returns_np.shape[0]

    def _load_or_download_returns(self, cfg: YahooConfig) -> pd.DataFrame:
        if cfg.cache_path is not None:
            try:
                df = pd.read_parquet(cfg.cache_path)
                # assume cached is returns with columns tickers
                return df
            except Exception:
                pass

        stock_data = yf.download(cfg.tickers, start=cfg.start_date, end=cfg.end_date, auto_adjust=False)

        # MultiIndex columns: ('Close', 'MSFT') etc.
        if isinstance(stock_data.columns, pd.MultiIndex):
            close_prices = stock_data[cfg.price_field]
        else:
            # single ticker case
            close_prices = stock_data[[cfg.price_field]]

        daily_returns = close_prices.pct_change().dropna()

        if cfg.cache_path is not None:
            daily_returns.to_parquet(cfg.cache_path)

        return daily_returns

    def split_by_date(
        self,
        train_end: str,
        val_end: str,
    ) -> Tuple["YahooReturnsDatasetView", "YahooReturnsDatasetView", "YahooReturnsDatasetView"]:
        """
        Returns (train_view, val_view, test_view) as lightweight views.
        """
        idx = self.returns_df.index
        train_mask = idx <= pd.Timestamp(train_end)
        val_mask = (idx > pd.Timestamp(train_end)) & (idx <= pd.Timestamp(val_end))
        test_mask = idx > pd.Timestamp(val_end)

        return (
            YahooReturnsDatasetView(self, np.where(train_mask)[0]),
            YahooReturnsDatasetView(self, np.where(val_mask)[0]),
            YahooReturnsDatasetView(self, np.where(test_mask)[0]),
        )


class YahooReturnsDatasetView:
    """
    A view of the parent dataset restricted to specific row indices.
    """
    def __init__(self, parent: YahooReturnsDataset, row_idx: np.ndarray):
        self.parent = parent
        self.row_idx = row_idx
        self.K = parent.K
        self.N = len(row_idx)

    def precompute_expanding_cov(self, diag: bool = False, eps: float = 1e-6):
        rets = self.as_numpy()  # [N, K] for this split only, time‑ordered
        N, K = rets.shape
        covs = np.zeros((N, K, K), dtype=np.float32)

        mean = np.zeros(K, dtype=np.float32)
        M2 = np.zeros((K, K), dtype=np.float32)
        n = 0
        eye = np.eye(K, dtype=np.float32)

        for t in range(N):
            # cov from returns[0:t], i.e., strictly previous returns only
            if n >= 2:
                cov = M2 / (n - 1)
                if diag:
                    cov = np.diag(np.diag(cov))
            else:
                cov = eye.copy()
            covs[t] = cov + eps * eye

            # update with returns[t]
            x = rets[t]
            n += 1
            delta = x - mean
            mean += delta / n
            delta2 = x - mean
            M2 += np.outer(delta, delta2)

        return covs


    def sample_window(self, T: int, device: str = "cpu") -> torch.Tensor:
        """
        Random contiguous window of length T: [T, K]
        """
        if self.N <= T:
            raise ValueError(f"Not enough rows in split: N={self.N}, need T={T}")

        start_pos = np.random.randint(0, self.N - T)
        rows = self.row_idx[start_pos : start_pos + T]
        window = self.parent.returns_np[rows]  # [T, K]
        return torch.from_numpy(window).to(device)

    def iter_windows(self, T: int, stride: int = 1):
        for i in range(0, self.N - T + 1, stride):
            rows = self.row_idx[i : i + T]
            yield torch.from_numpy(self.parent.returns_np[rows])

    def iter_windows_with_cov(self, T: int, stride: int, covs: np.ndarray):
        for i in range(0, self.N - T + 1, stride):
            rows = self.row_idx[i : i + T]
            x = torch.from_numpy(self.parent.returns_np[rows])     # [T, K]
            cov_win = torch.from_numpy(covs[i : i + T])            # [T, K, K]
            yield x, cov_win


    def as_numpy(self) -> np.ndarray:
        """All rows in this split as numpy [N, K]."""
        return self.parent.returns_np[self.row_idx]

    def as_tensor(self, device: str = "cpu") -> torch.Tensor:
        """All rows in this split as tensor [N, K]."""
        return torch.from_numpy(self.as_numpy()).to(device)
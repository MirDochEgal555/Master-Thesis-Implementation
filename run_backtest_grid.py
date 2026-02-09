from portfolio_rl.backtest_grid import grid_search
from portfolio_rl.data import YahooConfig, YahooReturnsDataset

if __name__ == "__main__":
    tickers = ["JPM", "JNJ", "XOM", "PG", "MSFT"]
    ycfg = YahooConfig(
        tickers=tickers,
        start_date="2022-01-01",
        end_date="2024-12-31",
        price_field="Close",
        cache_path="returns.parquet",
    )
    dataset = YahooReturnsDataset(ycfg)
    train_view, val_view, test_view = dataset.split_by_date(
        train_end="2023-03-24",
        val_end="2023-09-30",
    )
    train_covs = train_view.precompute_expanding_cov(diag=True)
    data_bundle = (train_view, val_view, test_view, train_covs)

    seeds = list(range(50))
    window_sizes = [1]
    lambdas = [0.1]

    results, summary = grid_search(seeds, window_sizes, lambdas, max_workers=4, data_bundle=data_bundle)

    print("\n=== TOP CONFIGS (mean over seeds) ===")
    for row in summary[:10]:
        print(row)

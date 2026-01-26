from portfolio_rl.backtest_grid import grid_search

if __name__ == "__main__":
    seeds = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20]
    window_sizes = [1]
    lambdas = [0.001]

    results, summary = grid_search(seeds, window_sizes, lambdas, max_workers=4)

    print("\n=== TOP CONFIGS (mean over seeds) ===")
    for row in summary[:10]:
        print(row)
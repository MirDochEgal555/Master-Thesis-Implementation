from portfolio_rl.grid_search import grid_search

if __name__ == "__main__":
    seeds = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
    window_sizes = [1]
    lambdas = [0.001, 0.1, 1, 10] 
    networksizes = [64]
    learnrates = [1e-3]

    results, summary = grid_search(seeds, window_sizes, lambdas, networksizes, learnrates, max_workers=None)

    print("\n=== TOP CONFIGS (mean over seeds) ===")
    for row in summary[:10]:
        print(row)
# Master Thesis Implementation

Research code for reinforcement-learning portfolio optimization with optional
Kalman filtering and dynamics modeling. The repository includes scripts for
single runs, grid searches, larger sweeps, baseline strategy evaluation, and
post-run reporting/plotting.

## Project Layout
- `main.py`: Single-run training/backtest entry point (`portfolio_rl.backtest_seed.run_one`).
- `run_grid_search.py`: Parallel grid search across seeds, window size, lambda, network size, and learning rate.
- `run_backtest_grid.py`: Parallel backtest grid across seeds, window size, and lambda.
- `sweep_grid.py`: Broader sweep utility that appends results to `sweep_results.csv`.
- `baselines.py`: Baseline evaluation (equal-weight, inverse-volatility, minimum-variance).
- `grid_search_report.py`: Parses `grid_search_lines.txt` into CSV summaries.
- `plot_learning_stats.py`: Plots validation Sharpe, normalized losses, and discounted reward from a stats CSV.
- `learning_comparison.py`: Compares learning curves across multiple runs.
- `kalman_filter_fitting.py`: Q/R sweep for Kalman filter diagnostics and heatmaps.
- `portfolio_rl/`: Core package (data loading, models, trainer, rollouts, Kalman, dynamics, backtesting).
- `requirements.txt`: Python dependencies.

## Setup
```bash
python -m venv .venv
```
Windows (PowerShell):
```powershell
.venv\Scripts\Activate.ps1
```
macOS/Linux:
```bash
source .venv/bin/activate
```
Install dependencies:
```bash
pip install -r requirements.txt
```

## Quick Start
Single run (uses hard-coded settings in `main.py`):
```bash
python main.py
```

Grid search:
```bash
python run_grid_search.py
```

Backtest grid:
```bash
python run_backtest_grid.py
```

Baseline strategies:
```bash
python baselines.py
```

## Common Outputs
- `returns.parquet`: Cached return series downloaded from `yfinance` (created on first run).
- `weights.txt`: Latest test weights from `main.py` or `run_backtest_grid.py`.
- `learning_stats_kf.csv`: Per-epoch training/validation stats from `main.py`.
- `grid_search_lines.txt`: Line-by-line completed grid-search jobs.
- `grid_search_summary.txt`: Aggregated summary from `run_grid_search.py`.
- `backtest_grid.txt`: Aggregated summary from `run_backtest_grid.py`.
- `sweep_results.csv`: Incremental sweep results from `sweep_grid.py`.
- `baseline_metrics_val.csv`, `baseline_metrics_test.csv`: Baseline metrics.

## Reporting and Plotting
Generate CSV reports from `grid_search_lines.txt`:
```bash
python grid_search_report.py
```
Generated files:
- `grid_search_raw.csv`
- `grid_search_seed_avg.csv`
- `grid_search_lambda_avg.csv`
- `grid_search_window_avg.csv`
- `grid_search_networksize_avg.csv`
- `grid_search_learnrate_avg.csv`

Plot learning statistics from a run:
```bash
python plot_learning_stats.py learning_stats_kf.csv --out-dir .
```

Compare multiple learning runs:
```bash
python learning_comparison.py --run "A=path/to/run1.csv,path/to/run2.csv" --run "B=path/to/run3.csv" --out-path comparison.png --global-step --smooth 3
```

Kalman Q/R fitting sweep (writes `kalman_fitting_results_sorted.txt` and heatmaps):
```bash
python kalman_filter_fitting.py
```

## Notes
- Most driver scripts use fixed parameter lists directly in the file. Edit those lists to run your own experiments.
- Process-based parallel scripts rely on `if __name__ == "__main__"` entry points; run them as scripts (not from interactive notebooks without proper guards).

## License
MIT License. See `LICENSE`.

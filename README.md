# Master Thesis Implementation

Research code for reinforcement-learning portfolio optimization with optional
Kalman filtering and dynamics modeling. The active workflow is centered on
ablations, sweeps, baseline evaluation, and post-run plotting of checkpoints
and losses.

## Project Layout
- `ablations.py`: Ablation sweep over Kalman and simulation settings; appends to `ablations.csv`.
- `sweep_grid.py`: Broad hyperparameter sweep utility that appends to `sweep_results.csv`.
- `total_results.py`: Full experiment runner that writes both `total_results_final.csv` and `total_results_best.csv`, and can save checkpoints/loss traces.
- `3000 updates/`: Stored `total_results.py` artifacts for the 3000-update run (tables, policies, checkpoints, losses, plots).
- `10000 updates/`: Stored `total_results.py` artifacts for the 10000-update run (tables, policies, checkpoints, losses).
- `baselines.py`: Baseline evaluation (equal-weight, inverse-volatility, minimum-variance).
- `kalman_filter_fitting.py`: Q/R sweep for Kalman filter diagnostics and heatmaps.
- `kalman_fitting/`: Stored Kalman fitting artifacts (`kalman_fitting_results_sorted.txt` and heatmaps).
- `plot_losses_nonzero.py`: Plot normalized non-zero loss curves from a losses CSV.
- `policy_checkpoint_matplot.py`: Evaluate saved checkpoint(s) and export weights as CSV/MAT/PNG.
- `portfolio_rl/`: Core package (data loading, models, trainer, rollouts, Kalman, dynamics, backtesting).
- `archive/`: Legacy scripts and historical experiment artifacts moved out of the active workflow.
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
Run ablations:
```bash
python ablations.py
```

Run broad sweep:
```bash
python sweep_grid.py
```

Run final-vs-best evaluation sweep:
```bash
python total_results.py
```

Baseline strategies:
```bash
python baselines.py
```

Kalman Q/R fitting sweep (writes artifacts in `kalman_fitting/`):
```bash
python kalman_filter_fitting.py
```

## Common Outputs
- `returns.parquet`: Cached return series downloaded from `yfinance` (created on first run).
- `ablations.csv`: Aggregated ablation results from `ablations.py`.
- `sweep_results.csv`: Incremental sweep results from `sweep_grid.py`.
- `3000 updates/total_results_final.csv`, `3000 updates/total_results_best.csv`: `total_results.py` metrics for the 3000-update configuration.
- `10000 updates/total_results_final.csv`, `10000 updates/total_results_best.csv`: `total_results.py` metrics for the 10000-update configuration.
- `3000 updates/losses/losses_combo*_seed*.csv`, `10000 updates/losses/losses_combo*_seed*.csv`: Per-epoch loss traces from `total_results.py`.
- `baseline_metrics_val.csv`, `baseline_metrics_test.csv`: Baseline metrics.
- `weights_inverse_vol_*.csv`, `weights_min_variance_*.csv`: Baseline strategy weights.
- `kalman_fitting/kalman_fitting_results_sorted.txt`, `kalman_fitting/*_heatmap.png`: Ranked Kalman Q/R diagnostics and heatmaps.
- `policy_weight_outputs/*_weights.csv|.mat|.png`: Checkpoint evaluation outputs.

## Reporting and Plotting
Plot normalized non-zero losses from a losses CSV:
```bash
python plot_losses_nonzero.py grid_search_results/losses/losses_combo1_seed4.csv --out losses_nonzero.png
```

Evaluate one or more policy checkpoints and export weights + plots:
```bash
python policy_checkpoint_matplot.py --checkpoint-dir grid_search_results/final_policies --out-dir policy_weight_outputs
```

## Notes
- Most driver scripts use fixed parameter lists directly in the file. Edit those lists to run your own experiments.
- Legacy entrypoints (single-run and older grid-search/report scripts) are kept under `archive/`.
- Process-based parallel scripts rely on `if __name__ == "__main__"` entry points; run them as scripts (not from interactive notebooks without proper guards).

## License
MIT License. See `LICENSE`.

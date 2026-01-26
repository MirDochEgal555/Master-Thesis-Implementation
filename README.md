# Master Thesis Implementation

Research code for a reinforcement-learning portfolio optimizer with optional
Kalman filtering and dynamics modeling. Includes grid-search utilities to
evaluate validation/test Sharpe and return metrics across window sizes and
risk aversion (lambda).

## Project Layout
- `main.py`: Single-run backtest/seed entry point (uses `portfolio_rl/backtest_seed.py`).
- `run_grid_search.py`: Parallel grid search over seeds, window sizes, and lambda.
- `run_backtest_grid.py`: Backtest grid over many seeds for a fixed config.
- `portfolio_rl/`: Core RL, data, and model code (policy/value nets, trainer,
  Kalman filter, dynamics model, rollouts).
- `grid_search_results/`: Example outputs plus `grid_search_report.py`.
- `requirements.txt`: Python dependencies.
- `returns.parquet`: Cached return series (created on first run).

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
Then install deps:
```bash
pip install -r requirements.txt
```

## Usage
Single run (uses settings in `main.py`):
```bash
python main.py
```

Grid search:
```bash
python run_grid_search.py
```

Backtest grid over many seeds:
```bash
python run_backtest_grid.py
```

## Outputs
- `grid_search_lines.txt`: line-by-line results for each config (seed/window/lambda).
- `grid_search_summary.txt`: aggregated mean Sharpe per (window, lambda).
- `backtest_grid.txt`: aggregated mean Sharpe per (window, lambda) for backtests.
- `weights.txt`: weights from the most recently completed run.

## Reporting
Generate CSV summaries from a `grid_search_lines.txt` file:
```bash
python grid_search_results/grid_search_report.py
```
Outputs:
- `grid_search_raw.csv`
- `grid_search_seed_avg.csv`
- `grid_search_lambda_avg.csv`
- `grid_search_window_avg.csv`

## Notes
- Data is pulled via `yfinance` and cached to `returns.parquet`.
- Default settings are CPU-first and sized for experimentation; adjust
  `portfolio_rl/config.py`, `main.py`, or the run scripts for larger sweeps.

## License
MIT License. See `LICENSE`.

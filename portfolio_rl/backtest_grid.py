# grid_search.py
from itertools import product
from concurrent.futures import ProcessPoolExecutor, as_completed
import numpy as np
import multiprocessing as mp


from .backtest_seed import run_one
from .config import TrainConfig


def grid_search(seeds, window_sizes, lambdas, max_workers=4, line_log_path="grid_search_lines.txt", data_bundle=None):
    jobs = list(product(seeds, window_sizes, lambdas))
    results = []
    mp.set_start_method("spawn", force=True)

    with ProcessPoolExecutor(max_workers=max_workers) as ex:
        futs = [ex.submit(run_one, s, w, l, data_bundle=data_bundle) for (s, w, l) in jobs]
        with open(line_log_path, "w", encoding="utf-8") as log_f:
            for f in as_completed(futs):
                r = f.result()
                results.append(r)
                line = (
                    f"done seed={r['seed']} w={r['window_size']} lam={r['lam']}"
                    f" sharpe={r['test_sharpe']:.3f}"
                    f" cumret={r['test_total_return']:.3f}"
                    f" meanret={r['test_mean_return']:.3f}"
                    f" stdret={r['test_std_return']:.3f}"
                    f" maxdd={r['test_max_drawdown']:.3f}"
                )
                print(line)
                log_f.write(line + "\n")
                log_f.flush()
                np.savetxt("weights.txt", r["test_weights"], delimiter=",")

    # aggregate across seeds: mean sharpe per (w, lam)
    agg = {}
    for r in results:
        key = (r["window_size"], r["lam"])
        agg.setdefault(key, []).append(r["test_sharpe"])

    summary = []
    for (w, lam), vals in agg.items():
        summary.append({
            "window_size": w,
            "lam": lam,
            "mean_sharpe": float(np.mean(vals)),
            "std_sharpe": float(np.std(vals)),
            "n": len(vals),
        })

    summary.sort(key=lambda d: d["mean_sharpe"], reverse=True)

    with open("backtest_grid.txt", "w", encoding="utf-8") as f:
        for row in summary:
            f.write(
                f"w={row['window_size']} lam={row['lam']} "
                f"mean_sharpe={row['mean_sharpe']:.6f} "
                f"std_sharpe={row['std_sharpe']:.6f} "
                f"n={row['n']}\n"
            )


    return results, summary

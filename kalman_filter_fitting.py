from pathlib import Path

import numpy as np
import torch
from scipy.stats import chi2
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


from portfolio_rl.data import YahooConfig, YahooReturnsDataset
from portfolio_rl.kalman import LearnableKalman

def evaluate_qr(observations, q_vals, r_vals, dim):
    results = []
    count = 0

    for q in q_vals:
        q = float(q)
        for r in r_vals:
            r = float(r)
            #if r > q or q/r > 100:
              #continue
            
            if count % 100 == 0:
                print(count)
            count += 1

            kf = LearnableKalman(
                dim=dim,
                Q=q * torch.eye(dim),
                R=r * torch.eye(dim),
            )

            outputs, diagnostics = kf.filter(observations=observations)

            stats = chi_square_test(diagnostics["nis"], dim=dim, alpha=0.95)

            results.append({
                "q": q,
                "r": r,
                "mean_whitened_innovation": diagnostics["whitened_innovation"].mean().item(),
                "std_whitened_innovation": diagnostics["whitened_innovation"].std(unbiased=False).item(),
                **stats
            })

    return results


def chi_square_test(nis, dim, alpha=0.95):
    """
    nis: array of NIS values (T,)
    dim: measurement dimension (e.g. 5)
    alpha: confidence level
    """
    nis = torch.as_tensor(nis)

    lower = float(chi2.ppf((1 - alpha) / 2, dim))
    upper = float(chi2.ppf(1 - (1 - alpha) / 2, dim))

    frac_inside = ((nis >= lower) & (nis <= upper)).float().mean().item()
    frac_above  = (nis > upper).float().mean().item()
    frac_below  = (nis < lower).float().mean().item()

    summary = {
        "mean_nis": nis.mean().item(),
        "expected_mean": dim,
        "lower": lower,
        "upper": upper,
        "frac_inside": frac_inside,
        "frac_above": frac_above,
        "frac_below": frac_below,
    }
    return summary


def _build_metric_grid(results, q_vals, r_vals, key, transform=None):
    q_list = sorted({float(q) for q in q_vals})
    r_list = sorted({float(r) for r in r_vals})
    grid = np.full((len(q_list), len(r_list)), np.nan, dtype=float)
    q_index = {q: i for i, q in enumerate(q_list)}
    r_index = {r: j for j, r in enumerate(r_list)}
    for row in results:
        value = float(row[key])
        if transform is not None:
            value = float(transform(row, value))
        grid[q_index[float(row["q"])]][r_index[float(row["r"])]] = value
    return q_list, r_list, grid


def save_plots(results, q_vals, r_vals, out_dir="plots/kalman_fitting"):
    try:
        import matplotlib.pyplot as plt
    except Exception as exc:
        print(f"[kalman_filter_fitting] matplotlib unavailable, skipping plots: {exc}")
        return

    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    best = min(results, key=lambda row: float(row["score"])) if results else None
    best_q = float(best["q"]) if best is not None else None
    best_r = float(best["r"]) if best is not None else None

    def _compute_edges(vals):
        vals = np.asarray(vals, dtype=float)
        if len(vals) < 2:
            step = vals[0] * 0.1 if vals[0] != 0 else 1.0
            return np.array([vals[0] - step, vals[0] + step], dtype=float)
        mids = np.sqrt(vals[:-1] * vals[1:])
        first = vals[0] ** 2 / mids[0]
        last = vals[-1] ** 2 / mids[-1]
        return np.concatenate([[first], mids, [last]])

    def _save_heatmap(q_list, r_list, grid, title, filename):
        fig, ax = plt.subplots(figsize=(7, 6))
        q_edges = _compute_edges(q_list)
        r_edges = _compute_edges(r_list)
        im = ax.pcolormesh(
            r_edges,
            q_edges,
            grid,
            shading="auto",
        )
        ax.set_xlabel("r")
        ax.set_ylabel("q")
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_title(title)
        fig.colorbar(im, ax=ax)
        if best_q is not None and best_r is not None:
            ax.scatter(
                best_r,
                best_q,
                marker="x",
                s=80,
                linewidths=2.0,
                color="white",
            )
        fig.tight_layout()
        fig.savefig(out_path / filename, dpi=150)
        plt.close(fig)

    for key, title, fname, transform in [
        ("score", "Kalman Fit Score (Inverted)", "score_heatmap.png", lambda _row, v: -v),
        ("mean_nis", "Mean NIS Distance to Expected (Inverted)", "mean_nis_heatmap.png", lambda row, v: -abs(v - float(row.get("expected_mean", 0.0)))),
        ("frac_inside", "Fraction Inside Chi^2 Bounds", "frac_inside_heatmap.png", None),
    ]:
        q_list, r_list, grid = _build_metric_grid(results, q_vals, r_vals, key, transform=transform)
        _save_heatmap(q_list, r_list, grid, title, fname)

def make_grid(df, metric):
    q_vals = np.sort(df["q"].unique())
    r_vals = np.sort(df["r"].unique())

    grid = np.full((len(q_vals), len(r_vals)), np.nan)

    q_to_i = {q: i for i, q in enumerate(q_vals)}
    r_to_j = {r: j for j, r in enumerate(r_vals)}

    for _, row in df.iterrows():
        i = q_to_i[row["q"]]
        j = r_to_j[row["r"]]
        grid[i, j] = row[metric]

    return grid, q_vals, r_vals


def plot_heatmap(grid, q_vals, r_vals, title, cmap="viridis"):
    fig, ax = plt.subplots(figsize=(8, 6))

    im = ax.imshow(grid, origin="lower", aspect="auto", cmap=cmap)

    ax.set_xticks(np.arange(len(r_vals)))
    ax.set_yticks(np.arange(len(q_vals)))

    ax.set_xticklabels([f"{r:.0e}" for r in r_vals])
    ax.set_yticklabels([f"{q:.0e}" for q in q_vals])

    ax.set_xlabel("R (measurement noise)")
    ax.set_ylabel("Q (process noise)")
    ax.set_title(title)

    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label(title)

    # annotate values
    #for i in range(grid.shape[0]):
        #for j in range(grid.shape[1]):
            #if np.isfinite(grid[i, j]):
                #ax.text(j, i, f"{grid[i,j]:.2f}",
                        #ha="center", va="center", fontsize=8)

    ax.text(6, 18, "X", fontsize=8, color="r")

    plt.tight_layout()
    plt.show()


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

    q_vals = torch.logspace(-6, -2, 100)   # 1e-6 → 1e-3
    r_vals = torch.logspace(-6, -2, 100)

    print(q_vals)
    print(r_vals)

    dim = 5  # number of assets

    results = evaluate_qr(
        observations=train_view.as_numpy(),
        q_vals=q_vals,
        r_vals=r_vals,
        dim=dim,
    )

    def score(row, d=5):
        return (
            abs(row["mean_nis"] - d)
            + 5 * abs(row["frac_inside"] - 0.95)
            )

    for row in results:
        row["score"] = score(row)


    results_sorted = sorted(results, key=score)

    for r in results_sorted:
        if False:
            print(
                f"q={r['q']:.1e}, r={r['r']:.1e}, "
                f"meanNIS={r['mean_nis']:.2f}, "
                f"inside={r['frac_inside']:.2f}, "
                f"above={r['frac_above']:.2f}, "
                f"below={r['frac_below']:.2f}, "
                f"score={r['score']:.2f}, "
                f"mean whitened innovation={r['mean_whitened_innovation']:.4f}, "
                f"std whitened innovation={r['std_whitened_innovation']:.4f}"
            )

    with open("kalman_fitting_results_sorted.txt", "w", encoding="utf-8") as f:
        for r in results_sorted:
            f.write(
                f"q={r['q']:.1e}, r={r['r']:.1e}, "
                f"meanNIS={r['mean_nis']:.2f}, "
                f"inside={r['frac_inside']:.2f}, "
                f"above={r['frac_above']:.2f}, "
                f"below={r['frac_below']:.2f}, "
                f"score={r['score']:.2f}, "
                f"mean whitened innovation={r['mean_whitened_innovation']:.4f}, "
                f"std whitened innovation={r['std_whitened_innovation']:.4f}\n"
            )

    save_plots(results, q_vals, r_vals)

    df = pd.DataFrame(results)

    # optional but handy
    df["log_q"] = np.log10(df["q"])
    df["log_r"] = np.log10(df["r"])
    df["log_q_over_r"] = np.log10(df["q"] / df["r"])

    df["nis_error"] = np.abs(df["mean_nis"] - 5)
    df["std_error"] = np.abs(df["std_whitened_innovation"] - 1)

    for metric, title, cmap in [
    ("nis_error", "Mean NIS Error (target ≈ d=5)", "viridis_r"),
    ("frac_inside",  "Fraction inside χ² interval", "viridis"),
    ("mean_whitened_innovation",   "Mean whitened innovation (target ≈ 0)", "viridis"),
    ("std_error",    "Std whitened innovation Error (target ≈ 1)", "viridis_r")
    ]:
        grid, q_vals, r_vals = make_grid(df, metric)
        plot_heatmap(grid, q_vals, r_vals, title, cmap)

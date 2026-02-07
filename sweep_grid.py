import csv
import itertools
import os
import statistics
from concurrent.futures import ProcessPoolExecutor, as_completed

from portfolio_rl.backtest_seed import run_one
from portfolio_rl.config import TrainConfig


def _parse_bool(raw):
    val = raw.strip().lower()
    if val in {"true", "1", "t", "yes", "y"}:
        return True
    if val in {"false", "0", "f", "no", "n"}:
        return False
    raise ValueError(f"Unsupported bool value: {raw}")


def _parse_value(raw, sample):
    if raw is None or raw == "":
        raise ValueError("Empty value")
    if isinstance(sample, bool):
        return _parse_bool(raw)
    if isinstance(sample, int):
        return int(float(raw))
    if isinstance(sample, float):
        return float(raw)
    return raw


def _run_job(args):
    combo_id, combo, keys, seed, fixed = args
    cfg_kwargs = dict(zip(keys, combo))

    cfg = TrainConfig(
        device="cpu",
        T=(fixed["window_size"] + 5),
        updates=fixed["updates"],
        window_size=fixed["window_size"],
        print_every=999999,
        use_kf=True,
        episodes_per_batch=1,
        lam=float(cfg_kwargs["lam"]),
        gamma=0.99,
        dyn_enabled=True,
        use_critic=False,
        kappa_unc=float(cfg_kwargs["kappa_unc"]),
        dyn_use_sim=True,
        actor_weight=float(cfg_kwargs["actor_weight"]),
        lr_actor=float(cfg_kwargs["lr_actor"]),
        dyn_sim_M=int(cfg_kwargs["dyn_sim_M"]),
        dyn_sim_deterministic=bool(cfg_kwargs["dyn_sim_deterministic"]),
        dyn_sim_pl_weight=float(cfg_kwargs["dyn_sim_pl_weight"]),
    )

    res = run_one(
        seed=seed,
        window_size=cfg.window_size,
        lam=cfg.lam,
        cfg=cfg,
        verbose=False,
        save_best_path=None,
        evaluate_best_on_test=False,
        eval_on_validation=True,
        stats_csv_path=None,
        networksize=fixed["networksize"],
        learnrate=cfg.lr_actor,
        print_results=True,
    )

    return {
        "combo_id": combo_id,
        "combo": combo,
        "seed": seed,
        "best_val_sharpe": res.get("best_val_sharpe", float("nan")),
        "test_sharpe": res.get("test_sharpe", float("nan")),
        "test_total_return": res.get("test_total_return", float("nan")),
        "test_mean_return": res.get("test_mean_return", float("nan")),
        "test_std_return": res.get("test_std_return", float("nan")),
    }


def main():
    # --- fixed settings (edit as needed) ---
    window_size = 1
    updates = 1000
    networksize = 128
    seeds = [0, 1, 2]
    max_workers = min(8, os.cpu_count() or 1)

    # --- hyperparameter grid ---
    grid = {
        "lr_actor": [1e-3],
        "dyn_sim_M": [10],
        "dyn_sim_deterministic": [True],
        "dyn_sim_pl_weight": [0.1],
        "lam": [1.0],
        "actor_weight": [1.0],
        "kappa_unc": [1.0],
    }

    keys = list(grid.keys())
    combos = list(itertools.product(*[grid[k] for k in keys]))

    out_path = "sweep_results.csv"
    combo_seed_vals = {i: [] for i in range(len(combos))}
    existing_settings = set()
    combo_to_id = {combo: i for i, combo in enumerate(combos)}

    if os.path.exists(out_path) and os.path.getsize(out_path) > 0:
        with open(out_path, "r", newline="") as f:
            reader = csv.DictReader(f)
            if reader.fieldnames:
                needed = set(keys + ["seed"])
                if needed.issubset(reader.fieldnames):
                    samples = {k: grid[k][0] for k in keys}
                    for row in reader:
                        try:
                            combo = tuple(_parse_value(row[k], samples[k]) for k in keys)
                            combo_id = combo_to_id.get(combo)
                            if combo_id is None:
                                continue
                            seed = int(float(row["seed"]))
                        except (KeyError, ValueError):
                            continue

                        setting = (combo, seed)
                        if setting in existing_settings:
                            continue
                        existing_settings.add(setting)

                        if "best_val_sharpe" in row and row["best_val_sharpe"] not in (None, ""):
                            try:
                                combo_seed_vals[combo_id].append(float(row["best_val_sharpe"]))
                            except ValueError:
                                pass

    write_header = not (os.path.exists(out_path) and os.path.getsize(out_path) > 0)
    file_mode = "w" if write_header else "a"
    with open(out_path, file_mode, newline="") as f:
        writer = csv.writer(f)
        if write_header:
            writer.writerow(
                [
                    "combo_id",
                    *keys,
                    "seed",
                    "best_val_sharpe",
                    "test_sharpe",
                    "test_total_return",
                    "test_mean_return",
                    "test_std_return",
                ]
            )

        combo_summaries = []
        jobs = []
        fixed = {"window_size": window_size, "updates": updates, "networksize": networksize}

        for combo_id, combo in enumerate(combos):
            for seed in seeds:
                if (combo, seed) in existing_settings:
                    continue
                jobs.append((combo_id, combo, keys, seed, fixed))

        if jobs:
            with ProcessPoolExecutor(max_workers=max_workers) as ex:
                futures = [ex.submit(_run_job, j) for j in jobs]
                for fut in as_completed(futures):
                    row = fut.result()
                    combo_id = row["combo_id"]
                    combo = row["combo"]
                    seed = row["seed"]

                    writer.writerow(
                        [
                            combo_id,
                            *combo,
                            seed,
                            row["best_val_sharpe"],
                            row["test_sharpe"],
                            row["test_total_return"],
                            row["test_mean_return"],
                            row["test_std_return"],
                        ]
                    )

                    combo_seed_vals[combo_id].append(row["best_val_sharpe"])

        for combo_id, combo in enumerate(combos):
            seed_rows = combo_seed_vals.get(combo_id, [])
            clean = [v for v in seed_rows if isinstance(v, (int, float))]
            if clean:
                combo_summaries.append(
                    {
                        "combo_id": combo_id,
                        "params": dict(zip(keys, combo)),
                        "mean_best_val_sharpe": statistics.mean(clean),
                        "std_best_val_sharpe": statistics.pstdev(clean) if len(clean) > 1 else 0.0,
                    }
                )

    # Print top-5 configs by mean best val Sharpe
    combo_summaries.sort(key=lambda x: x["mean_best_val_sharpe"], reverse=True)
    print("Top 5 configs by mean best val Sharpe:")
    for row in combo_summaries[:5]:
        print(row)


if __name__ == "__main__":
    main()

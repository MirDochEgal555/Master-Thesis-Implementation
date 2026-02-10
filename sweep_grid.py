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
    window_size = int(cfg_kwargs["window_size"])
    save_best_path = None
    policy_dir = fixed.get("policy_dir")
    if policy_dir:
        save_best_path = os.path.join(policy_dir, f"policy_combo{combo_id}_seed{seed}.pt")

    kf_mode = cfg_kwargs.get("kf_mode", "learned")
    use_kf = (kf_mode != "none")
    kf_fixed = (kf_mode == "fixed")
    dyn_use_sim = bool(cfg_kwargs.get("dyn_use_sim", True))
    kf_q = float(cfg_kwargs.get("kf_q", 1e-3))
    kf_r = float(cfg_kwargs.get("kf_r", 1e-3))
    evaluate_best_on_test = bool(cfg_kwargs.get("evaluate_best_on_test", False))


    cfg = TrainConfig(
        device="cpu",
        T=(window_size + 5),
        updates=fixed["updates"],
        window_size=window_size,
        print_every=5,
        use_kf=use_kf,
        episodes_per_batch=1,
        lam=float(cfg_kwargs["lam"]),
        gamma=0.99,
        dyn_enabled=True,
        use_critic=False,
        kappa_unc=float(cfg_kwargs["kappa_unc"]),
        dyn_use_sim=dyn_use_sim,
        actor_weight=float(cfg_kwargs["actor_weight"]),
        lr_actor=float(cfg_kwargs["lr_actor"]),
        dyn_sim_M=int(cfg_kwargs["dyn_sim_M"]),
        dyn_sim_deterministic=bool(cfg_kwargs["dyn_sim_deterministic"]),
        dyn_sim_pl_weight=float(cfg_kwargs["dyn_sim_pl_weight"]),
        kf_weight=0.0 if kf_fixed else 1.0,
    )

    res = run_one(
        seed=seed,
        window_size=cfg.window_size,
        lam=cfg.lam,
        cfg=cfg,
        verbose=False,
        save_best_path=save_best_path,
        eval_on_validation=False,
        stats_csv_path=None,
        networksize=fixed["networksize"],
        learnrate=cfg.lr_actor,
        print_results=True,
        kf_fixed=kf_fixed,
        kf_q=kf_q,
        kf_r=kf_r,
        evaluate_best_on_test=evaluate_best_on_test,
    )

    return {
        "combo_id": combo_id,
        "combo": combo,
        "seed": seed,
        "best_val_sharpe": res.get("best_val_sharpe", float("nan")),
        "best_val_sharpe_epoch": res.get("best_val_sharpe_epoch"),
        "best_policy_path": res.get("best_policy_path"),
        "used_best_policy": res.get("used_best_policy"),
        "used_policy_epoch": res.get("used_policy_epoch"),
        "used_policy_path": res.get("used_policy_path"),
        "test_sharpe": res.get("test_sharpe", float("nan")),
        "test_total_return": res.get("test_total_return", float("nan")),
        "test_mean_return": res.get("test_mean_return", float("nan")),
        "test_std_return": res.get("test_std_return", float("nan")),
        "min_return": res.get("test_max_drawdown", float("nan")),
    }


def main():
    # --- fixed settings (edit as needed) ---
    #window_size = 1
    updates = 2000
    networksize = 128
    seeds = list(range(20))
    max_workers = min(8, os.cpu_count() or 1)
    policy_dir = "grid_search_results/policies"

    # --- hyperparameter grid ---
    grid = {
        "window_size": [10],
        "lr_actor": [1e-3],
        "dyn_sim_M": [10],
        "dyn_sim_deterministic": [True],
        "dyn_sim_pl_weight": [0.1],
        "lam": [1.0],
        "actor_weight": [1.0],
        "kappa_unc": [0.1],
        # new ablations
        "kf_mode": ["fixed","learned"],
        "dyn_use_sim": [True],
        "evaluate_best_on_test": [False,True],
        # fixed KF params (only used when kf_mode == "fixed")
        "kf_q": [0.00022],
        "kf_r": [0.00015],
    }

    keys = list(grid.keys())
    combos = list(itertools.product(*[grid[k] for k in keys]))

    out_path = "sweep_results.csv"
    combo_seed_vals = {i: [] for i in range(len(combos))}
    existing_settings = set()
    combo_to_id = {combo: i for i, combo in enumerate(combos)}
    existing_rows = []
    existing_fieldnames = []
    needs_header_update = False

    if os.path.exists(out_path) and os.path.getsize(out_path) > 0:
        with open(out_path, "r", newline="") as f:
            reader = csv.DictReader(f)
            if reader.fieldnames:
                existing_fieldnames = reader.fieldnames
                needed = set(keys + ["seed"])
                if (
                    "min_return" not in reader.fieldnames
                    or "used_policy_epoch" not in reader.fieldnames
                    or "used_policy_path" not in reader.fieldnames
                    or not needed.issubset(reader.fieldnames)
                ):
                    needs_header_update = True
                if needed.issubset(reader.fieldnames):
                    samples = {k: grid[k][0] for k in keys}
                    for row in reader:
                        existing_rows.append(row)
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
                else:
                    for row in reader:
                        existing_rows.append(row)

    file_exists = os.path.exists(out_path) and os.path.getsize(out_path) > 0
    write_header = (not file_exists) or needs_header_update
    file_mode = "w" if write_header else "a"
    with open(out_path, file_mode, newline="") as f:
        writer = csv.writer(f)
        header = [
            "combo_id",
            *keys,
            "seed",
            "best_val_sharpe",
            "best_val_sharpe_epoch",
            "best_policy_path",
            "used_best_policy",
            "used_policy_epoch",
            "used_policy_path",
            "test_sharpe",
            "test_total_return",
            "test_mean_return",
            "test_std_return",
            "min_return",
        ]
        if write_header:
            writer.writerow(header)
            if needs_header_update and existing_rows:
                for row in existing_rows:
                    writer.writerow([row.get(col, "") for col in header])

        combo_summaries = []
        jobs = []
        fixed = {"updates": updates, "networksize": networksize, "policy_dir": policy_dir}

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
                            row["best_val_sharpe_epoch"],
                            row["best_policy_path"],
                            row["used_best_policy"],
                            row["used_policy_epoch"],
                            row["used_policy_path"],
                            row["test_sharpe"],
                            row["test_total_return"],
                            row["test_mean_return"],
                            row["test_std_return"],
                            row["min_return"],
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

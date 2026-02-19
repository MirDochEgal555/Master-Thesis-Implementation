import csv
import itertools
import os
import pickle
import statistics
from concurrent.futures import ProcessPoolExecutor, as_completed

import torch

from portfolio_rl.backtest_seed import _load_data_bundle_cached, run_one
from portfolio_rl.config import TrainConfig
from portfolio_rl.kalman import LearnableKalman
from portfolio_rl.models import PolicyNet
from portfolio_rl.trainer import Trainer


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


def _load_checkpoint(path, device):
    try:
        return torch.load(path, map_location=device)
    except pickle.UnpicklingError:
        from torch.serialization import safe_globals

        with safe_globals([TrainConfig]):
            return torch.load(path, map_location=device, weights_only=False)


def _evaluate_checkpoint_on_test(
    ckpt_path,
    cfg,
    *,
    networksize,
    kf_fixed,
    kf_q,
    kf_r,
    cache_path="returns.parquet",
):
    torch.set_num_threads(1)
    os.environ["OMP_NUM_THREADS"] = "1"
    os.environ["MKL_NUM_THREADS"] = "1"

    device = torch.device(cfg.device)
    train_view, _, test_view, _ = _load_data_bundle_cached(cache_path=cache_path)
    K = train_view.K

    policy = PolicyNet(K=K, hidden=networksize).to(device)
    if cfg.use_kf:
        if kf_fixed:
            kf = LearnableKalman(
                dim=K,
                Q=kf_q * torch.eye(K),
                R=kf_r * torch.eye(K),
            ).to(device)
        else:
            kf = LearnableKalman(dim=K).to(device)
    else:
        kf = None

    ckpt = _load_checkpoint(ckpt_path, device)
    policy.load_state_dict(ckpt["policy_state_dict"])
    if kf is not None and ckpt.get("kf_state_dict") is not None:
        kf.load_state_dict(ckpt["kf_state_dict"])

    metrics = Trainer.evaluate_full_run(
        policy,
        kf,
        test_view,
        window_size=cfg.window_size,
        device=cfg.device,
        sample_policy=False,
        return_weights=False,
    )
    used_policy_epoch = int(ckpt.get("epoch", cfg.updates - 1 if cfg.updates > 0 else 0))
    return metrics, used_policy_epoch


def _row_to_list(row, header, defaults=None):
    defaults = defaults or {}
    values = []
    for col in header:
        val = row.get(col, None)
        if (val is None or val == "") and col in defaults:
            val = defaults[col]
        if val is None:
            val = ""
        values.append(val)
    return values


def _build_output_row(combo_id, combo, keys, seed, eval_best, row_specific):
    row = {
        "combo_id": combo_id,
        **dict(zip(keys, combo)),
        "evaluate_best_on_test": eval_best,
        "seed": seed,
        "best_val_sharpe": row_specific.get("best_val_sharpe", float("nan")),
        "best_val_sharpe_epoch": row_specific.get("best_val_sharpe_epoch"),
        "best_policy_path": row_specific.get("best_policy_path"),
        "used_best_policy": row_specific.get("used_best_policy"),
        "used_policy_epoch": row_specific.get("used_policy_epoch"),
        "used_policy_path": row_specific.get("used_policy_path"),
        "test_sharpe": row_specific.get("test_sharpe", float("nan")),
        "test_total_return": row_specific.get("test_total_return", float("nan")),
        "test_mean_return": row_specific.get("test_mean_return", float("nan")),
        "test_std_return": row_specific.get("test_std_return", float("nan")),
        "min_return": row_specific.get("min_return", float("nan")),
    }
    return row


def _load_existing_rows(path, keys, grid, combo_to_id, combo_seed_vals=None):
    existing_settings = set()
    existing_rows = []
    needs_header_update = False

    if os.path.exists(path) and os.path.getsize(path) > 0:
        with open(path, "r", newline="") as f:
            reader = csv.DictReader(f)
            if reader.fieldnames:
                needed = set(keys + ["seed"])
                if (
                    "min_return" not in reader.fieldnames
                    or "used_policy_epoch" not in reader.fieldnames
                    or "used_policy_path" not in reader.fieldnames
                    or "evaluate_best_on_test" not in reader.fieldnames
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

                        if (
                            combo_seed_vals is not None
                            and "best_val_sharpe" in row
                            and row["best_val_sharpe"] not in (None, "")
                        ):
                            try:
                                combo_seed_vals[combo_id].append(float(row["best_val_sharpe"]))
                            except ValueError:
                                pass
                else:
                    for row in reader:
                        existing_rows.append(row)

    return existing_settings, existing_rows, needs_header_update


def _run_job(args):
    combo_id, combo, keys, seed, fixed = args
    cfg_kwargs = dict(zip(keys, combo))
    window_size = int(cfg_kwargs["window_size"])
    save_best_path = None
    best_policy_dir = fixed.get("best_policy_dir")
    if best_policy_dir:
        save_best_path = os.path.join(best_policy_dir, f"policy_combo{combo_id}_seed{seed}.pt")

    losses_dir = fixed.get("losses_dir")
    losses_csv_path = None
    if losses_dir:
        losses_csv_path = os.path.join(losses_dir, f"losses_combo{combo_id}_seed{seed}.csv")

    final_policy_dir = fixed.get("final_policy_dir")
    final_policy_path = None
    if final_policy_dir:
        final_policy_path = os.path.join(final_policy_dir, f"final_policy_combo{combo_id}_seed{seed}.pt")

    kf_mode = cfg_kwargs.get("kf_mode", "learned")
    use_kf = (kf_mode != "none")
    kf_fixed = (kf_mode == "fixed")
    dyn_use_sim = bool(cfg_kwargs.get("dyn_use_sim", True))
    kf_q = float(cfg_kwargs.get("kf_q", 1e-3))
    kf_r = float(cfg_kwargs.get("kf_r", 1e-3))


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
        stats_csv_path=losses_csv_path,
        stats_every=cfg.print_every,
        write_stats_on_eval=True,
        save_final_path=final_policy_path,
        networksize=fixed["networksize"],
        learnrate=cfg.lr_actor,
        print_results=True,
        kf_fixed=kf_fixed,
        kf_q=kf_q,
        kf_r=kf_r,
        evaluate_best_on_test=False,
        warmup_kf_epochs=fixed["warmup_kf_epochs"],
        warmup_dyn_epochs=fixed["warmup_dyn_epochs"],
        return_weights=False,
    )

    base_row = {
        "combo_id": combo_id,
        "combo": combo,
        "seed": seed,
        "best_val_sharpe": res.get("best_val_sharpe", float("nan")),
        "best_val_sharpe_epoch": res.get("best_val_sharpe_epoch"),
        "best_policy_path": res.get("best_policy_path"),
    }

    final_row = {
        **base_row,
        "used_best_policy": False,
        "used_policy_epoch": res.get("used_policy_epoch"),
        "used_policy_path": final_policy_path if final_policy_path else res.get("used_policy_path"),
        "test_sharpe": res.get("test_sharpe", float("nan")),
        "test_total_return": res.get("test_total_return", float("nan")),
        "test_mean_return": res.get("test_mean_return", float("nan")),
        "test_std_return": res.get("test_std_return", float("nan")),
        "min_return": res.get("test_max_drawdown", float("nan")),
    }

    best_metrics = None
    best_used_epoch = None
    if save_best_path and os.path.exists(save_best_path):
        try:
            best_metrics, best_used_epoch = _evaluate_checkpoint_on_test(
                save_best_path,
                cfg,
                networksize=fixed["networksize"],
                kf_fixed=kf_fixed,
                kf_q=kf_q,
                kf_r=kf_r,
                cache_path=fixed.get("cache_path", "returns.parquet"),
            )
        except Exception as exc:
            print(f"Best policy eval failed for combo {combo_id}, seed {seed}: {exc}")

    best_row = {
        **base_row,
        "used_best_policy": bool(best_metrics),
        "used_policy_epoch": best_used_epoch,
        "used_policy_path": save_best_path,
        "test_sharpe": float(best_metrics["sharpe"]) if best_metrics else float("nan"),
        "test_total_return": float(best_metrics["total_reward"]) if best_metrics else float("nan"),
        "test_mean_return": float(best_metrics["mean_return"]) if best_metrics else float("nan"),
        "test_std_return": float(best_metrics["std_return"]) if best_metrics else float("nan"),
        "min_return": float(best_metrics["max_drawdown"]) if best_metrics else float("nan"),
    }

    return {
        "combo_id": combo_id,
        "combo": combo,
        "seed": seed,
        "best_val_sharpe": base_row["best_val_sharpe"],
        "final_row": final_row,
        "best_row": best_row,
    }


def main():
    # --- fixed settings (edit as needed) ---
    #window_size = 1
    updates = 3000
    warmup_kf_epochs = 100
    warmup_dyn_epochs = 100
    networksize = 128
    seeds = list(range(50))
    max_workers = min(8, os.cpu_count() or 1)
    save_best_policies = True  # required to evaluate best-on-test without retraining
    best_policy_dir = "grid_search_results/policies" if save_best_policies else None
    losses_dir = "grid_search_results/losses"
    final_policy_dir = "grid_search_results/final_policies"
    flush_every = 10

    # --- hyperparameter grid ---
    grid = {
        "window_size": [10],
        "lr_actor": [1e-3],
        "dyn_sim_M": [20],
        "dyn_sim_deterministic": [True],
        "dyn_sim_pl_weight": [0.1],
        "lam": [1.0],
        "actor_weight": [1.0],
        "kappa_unc": [1.0],
        # new ablations
        "kf_mode": ["learned", "fixed"],
        "dyn_use_sim": [True],
        # fixed KF params (only used when kf_mode == "fixed")
        "kf_q": [0.00022],
        "kf_r": [0.00015],
    }

    keys = list(grid.keys())
    combos = list(itertools.product(*[grid[k] for k in keys]))

    output_specs = {
        "final": {"path": "total_results_final.csv", "evaluate_best_on_test": False},
        "best": {"path": "total_results_best.csv", "evaluate_best_on_test": True},
    }
    summary_label = "final"

    combo_seed_vals = {i: [] for i in range(len(combos))}
    combo_to_id = {combo: i for i, combo in enumerate(combos)}
    header = [
        "combo_id",
        *keys,
        "evaluate_best_on_test",
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

    output_states = {}
    for label, spec in output_specs.items():
        existing_settings, existing_rows, needs_header_update = _load_existing_rows(
            spec["path"],
            keys,
            grid,
            combo_to_id,
            combo_seed_vals if label == summary_label else None,
        )
        file_exists = os.path.exists(spec["path"]) and os.path.getsize(spec["path"]) > 0
        write_header = (not file_exists) or needs_header_update
        file_mode = "w" if write_header else "a"
        f = open(spec["path"], file_mode, newline="")
        writer = csv.writer(f)
        if write_header:
            writer.writerow(header)
            if needs_header_update and existing_rows:
                defaults = {"evaluate_best_on_test": spec["evaluate_best_on_test"]}
                for row in existing_rows:
                    writer.writerow(_row_to_list(row, header, defaults))

        output_states[label] = {
            "file": f,
            "writer": writer,
            "existing_settings": existing_settings,
            "rows_since_flush": 0,
        }

    combo_summaries = []
    jobs = []
    fixed = {
        "updates": updates,
        "warmup_kf_epochs": warmup_kf_epochs,
        "warmup_dyn_epochs": warmup_dyn_epochs,
        "networksize": networksize,
        "best_policy_dir": best_policy_dir,
        "losses_dir": losses_dir,
        "final_policy_dir": final_policy_dir,
    }

    for combo_id, combo in enumerate(combos):
        for seed in seeds:
            if all(
                (combo, seed) in output_states[label]["existing_settings"]
                for label in output_specs
            ):
                continue
            jobs.append((combo_id, combo, keys, seed, fixed))

    if jobs:
        with ProcessPoolExecutor(max_workers=max_workers) as ex:
            futures = [ex.submit(_run_job, j) for j in jobs]
            for fut in as_completed(futures):
                try:
                    row = fut.result()
                except Exception as exc:
                    print(f"Job failed: {exc}")
                    continue
                combo_id = row["combo_id"]
                combo = row["combo"]
                seed = row["seed"]

                summary_state = output_states[summary_label]
                if (combo, seed) not in summary_state["existing_settings"]:
                    combo_seed_vals[combo_id].append(row["best_val_sharpe"])

                for label, spec in output_specs.items():
                    state = output_states[label]
                    if (combo, seed) in state["existing_settings"]:
                        continue
                    row_specific = row["final_row"] if label == "final" else row["best_row"]
                    out_row = _build_output_row(
                        combo_id,
                        combo,
                        keys,
                        seed,
                        spec["evaluate_best_on_test"],
                        row_specific,
                    )
                    state["writer"].writerow(_row_to_list(out_row, header))
                    state["rows_since_flush"] += 1
                    if state["rows_since_flush"] >= flush_every:
                        state["file"].flush()
                        state["rows_since_flush"] = 0
                    state["existing_settings"].add((combo, seed))

        for state in output_states.values():
            if state["rows_since_flush"] > 0:
                state["file"].flush()

    for state in output_states.values():
        state["file"].close()

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

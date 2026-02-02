# run_experiment.py (or inside main.py)
import os
import pickle
import torch
import numpy as np
import pandas as pd

from .config import TrainConfig
from .models import PolicyNet, ValueNet
from .kalman import LearnableKalman
from .trainer import Trainer
from .data import YahooConfig, YahooReturnsDataset
from .utils import set_seed


def run_one(
    seed: int,
    window_size: int,
    lam: float,
    *,
    cache_path="returns.parquet",
    cfg=None,
    verbose=False,
    save_best_path=None,
    evaluate_best_on_test=False,
    stats_csv_path=None,
    networksize: int, 
    learnrate: float,
    data_bundle=None,
):
    # one process = one thread (important for parallel speed)
    torch.set_num_threads(1)
    os.environ["OMP_NUM_THREADS"] = "1"
    os.environ["MKL_NUM_THREADS"] = "1"

    set_seed(seed)

    if data_bundle is None:
        tickers = ['JPM', 'JNJ', 'MSFT', 'PG', 'XOM']
        ycfg = YahooConfig(
            tickers=tickers,
            start_date="2022-01-01",
            end_date="2024-12-31",
            price_field="Close",
            cache_path=cache_path,
        )

        dataset = YahooReturnsDataset(ycfg)
        train_view, val_view, test_view = dataset.split_by_date(
            train_end="2023-03-24",
            val_end="2023-09-30",
        )

        train_covs = train_view.precompute_expanding_cov(diag=True)
    else:
        train_view, val_view, test_view, train_covs = data_bundle
        try:
            tickers = list(train_view.parent.cfg.tickers)
        except Exception:
            tickers = None

    def _avg_weight_cols(weights, tickers):
        if not tickers:
            return {}
        if not weights:
            return {f"avg_weight_{t}": np.nan for t in tickers}
        try:
            weights_tensor = torch.stack(
                [w if torch.is_tensor(w) else torch.as_tensor(w) for w in weights],
                dim=0,
            )
        except Exception:
            return {f"avg_weight_{t}": np.nan for t in tickers}
        avg_weights = weights_tensor.mean(dim=0).cpu().numpy()
        return {f"avg_weight_{t}": float(w) for t, w in zip(tickers, avg_weights)}


    # ---- use the SAME trainer instance across warmup+train ----
    if cfg is None:
        cfg = TrainConfig(
            device="cpu",
            T=(window_size+5),
            updates=1500,          # keep smaller for sweeps; increase later
            window_size=window_size,
            print_every=99999999,    # silence inside worker
            use_kf=False,
            episodes_per_batch=1,
            lam=float(lam),
            gamma=0.99,
            dyn_enabled=False,
            use_critic=False,
            kappa_unc=0.0,
            dyn_sim_deterministic=False,
            lr_actor = learnrate,
        )

    device = torch.device(cfg.device)
    K = train_view.K

    policy = PolicyNet(K=K,hidden=networksize).to(device)
    value = ValueNet(K).to(device)
    kf = LearnableKalman(dim=K).to(device) if cfg.use_kf else None
    trainer = Trainer(policy, value, kf, cfg)
    stats_rows = []

    if verbose:
        print("-----------------Training KF-----------------")
    for epoch in range(100):
        break
        state = None
        tl, pl, vl, kfl, dynl, simpl, simvl, dr = [], [], [], [], [], [], [], []
        for obs, cov in train_view.iter_windows_with_cov(T=cfg.T, stride=cfg.T, covs=train_covs):
            out, state = trainer.train_step(obs, update=epoch, state=state, warmupkf=True, covariances=cov)

            tl.append(out['total_loss'])
            pl.append(out['policy_loss'])
            vl.append(out['value_loss'])
            kfl.append(out['kf_loss'])
            dynl.append(out['dyn_loss'])
            simpl.append(out['sim_policy_loss'])
            simvl.append(out['sim_value_loss'])
            dr.append(out['return0'])

        if epoch % cfg.print_every == 0 and epoch != 0:
            print(
                f"Update:{epoch}"
                f"Total loss:{np.mean(tl):.5f}"
                f" Policy loss:{np.mean(pl):.5f}"
                f" Value loss:{np.mean(vl):.5f}"
                f" KF loss:{np.mean(kfl):.5f}"
                f" Dynamics loss:{np.mean(dynl):.5f}"
                f" Policy loss Sim:{np.mean(simpl):.5f}"
                f" Value loss Sim:{np.mean(simvl):.5f}"
                f" Discounted Reward:{np.mean(dr):.3f}"
            )
        stats_rows.append(
            {
                "phase": "warmup_kf",
                "epoch": epoch,
                "seed": seed,
                "window_size": window_size,
                "lam": float(lam),
                "total_loss": float(np.mean(tl)) if tl else np.nan,
                "policy_loss": float(np.mean(pl)) if pl else np.nan,
                "value_loss": float(np.mean(vl)) if vl else np.nan,
                "kf_loss": float(np.mean(kfl)) if kfl else np.nan,
                "dyn_loss": float(np.mean(dynl)) if dynl else np.nan,
                "sim_policy_loss": float(np.mean(simpl)) if simpl else np.nan,
                "sim_value_loss": float(np.mean(simvl)) if simvl else np.nan,
                "discounted_reward": float(np.mean(dr)) if dr else np.nan,
                "val_sharpe": np.nan,
                **_avg_weight_cols(None, tickers),
            }
        )
    

    if verbose:
        print("-----------------Training Dynamics-----------------")

    # warmup (optional)
    for epoch in range(100):
        break
        state = None
        tl, pl, vl, kfl, dynl, simpl, simvl, dr = [], [], [], [], [], [], [], []
        for obs, cov in train_view.iter_windows_with_cov(T=cfg.T, stride=cfg.T, covs=train_covs):
            out, state = trainer.train_step(obs, update=epoch, state=state, warmupdyn=True, covariances=cov)

            tl.append(out['total_loss'])
            pl.append(out['policy_loss'])
            vl.append(out['value_loss'])
            kfl.append(out['kf_loss'])
            dynl.append(out['dyn_loss'])
            simpl.append(out['sim_policy_loss'])
            simvl.append(out['sim_value_loss'])
            dr.append(out['return0'])

        if epoch % cfg.print_every == 0 and epoch != 0:
            print(
                f"Update:{epoch}"
                f"Total loss:{np.mean(tl):.5f}"
                f" Policy loss:{np.mean(pl):.5f}"
                f" Value loss:{np.mean(vl):.5f}"
                f" KF loss:{np.mean(kfl):.5f}"
                f" Dynamics loss:{np.mean(dynl):.5f}"
                f" Policy loss Sim:{np.mean(simpl):.5f}"
                f" Value loss Sim:{np.mean(simvl):.5f}"
                f" Discounted Reward:{np.mean(dr):.3f}"
            )
        stats_rows.append(
            {
                "phase": "warmup_dyn",
                "epoch": epoch,
                "seed": seed,
                "window_size": window_size,
                "lam": float(lam),
                "total_loss": float(np.mean(tl)) if tl else np.nan,
                "policy_loss": float(np.mean(pl)) if pl else np.nan,
                "value_loss": float(np.mean(vl)) if vl else np.nan,
                "kf_loss": float(np.mean(kfl)) if kfl else np.nan,
                "dyn_loss": float(np.mean(dynl)) if dynl else np.nan,
                "sim_policy_loss": float(np.mean(simpl)) if simpl else np.nan,
                "sim_value_loss": float(np.mean(simvl)) if simvl else np.nan,
                "discounted_reward": float(np.mean(dr)) if dr else np.nan,
                "val_sharpe": np.nan,
                **_avg_weight_cols(None, tickers),
            }
        )

    if verbose:
        print("-----------------Warmup Critic-----------------")

    for epoch in range(100):
        break
        state = None
        tl, pl, vl, kfl, dynl, simpl, simvl, dr = [], [], [], [], [], [], [], []
        for obs, cov in train_view.iter_windows_with_cov(T=cfg.T, stride=cfg.T, covs=train_covs):
            out, state = trainer.train_step(obs, update=epoch, state=state, warmupcrit=True, covariances=cov)

            tl.append(out['total_loss'])
            pl.append(out['policy_loss'])
            vl.append(out['value_loss'])
            kfl.append(out['kf_loss'])
            dynl.append(out['dyn_loss'])
            simpl.append(out['sim_policy_loss'])
            simvl.append(out['sim_value_loss'])
            dr.append(out['return0'])

        if epoch % cfg.print_every == 0 and epoch != 0:
            print(
                f"Update:{epoch}"
                f"Total loss:{np.mean(tl):.5f}"
                f" Policy loss:{np.mean(pl):.5f}"
                f" Value loss:{np.mean(vl):.5f}"
                f" KF loss:{np.mean(kfl):.5f}"
                f" Dynamics loss:{np.mean(dynl):.5f}"
                f" Policy loss Sim:{np.mean(simpl):.5f}"
                f" Value loss Sim:{np.mean(simvl):.5f}"
                f" Discounted Reward:{np.mean(dr):.3f}"
            )
        stats_rows.append(
            {
                "phase": "warmup_crit",
                "epoch": epoch,
                "seed": seed,
                "window_size": window_size,
                "lam": float(lam),
                "total_loss": float(np.mean(tl)) if tl else np.nan,
                "policy_loss": float(np.mean(pl)) if pl else np.nan,
                "value_loss": float(np.mean(vl)) if vl else np.nan,
                "kf_loss": float(np.mean(kfl)) if kfl else np.nan,
                "dyn_loss": float(np.mean(dynl)) if dynl else np.nan,
                "sim_policy_loss": float(np.mean(simpl)) if simpl else np.nan,
                "sim_value_loss": float(np.mean(simvl)) if simvl else np.nan,
                "discounted_reward": float(np.mean(dr)) if dr else np.nan,
                "val_sharpe": np.nan,
                **_avg_weight_cols(None, tickers),
            }
        )

    if verbose:
        print("-----------------Warmup Critic Ended - Training Policy-----------------")

    best_val_sharpe = -np.inf
    best_val_sharpe_epoch = 0

    # train
    for epoch in range(cfg.updates):
        state = None
        tl, pl, vl, kfl, dynl, simpl, simvl, dr = [], [], [], [], [], [], [], []
        for obs, cov in train_view.iter_windows_with_cov(T=cfg.T, stride=cfg.T, covs=train_covs):
            out, state = trainer.train_step(obs, update=epoch, state=state, covariances=cov)

            tl.append(out['total_loss'])
            pl.append(out['policy_loss'])
            vl.append(out['value_loss'])
            kfl.append(out['kf_loss'])
            dynl.append(out['dyn_loss'])
            simpl.append(out['sim_policy_loss'])
            simvl.append(out['sim_value_loss'])
            dr.append(out['return0'])

        val_sharpe = np.nan
        avg_weight_cols = _avg_weight_cols(None, tickers)
        if epoch % cfg.print_every == 0 and epoch != 0:
            print(
                f"Update:{epoch}"
                f"Total loss:{np.mean(tl):.5f}"
                f" Policy loss:{np.mean(pl):.5f}"
                f" Value loss:{np.mean(vl):.5f}"
                f" KF loss:{np.mean(kfl):.5f}"
                f" Dynamics loss:{np.mean(dynl):.5f}"
                f" Policy loss Sim:{np.mean(simpl):.5f}"
                f" Value loss Sim:{np.mean(simvl):.5f}"
                f" Discounted Reward:{np.mean(dr):.3f}"
            )
            metrics = Trainer.evaluate_full_run(
                policy, kf, val_view,
                window_size=cfg.window_size,
                device=cfg.device,
                sample_policy=False,
            )
            val_sharpe = float(metrics["sharpe"])
            avg_weight_cols = _avg_weight_cols(metrics.get("weights"), tickers)
            print("Validation Sharpe:", val_sharpe)
            if save_best_path and val_sharpe > best_val_sharpe:
                best_val_sharpe = val_sharpe
                best_val_sharpe_epoch = epoch
                save_dir = os.path.dirname(save_best_path)
                if save_dir:
                    os.makedirs(save_dir, exist_ok=True)
                torch.save(
                    {
                        "epoch": epoch,
                        "val_sharpe": val_sharpe,
                        "policy_state_dict": policy.state_dict(),
                        "kf_state_dict": kf.state_dict() if kf is not None else None,
                        "cfg": vars(cfg),
                    },
                    save_best_path,
                )
        stats_rows.append(
            {
                "phase": "train",
                "epoch": epoch,
                "seed": seed,
                "window_size": window_size,
                "lam": float(lam),
                "total_loss": float(np.mean(tl)) if tl else np.nan,
                "policy_loss": float(np.mean(pl)) if pl else np.nan,
                "value_loss": float(np.mean(vl)) if vl else np.nan,
                "kf_loss": float(np.mean(kfl)) if kfl else np.nan,
                "dyn_loss": float(np.mean(dynl)) if dynl else np.nan,
                "sim_policy_loss": float(np.mean(simpl)) if simpl else np.nan,
                "sim_value_loss": float(np.mean(simvl)) if simvl else np.nan,
                "discounted_reward": float(np.mean(dr)) if dr else np.nan,
                "val_sharpe": val_sharpe,
                **avg_weight_cols,
            }
        )

    if verbose:
        print("-----------------Training Ended-----------------")
    if stats_csv_path and stats_rows:
        stats_dir = os.path.dirname(stats_csv_path)
        if stats_dir:
            os.makedirs(stats_dir, exist_ok=True)
        stats_df = pd.DataFrame(stats_rows)
        stats_df.to_csv(
            stats_csv_path,
            mode="a",
            header=not os.path.exists(stats_csv_path),
            index=False,
        )

    # evaluate once (test)
    used_best_policy = False
    if evaluate_best_on_test and save_best_path and os.path.exists(save_best_path):
        try:
            ckpt = torch.load(save_best_path, map_location=device)
        except pickle.UnpicklingError:
            from torch.serialization import safe_globals

            with safe_globals([TrainConfig]):
                ckpt = torch.load(save_best_path, map_location=device, weights_only=False)
        policy.load_state_dict(ckpt["policy_state_dict"])
        if kf is not None and ckpt.get("kf_state_dict") is not None:
            kf.load_state_dict(ckpt["kf_state_dict"])
        used_best_policy = True
        print("Using policy at epoch:", best_val_sharpe_epoch)

    metrics = Trainer.evaluate_full_run(
        policy, kf, test_view,
        window_size=cfg.window_size,
        device=cfg.device,
        sample_policy=False,
    )
    if verbose:
        print(
            f"test_sharpe: {float(metrics['sharpe'])},"
            f"test_total_return: {float(metrics['total_reward'])},"
            f"test_mean_return: {float(metrics['mean_return'])},"
            f"test_std_return: {float(metrics['std_return'])}"
        )

    weights = metrics["weights"]
    if weights:
        weights_tensor = torch.stack(weights, dim=0)
        avg_weights = weights_tensor.mean(dim=0).cpu().numpy()
    else:
        avg_weights = None

    return {
        "seed": seed,
        "window_size": window_size,
        "lam": float(lam),
        "test_sharpe": float(metrics["sharpe"]),
        "test_total_return": float(metrics["total_reward"]),
        "test_mean_return": float(metrics["mean_return"]),
        "test_std_return": float(metrics["std_return"]),
        "test_weights": metrics["weights"],
        "test_avg_weights": avg_weights,
        "tickers": tickers,
        "last_return0": float(out["return0"]),
        "best_val_sharpe": float(best_val_sharpe),
        "best_policy_path": save_best_path,
        "used_best_policy": used_best_policy,
    }

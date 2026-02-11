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
    eval_on_validation=False,
    stats_csv_path=None,
    networksize: int, 
    learnrate: float,
    data_bundle=None,
    print_results=False,
    kf_fixed=False,
    kf_q=1e-3,
    kf_r=1e-3,
):
    # one process = one thread (important for parallel speed)
    torch.set_num_threads(1)
    os.environ["OMP_NUM_THREADS"] = "1"
    os.environ["MKL_NUM_THREADS"] = "1"

    set_seed(seed)

    if data_bundle is None:
        tickers = ['MSFT', 'JPM', 'JNJ', 'XOM', 'PG']
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

        # inside backtest_seed.py after train_view is created
        rets = train_view.as_numpy()
        mu = rets.mean(axis=0)
        var = rets.var(axis=0)
        #print(list(zip(tickers, mu, var, mu - lam*var)))


        train_covs = train_view.precompute_expanding_cov(diag=True)
    else:
        train_view, val_view, test_view, train_covs = data_bundle


    # ---- use the SAME trainer instance across warmup+train ----
    if cfg is None:
        cfg = TrainConfig(
            device="cpu",
            T=(window_size+5),
            updates=1500,          # keep smaller for sweeps; increase later
            window_size=window_size,
            print_every=99999999,    # silence inside worker
            use_kf=True,
            episodes_per_batch=1,
            lam=float(lam),
            gamma=0.99,
            dyn_enabled=True,
            use_critic=False,
            kappa_unc=0.0,
            dyn_sim_deterministic=True,
            dyn_use_sim=True,
            actor_weight=0.0,
            dyn_sim_pl_weight=1.0,
            lr_actor = learnrate,
        )

    device = torch.device(cfg.device)
    K = train_view.K

    policy = PolicyNet(K=K,hidden=networksize).to(device)
    value = ValueNet(K).to(device)
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
    trainer = Trainer(policy, value, kf, cfg)
    stats_rows = []

    if verbose:
        print("-----------------Training KF-----------------")
    for epoch in range(100):
        #break
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

        if epoch % cfg.print_every == 0 and epoch != 0 and verbose:
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
            }
        )
    

    if verbose:
        print("-----------------Training Dynamics-----------------")

    # warmup (optional)
    for epoch in range(100):
        #break
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

        if epoch % cfg.print_every == 0 and epoch != 0 and verbose:
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
            }
        )

    if verbose:
        print("-----------------Warmup Critic Ended - Training Policy-----------------")

    best_val_sharpe = -np.inf
    best_val_sharpe_epoch = 0
    best_saved = False
    last_epoch = cfg.updates - 1 if cfg.updates > 0 else 0

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
        if epoch % cfg.print_every == 0 and epoch != 0:
            if verbose:
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
            if verbose: print("Validation Sharpe:", val_sharpe)
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
                best_saved = True
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
            }
        )

    if verbose:
        print("-----------------Training Ended-----------------")
    if stats_csv_path and stats_rows and not eval_on_validation:
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

    if save_best_path and not best_saved:
        save_dir = os.path.dirname(save_best_path)
        if save_dir:
            os.makedirs(save_dir, exist_ok=True)
        torch.save(
            {
                "epoch": last_epoch,
                "val_sharpe": float(best_val_sharpe),
                "policy_state_dict": policy.state_dict(),
                "kf_state_dict": kf.state_dict() if kf is not None else None,
                "cfg": vars(cfg),
            },
            save_best_path,
        )

    # evaluate once (test or validation-only)
    used_best_policy = False
    used_policy_epoch = last_epoch
    used_policy_path = save_best_path if save_best_path else None
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
        used_policy_epoch = int(ckpt.get("epoch", last_epoch))
        used_policy_path = save_best_path
        print("Using policy at epoch:", used_policy_epoch)

    eval_view = val_view if eval_on_validation else test_view
    metrics = Trainer.evaluate_full_run(
        policy, kf, eval_view,
        window_size=cfg.window_size,
        device=cfg.device,
        sample_policy=False,
    )
    if verbose or print_results:
        split_label = "val" if eval_on_validation else "test"
        print(
            f"{split_label}_sharpe: {float(metrics['sharpe'])},"
            f"{split_label}_total_return: {float(metrics['total_reward'])},"
            f"{split_label}_mean_return: {float(metrics['mean_return'])},"
            f"{split_label}_std_return: {float(metrics['std_return'])},"
            f"{split_label}_max_drawdown: {float(metrics['max_drawdown'])}"
        )

    return {
        "seed": seed,
        "window_size": window_size,
        "lam": float(lam),
        "test_sharpe": float(metrics["sharpe"]),
        "test_total_return": float(metrics["total_reward"]),
        "test_mean_return": float(metrics["mean_return"]),
        "test_std_return": float(metrics["std_return"]),
        "test_max_drawdown": float(metrics["max_drawdown"]),
        "test_weights": metrics["weights"],
        "last_return0": float(out["return0"]),
        "best_val_sharpe": float(best_val_sharpe),
        "best_val_sharpe_epoch": int(best_val_sharpe_epoch),
        "best_policy_path": save_best_path,
        "used_best_policy": used_best_policy,
        "used_policy_epoch": int(used_policy_epoch) if used_policy_epoch is not None else None,
        "used_policy_path": used_policy_path,
        "eval_split": "val" if eval_on_validation else "test",
    }

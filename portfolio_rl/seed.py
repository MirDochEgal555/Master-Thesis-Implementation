# run_experiment.py (or inside main.py)
import os
import torch
import numpy as np
import pandas as pd

from .config import TrainConfig
from .models import PolicyNet, ValueNet
from .kalman import LearnableKalman
from .trainer import Trainer
from .data import YahooConfig, YahooReturnsDataset
from .utils import set_seed


def run_one(seed: int, window_size: int, lam: float, networksize: int, learnrate: float, *, cache_path="returns.parquet", cfg=None):
    # one process = one thread (important for parallel speed)
    torch.set_num_threads(1)
    os.environ["OMP_NUM_THREADS"] = "1"
    os.environ["MKL_NUM_THREADS"] = "1"

    set_seed(seed)

    tickers = ['JPM', 'JNJ', 'XOM', 'PG', 'MSFT']
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
            dyn_enabled=False,
            use_critic=True,
            kappa_unc=0.0,
            dyn_sim_deterministic=False,
            lr_actor = learnrate,
            use_baseline=False,
        )

    device = torch.device(cfg.device)
    K = dataset.K

    policy = PolicyNet(K=K,hidden=networksize).to(device)
    value = ValueNet(K).to(device)
    kf = LearnableKalman(dim=K).to(device) if cfg.use_kf else None
    trainer = Trainer(policy, value, kf, cfg)

    #print("-----------------Warm Up-----------------")

    # warmup (optional)
    for epoch in range(50):
        #break
        state = None
        tl, pl, vl, kfl, dynl, simpl, simvl, dr = [], [], [], [], [], [], [], []
        for obs, cov in train_view.iter_windows_with_cov(T=cfg.T, stride=cfg.T, covs=train_covs):
            out, state = trainer.train_step(obs, update=epoch, state=state, warmupcrit=False, warmupdyn=True, warmupkf=False, covariances=cov)
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

    #print("-----------------Warm Up Ended-----------------")

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
            #metrics = Trainer.evaluate_full_run(
            #    policy, kf, val_view,
            #    window_size=cfg.window_size,
            #    lam=cfg.lam,
            #    device=cfg.device,
            #)
            #print("Validation Sharpe:", float(metrics["sharpe"]))

    #print("-----------------Training Ended-----------------")

    # evaluate once (validation)
    metrics = Trainer.evaluate_full_run(
        policy, kf, val_view,
        window_size=cfg.window_size,
        lam=cfg.lam,
        device=cfg.device,
    )

    return {
        "seed": seed,
        "window_size": window_size,
        "lam": float(lam),
        "netsize": networksize,
        "learnrate": float(learnrate),
        "test_sharpe": float(metrics["sharpe"]),
        "test_total_return": float(metrics["total_reward"]),
        "test_mean_return": float(metrics["mean_return"]),
        "test_std_return": float(metrics["std_return"]),
        "test_weights": metrics["weights"],
        "last_return0": float(out["return0"]),
    }
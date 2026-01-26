import numpy as np

from portfolio_rl.backtest_seed import run_one
from portfolio_rl.config import TrainConfig

if __name__ == "__main__":
    window_size=1
    lam=0.1

    cfg = TrainConfig(
            device="cpu",
            T=(window_size+5),
            updates=4500,          # keep smaller for sweeps; increase later
            window_size=window_size,
            print_every=5,    # silence inside worker
            use_kf=True,
            episodes_per_batch=1,
            lam=float(lam),
            gamma=0.99,
            dyn_enabled=True,
            use_critic=False,
            kappa_unc=0.0,
            dyn_sim_deterministic=False,
        )
    
    results = run_one(
        seed=0,
        window_size=cfg.window_size,
        lam=cfg.lam,
        cfg=cfg,
        verbose=True,
        save_best_path="checkpoints/best_policy_seed0.pt",
        evaluate_best_on_test=True,
        stats_csv_path="learning_stats_kf.csv",
        networksize=64,
        learnrate=1e-3,
    )

    np.savetxt("weights.txt", results["test_weights"], delimiter=",")

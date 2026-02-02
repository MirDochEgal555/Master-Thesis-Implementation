import numpy as np

from portfolio_rl.backtest_seed import run_one
from portfolio_rl.config import TrainConfig

if __name__ == "__main__":
    window_size=1
    lam=10.0

    cfg = TrainConfig(
            device="cpu",
            T=(window_size+5),
            updates=4500,          # keep smaller for sweeps; increase later
            window_size=window_size,
            print_every=5,    # silence inside worker
            use_kf=False,
            episodes_per_batch=1,
            lam=float(lam),
            gamma=0.99,
            dyn_enabled=False,
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
        networksize=32,
        learnrate=1e-5,
    )

    weights = results.get("test_weights", [])
    if weights:
        weights_np = np.stack(
            [
                w.detach().cpu().numpy() if hasattr(w, "detach") else np.asarray(w)
                for w in weights
            ],
            axis=0,
        )
        np.savetxt("weights.txt", weights_np, delimiter=",")

        avg_weights = results.get("test_avg_weights")
        if avg_weights is None:
            avg_weights = weights_np.mean(axis=0)

        tickers = results.get("tickers")
        if tickers:
            with open("avg_weights.csv", "w", encoding="utf-8") as f:
                f.write("asset,avg_weight\n")
                for asset, w in zip(tickers, avg_weights):
                    f.write(f"{asset},{w}\n")
        else:
            np.savetxt("avg_weights.txt", avg_weights, delimiter=",")

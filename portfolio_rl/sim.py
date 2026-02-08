# portfolio_rl/sim.py
import torch
from .rollouts import discounted_return, compute_reward
from .features import RollingMean

def _sample_next_state(pred_mean, pred_var, deterministic: bool):
    if deterministic:
        return pred_mean
    eps = torch.randn_like(pred_mean)
    return pred_mean + torch.sqrt(pred_var + 1e-8) * eps


def simulate_rollouts_from_dynamics(
    policy,
    value,
    dyn_model,
    kf,
    init_states: torch.Tensor,     # [B, K] initial state pool
    T_sim: int,
    gamma: float,
    lam: float,
    kappa_unc: float,
    turn_coef: float,
    turn_target: float,
    cost_coef: float,
    deterministic_next_state: bool = False,
    use_roll_features: bool = True,
    window_size: int = 1,
    roll_detach: bool = True,
    use_model_uncertainty: bool = True,
    use_kf_uncertainty: bool = True,
):
    """
    Returns lists of per-episode tensors:
      returns: list of [T_sim]
      values:  list of [T_sim]
    """
    device = init_states.device
    B, K = init_states.shape

    sim_returns = []
    sim_values = []

    eyeK = torch.eye(K, device=device)

    A = kf.A if (kf is not None and use_kf_uncertainty) else None
    Q = kf.Q if (kf is not None and use_kf_uncertainty) else None

    for b in range(B):
        # buffers (lists preserve autograd)
        vals = []
        rews = []

        # "state" here is the KF mean-like vector (K,)
        s_t = init_states[b]
        roll = RollingMean(window_size, K, device) if use_roll_features else None

        # prev portfolio
        w_prev = torch.full((K,), 1.0 / K, device=device)
        mean = torch.zeros(K, device=device)
        M2 = torch.zeros(K, K, device=device)
        n = 0
        cov_prev = eyeK
        eps = 1e-6
        V_t = eyeK.clone() if A is not None else None
        unc_prev = V_t.clone() if V_t is not None else torch.zeros_like(eyeK)

        for t in range(T_sim):
            base_feat = s_t
            feat = roll.update(base_feat, detach=roll_detach) if roll is not None else base_feat
            # critic
            v_pred = value(feat)
            vals.append(v_pred.squeeze())

            # deterministic policy (same as training)
            logits = policy(feat)
            w = torch.softmax(logits, dim=-1)
            w = (w + 1e-4) / (w.sum(dim=-1, keepdim=True) + 1e-12)

            # reward shaping identical style to training (use current state)
            r_rl, _bal = compute_reward(
                w_prev,
                s_t,
                cov_prev,
                unc_prev,
                lam=lam,
                kappa_unc=kappa_unc,
            )

            turn_l1 = (w - w_prev).abs().sum()
            turn_ratio = turn_l1 / (turn_target + 1e-12)
            turn_bonus = torch.exp(-0.5 * 3.0 * (turn_ratio - 1.0) ** 2)
            trade_cost = cost_coef * turn_l1

            r_t = r_rl - trade_cost + turn_coef * turn_bonus
            rews.append(r_t)

            # update empirical covariance with current return
            n += 1
            delta = s_t - mean
            mean = mean + delta / n
            delta2 = s_t - mean
            M2 = M2 + torch.outer(delta, delta2)

            if n >= 2:
                cov_prev = M2 / (n - 1)
            else:
                cov_prev = eyeK
            cov_prev = cov_prev + eps * eyeK

            # ----- dynamics step: s_{t+1} ~ N(mu, var) given (s_t, w) -----
            pred_mean, pred_var = dyn_model(s_t.unsqueeze(0), w.unsqueeze(0))  # [1,K]
            pred_mean = pred_mean.squeeze(0)
            pred_var = pred_var.squeeze(0)
            s_next = _sample_next_state(pred_mean, pred_var, deterministic_next_state)

            unc_next = torch.zeros_like(cov_prev)
            if use_model_uncertainty:
                model_unc = torch.diag(pred_var + eps)
                cov_prev = cov_prev + model_unc
                unc_next = unc_next + model_unc

            if A is not None and Q is not None:
                V_t = A @ V_t @ A.T + Q
                V_t = 0.5 * (V_t + V_t.T)
                cov_prev = cov_prev + V_t
                unc_next = unc_next + V_t

            unc_prev = unc_next

            # update (keep graph for BPTT through model)
            w_prev = w
            s_t = s_next
    

        vals = torch.stack(vals)
        rews = torch.stack(rews)
        rets = discounted_return(rews, gamma)

        sim_returns.append(rets)
        sim_values.append(vals)

    return sim_returns, sim_values

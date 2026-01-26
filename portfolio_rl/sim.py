# portfolio_rl/sim.py
import torch
from .rollouts import discounted_return, compute_reward, logistic_normal_log_prob_clr_from_y

@torch.no_grad()
def _sample_next_state(pred_mean, pred_var, deterministic: bool):
    if deterministic:
        return pred_mean
    eps = torch.randn_like(pred_mean)
    return pred_mean + torch.sqrt(pred_var + 1e-8) * eps


def simulate_rollouts_from_dynamics(
    policy,
    value,
    dyn_model,
    init_states: torch.Tensor,     # [B, K] initial state pool
    T_sim: int,
    gamma: float,
    lam: float,
    kappa_unc: float,
    turn_coef: float,
    turn_target: float,
    cost_coef: float,
    deterministic_next_state: bool = False,
):
    """
    Returns lists of per-episode tensors:
      logps:   list of [T_sim]
      returns: list of [T_sim]
      values:  list of [T_sim]
    """
    device = init_states.device
    B, K = init_states.shape

    sim_logps = []
    sim_returns = []
    sim_values = []

    eyeK = torch.eye(K, device=device)

    for b in range(B):
        # buffers
        logps = torch.empty(T_sim, device=device)
        vals  = torch.empty(T_sim, device=device)
        rews  = torch.empty(T_sim, device=device)

        # "state" here is the KF mean-like vector (K,)
        s_t = init_states[b]

        # prev portfolio
        w_prev = torch.full((K,), 1.0 / K, device=device)
        mean = torch.zeros(K, device=device)
        M2 = torch.zeros(K, K, device=device)
        n = 0
        cov_prev = eyeK
        eps = 1e-6

        for t in range(T_sim):
            # critic
            v_pred = value(s_t)
            vals[t] = v_pred.squeeze()

            # policy sample (same as training)
            loc, log_std = policy(s_t)
            loc = loc - loc.mean(dim=-1, keepdim=True)
            std = torch.exp(log_std)

            y = loc + std * torch.randn_like(loc)
            y = y - y.mean(dim=-1, keepdim=True)

            w = torch.softmax(y, dim=-1)
            w = (w + 1e-4) / (w.sum(dim=-1, keepdim=True) + 1e-12)

            logps[t] = logistic_normal_log_prob_clr_from_y(y, w, loc, log_std, tau=1.0).squeeze()

            # ----- dynamics step: s_{t+1} ~ N(mu, var) given (s_t, w) -----
            pred_mean, pred_var = dyn_model(s_t.unsqueeze(0), w.unsqueeze(0))  # [1,K]
            s_next = _sample_next_state(pred_mean.squeeze(0), pred_var.squeeze(0), deterministic_next_state)

            # reward shaping identical style to training
            r_rl, _bal = compute_reward(w_prev, s_next, cov_prev, lam, kappa_unc=kappa_unc)

            turn_l1 = (w.detach() - w_prev).abs().sum()
            turn_ratio = turn_l1 / (turn_target + 1e-12)
            turn_bonus = torch.exp(-0.5 * 3.0 * (turn_ratio - 1.0) ** 2)
            trade_cost = cost_coef * turn_l1

            r_t = r_rl - trade_cost + turn_coef * turn_bonus
            rews[t] = r_t

            # update
            w_prev = w.detach()
            s_t = s_next.detach()

            n += 1
            delta = s_next - mean
            mean = mean + delta / n
            delta2 = s_next - mean
            M2 = M2 + torch.outer(delta, delta2)

            if n >= 2:
                cov_prev = M2 / (n - 1)
            else:
                cov_prev = eyeK
            cov_prev = cov_prev + eps * eyeK
    

        rets = discounted_return(rews, gamma)

        sim_logps.append(logps)
        sim_returns.append(rets)
        sim_values.append(vals)

    return sim_logps, sim_returns, sim_values

# portfolio_rl/sim.py
import torch
from .rollouts import discounted_return

def _sample_next_state(pred_mean, pred_var, deterministic: bool):
    if deterministic:
        return pred_mean
    eps = torch.randn_like(pred_mean)
    return pred_mean + torch.sqrt(pred_var + 1e-8) * eps


def _compute_reward_batched(
    weights: torch.Tensor,       # [B, K]
    true_return: torch.Tensor,   # [B, K]
    cov_matrix: torch.Tensor,    # [B, K, K]
    unc_matrix: torch.Tensor,    # [B, K, K]
    lam: float,
    kappa_unc: float,
):
    rp = (weights * true_return).sum(dim=-1)
    risk = torch.einsum("bi,bij,bj->b", weights, cov_matrix, weights)
    reward = rp - lam * risk

    if kappa_unc != 0.0:
        K = unc_matrix.shape[-1]
        eye = torch.eye(K, device=unc_matrix.device, dtype=unc_matrix.dtype).unsqueeze(0)
        unc_sym = 0.5 * (unc_matrix + unc_matrix.transpose(-1, -2))
        unc_stable = unc_sym + 1e-6 * eye
        sign, logdet = torch.linalg.slogdet(unc_stable)
        logdet = torch.where(sign > 0, logdet, torch.zeros_like(logdet))
        reward = reward - kappa_unc * (0.5 * logdet)

    return reward


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
    compute_values: bool = True,
):
    """
    Vectorized simulation over B trajectories.
    Returns:
      returns: [B, T_sim]
      values:  [B, T_sim] or None if compute_values=False
    """
    device = init_states.device
    B, K = init_states.shape

    eyeK = torch.eye(K, device=device)
    eyeB = eyeK.unsqueeze(0).expand(B, -1, -1)
    eps = 1e-6

    A = kf.A if (kf is not None and use_kf_uncertainty) else None
    Q = kf.Q if (kf is not None and use_kf_uncertainty) else None
    # Vectorized rollout state over B trajectories.
    s_t = init_states
    w_prev = torch.full((B, K), 1.0 / K, device=device)
    mean = torch.zeros(B, K, device=device)
    M2 = torch.zeros(B, K, K, device=device)
    cov_prev = eyeB.clone()

    V_t = eyeK.clone() if A is not None else None
    unc_prev = eyeB.clone() if V_t is not None else torch.zeros_like(eyeB)

    if use_roll_features:
        roll_buf = torch.zeros(B, window_size, K, device=device)
        roll_sum = torch.zeros(B, K, device=device)
        roll_len = 0
        roll_i = 0

    vals = []
    rews = []
    n = 0

    if compute_values and value is None:
        raise ValueError("value network is required when compute_values=True")

    for _ in range(T_sim):
        if use_roll_features:
            x = s_t.detach() if roll_detach else s_t
            if roll_len < window_size:
                roll_buf[:, roll_i] = x
                roll_sum = roll_sum + x
                roll_len += 1
            else:
                roll_sum = roll_sum + x - roll_buf[:, roll_i]
                roll_buf[:, roll_i] = x
            roll_i = (roll_i + 1) % window_size
            feat = roll_sum / roll_len
        else:
            feat = s_t

        # critic and policy (batched)
        if compute_values:
            v_pred = value(feat)             # [B]
        logits = policy(feat)                # [B, K]
        w = torch.softmax(logits, dim=-1)
        w = (w + 1e-4) / (w.sum(dim=-1, keepdim=True) + 1e-12)

        r_rl = _compute_reward_batched(
            w_prev,
            s_t,
            cov_prev,
            unc_prev,
            lam=lam,
            kappa_unc=kappa_unc,
        )

        turn_l1 = (w - w_prev).abs().sum(dim=-1)
        turn_ratio = turn_l1 / (turn_target + 1e-12)
        turn_bonus = torch.exp(-0.5 * 3.0 * (turn_ratio - 1.0) ** 2)
        trade_cost = cost_coef * turn_l1
        r_t = r_rl - trade_cost + turn_coef * turn_bonus

        if compute_values:
            vals.append(v_pred)
        rews.append(r_t)

        # update empirical covariance with current return (batched Welford)
        n += 1
        delta = s_t - mean
        mean = mean + delta / n
        delta2 = s_t - mean
        M2 = M2 + delta.unsqueeze(-1) * delta2.unsqueeze(-2)

        if n >= 2:
            cov_prev = M2 / (n - 1)
        else:
            cov_prev = eyeB.clone()
        cov_prev = cov_prev + eps * eyeB

        # ----- dynamics step: s_{t+1} ~ N(mu, var) given s_t -----
        pred_mean, pred_var = dyn_model(s_t)  # [B, K]
        s_next = _sample_next_state(pred_mean, pred_var, deterministic_next_state)

        unc_next = torch.zeros_like(cov_prev)
        if use_model_uncertainty:
            model_unc = torch.diag_embed(pred_var + eps)  # [B, K, K]
            cov_prev = cov_prev + model_unc
            unc_next = unc_next + model_unc

        if A is not None and Q is not None:
            V_t = A @ V_t @ A.T + Q
            V_t = 0.5 * (V_t + V_t.T)
            V_batch = V_t.unsqueeze(0)
            cov_prev = cov_prev + V_batch
            unc_next = unc_next + V_batch

        unc_prev = unc_next

        # keep graph for BPTT through model
        w_prev = w
        s_t = s_next

    rews_t = torch.stack(rews, dim=0)      # [T_sim, B]
    rets_t = discounted_return(rews_t, gamma)

    vals_out = None
    if compute_values:
        vals_t = torch.stack(vals, dim=0)  # [T_sim, B]
        vals_out = vals_t.transpose(0, 1)

    return rets_t.transpose(0, 1), vals_out

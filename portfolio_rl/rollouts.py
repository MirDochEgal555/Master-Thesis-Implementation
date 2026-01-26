import torch, math

def discounted_return(rewards: torch.Tensor, gamma: float) -> torch.Tensor:
    T = rewards.shape[0]
    out = torch.empty_like(rewards)
    running = torch.zeros((), device=rewards.device, dtype=rewards.dtype)
    for t in range(T - 1, -1, -1):
        running = rewards[t] + gamma * running
        out[t] = running
    return out


def compute_reward(
    weights,
    true_return,
    cov_matrix,
    lam: float = 0.1,
    scale: float = 200,
    kappa_unc: float = 0.0,        # NEW: coefficient for belief-uncertainty penalty
):
    """
    weights:     tensor (N,)
    true_return: tensor (N,) → observed return z_t or filtered mean m_t
    cov_matrix:  tensor (N, N) → here: *belief* covariance Σ_t
    kappa_unc:   strength of uncertainty penalty; 0.0 = disabled
    """
    # Markowitz part: mean - λ * variance
    rp = torch.dot(weights, true_return)
    cov_annual = cov_matrix  # you can still rescale if you want
    risk = torch.matmul(weights.unsqueeze(0),
                        torch.matmul(cov_annual, weights.unsqueeze(1))).squeeze()
    reward = rp - lam * risk

    # --- belief-uncertainty penalty: κ * H(Σ_t) ≈ κ * 0.5 * log det Σ_t ---
    if kappa_unc != 0.0:
        N = cov_matrix.shape[0]
        eye = torch.eye(N, device=cov_matrix.device, dtype=cov_matrix.dtype)
        cov_stable = cov_matrix + 1e-6 * eye

        sign, logdet = torch.slogdet(cov_stable)
        # handle pathological cases: if sign <= 0, ignore the penalty
        logdet = torch.where(sign > 0, logdet, torch.zeros_like(logdet))

        # differential entropy of N-dim Gaussian is
        #   H = 0.5 * (N * (1 + log(2π)) + log det Σ)
        # constants don't change the optimum → use 0.5 * log det Σ
        H_approx = 0.5 * logdet

        # subtract, because it's a *penalty* on uncertainty
        reward = reward - kappa_unc * H_approx
    balance = rp / (risk + 1e-12)
    return reward, balance



def logistic_normal_log_prob_clr_from_y(y, w, loc, log_std, tau=1.0, eps=1e-8):
    # y, loc, log_std: (..., K) already zero-sum; w: (..., K)
    std = torch.exp(log_std)
    logp_y = -0.5 * (((y - loc) / std)**2 + 2*log_std + math.log(2*math.pi)).sum(-1)
    # Jacobian (softmax on zero-sum subspace)
    K = w.shape[-1]
    logJ = (K - 1) * math.log(tau) - torch.log(w.clamp_min(eps)).sum(-1)
    return logp_y + logJ


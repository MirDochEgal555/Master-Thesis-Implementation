import torch
import torch.nn.utils as U
import numpy as np

from .rollouts import discounted_return, compute_reward, dirichlet_log_prob
from .dynamics import DynamicsModel, gaussian_nll
from .features import RollingMean
from .sim import simulate_rollouts_from_dynamics


class Trainer:
    def __init__(self, policy, value, kalman_filter, cfg, dyn_model=None):
        self.policy = policy
        self.value = value
        self.kf = kalman_filter
        self.cfg = cfg
        self.dyn = dyn_model
        self.opt_dyn = None
        self.opt_critic = None

        self.device = next(policy.parameters()).device

        if cfg.dyn_enabled:
            if self.dyn is None:
                self.dyn = DynamicsModel(K=policy.K if hasattr(policy, "K") else cfg.K, hidden=cfg.dyn_hidden).to(self.device)
            self.opt_dyn = torch.optim.Adam(self.dyn.parameters(), lr=cfg.lr_dyn)

        self.opt_actor = torch.optim.Adam(self.policy.parameters(), lr=cfg.lr_actor)
        if cfg.use_critic:
            self.opt_critic = torch.optim.Adam(self.value.parameters(), lr=cfg.lr_critic)
        self.opt_kf = None
        if self.kf is not None:
            self.opt_kf = torch.optim.Adam(self.kf.parameters(), lr=cfg.lr_kf)

    def train_step(self, observations: torch.Tensor, update: int, state=None, warmupdyn=False, warmupcrit=False, warmupkf=False, covariances=None):
        """
        observations: [T, K] tensor (already on device recommended)
        """
        cfg = self.cfg
        device = self.device
        obs = observations.to(device)
        covs = covariances.to(device) if covariances is not None else None

        T, K = obs.shape

        eyeK = torch.eye(K, device=device)
        eps_eyeK = 1e-6 * eyeK

        # ---- create or reuse state ----
        if state is None:
            state = {}

        roll = state.get("roll")
        if roll is None:
            roll = RollingMean(cfg.window_size, K, device)
            state["roll"] = roll

        # KF init
        if self.kf is not None:
            I = torch.eye(self.kf.dim, device=device)
            A, Q, R = self.kf.A, self.kf.Q, self.kf.R
            m_t = state.get("m_t", torch.zeros(K, device=device))
            V_t = state.get("V_t", torch.eye(K, device=device))
            state["m_t"], state["V_t"] = m_t.detach(), V_t.detach()
        else:
            m_t = None
            V_t = eyeK

        # rollout buffers
        logps   = []
        values  = torch.empty(T, device=device)
        rewards = []

        # ---- portfolio state carry ----
        logp_prev = torch.zeros((), device=device)
        w_prev = state.get("w_prev")
        if w_prev is None:
            w_prev = torch.full((K,), 1.0 / K, device=device)
        else:
            w_prev = w_prev.detach().to(device)

        cov_prev = state.get("cov_prev")
        if cov_prev is None:
            cov_prev = torch.eye(K, device=device)
        else:
            cov_prev = cov_prev.detach().to(device)


        # rolling mu mean
        mu_buf = torch.empty((cfg.window_size, K), device=device)
        mu_sum = torch.zeros(K, device=device)
        mu_len = 0
        mu_i = 0

        kf_nll_terms = []

        dyn_states = []
        dyn_actions = []
        dyn_targets = []

        prev_state = None
        prev_action = None

        real_state_pool = []

        for t in range(T):
            z_t = obs[t]
            cov_used = covs[t] if covs is not None else cov_prev


            # ----- KF step -----
            if self.kf is not None:
                m_pred = A @ m_t
                V_pred = A @ V_t @ A.T + Q
                y_t = z_t - m_pred
                S_t = V_pred + R
                S_t = 0.5 * (S_t + S_t.T)
                S_chol = torch.linalg.cholesky(S_t + eps_eyeK)
                X = torch.cholesky_solve(V_pred.T, S_chol)
                K_t = X.T

                if self.opt_kf is not None:
                    logdet_S = 2.0 * torch.log(torch.diag(S_chol) + 1e-12).sum()
                    quad = (y_t.unsqueeze(0) @ torch.cholesky_solve(y_t.unsqueeze(-1), S_chol)).squeeze()
                    kf_nll_terms.append(0.5 * (logdet_S + quad))

                m_t = m_pred + K_t @ y_t
                V_t = (I - K_t) @ V_pred @ (I - K_t).T + K_t @ R @ K_t.T

                # rolling mean of m_t
                if mu_len < cfg.window_size:
                    mu_buf[mu_i] = m_t
                    mu_sum += m_t
                    mu_len += 1
                else:
                    mu_sum += m_t - mu_buf[mu_i]
                    mu_buf[mu_i] = m_t
                mu_i = (mu_i + 1) % cfg.window_size
                feat = mu_sum / mu_len
                cov_full = V_t + eps_eyeK
            else:
                feat = z_t
                cov_full = cov_prev + eps_eyeK

            base_feat = (m_t.detach() if self.kf is not None else z_t.detach())
            feat = roll.update(base_feat)   # <--- moving average feature
            #feat = feat - feat.mean()
            #feat = feat / (feat.std(unbiased=False) + 1e-6)

            # pick a consistent state representation for dynamics + sim init
            state_t = (m_t if self.kf is not None else z_t).detach()
            real_state_pool.append(state_t)


            # ----- value -----
            v_pred = self.value(feat)
            values[t] = v_pred

            # ----- policy sample -----
            concentration = self.policy(feat)
            dist = torch.distributions.Dirichlet(concentration)
            w = dist.sample()
            eps = 1e-4
            w = w.clamp_min(eps)
            w = w / w.sum(dim=-1, keepdim=True)

            # state for dynamics
            curr_state = (m_t if self.kf is not None else z_t).detach()

            if prev_state is not None and prev_action is not None and self.opt_dyn is not None:
                dyn_states.append(prev_state)
                dyn_actions.append(prev_action)
                dyn_targets.append(curr_state)

            prev_state = curr_state
            prev_action = w.detach()


            logps.append(logp_prev)

            # turnover shaping + cost
            turn_l1 = (w.detach() - w_prev).abs().sum()
            turn_ratio = turn_l1 / (cfg.turn_target + 1e-12)
            turn_bonus = torch.exp(-0.5 * 3.0 * (turn_ratio - 1.0) ** 2)
            trade_cost = cfg.cost_coef * turn_l1
            # reward
            r_rl, _balance = compute_reward(w_prev, z_t, cov_used, cfg.lam, kappa_unc=cfg.kappa_unc)
            #print(_balance)
            r_t = r_rl - trade_cost + cfg.turn_coef * turn_bonus
            rewards.append(r_t)

            # update prev
            logp_prev = dirichlet_log_prob(w, concentration)
            w_prev = w.detach()
            if covs is None:
                cov_prev = cov_full.detach()

        rewards = torch.stack(rewards)
        logps = torch.stack(logps)

        returns = discounted_return(rewards, cfg.gamma)

        # losses
        baseline = values.detach() if self.opt_critic else returns.mean().detach()
        adv = returns - baseline if self.cfg.use_baseline else returns
        if cfg.normalize_adv:
            adv = (adv - adv.mean()) / (adv.std(unbiased=False) + 1e-8)

        policy_loss = -(logps * adv.detach()).mean() if not warmupdyn and not warmupcrit and not warmupkf else torch.tensor(0.0)
        value_loss = 0.5 * (returns - values).pow(2).mean() if self.opt_critic is not None and True else torch.tensor(0.0)

        kf_loss = torch.stack(kf_nll_terms).mean() if (self.opt_kf is not None and len(kf_nll_terms) > 0 and True) else torch.tensor(0.0, device=device)

        dyn_loss = torch.tensor(0.0, device=device)

        if self.opt_dyn is not None and len(dyn_states) > 0:
            states  = torch.stack(dyn_states, dim=0)   # [N, K]
            actions = torch.stack(dyn_actions, dim=0)  # [N, K]
            targets = torch.stack(dyn_targets, dim=0)  # [N, K]

            pred_mean, pred_var = self.dyn(states, actions)
            nll = gaussian_nll(targets, pred_mean, pred_var).sum(dim=-1).mean()
            dyn_loss = nll if warmupdyn else torch.tensor(0.0)

        sim_policy_loss = torch.tensor(0.0, device=device)
        sim_value_loss = torch.tensor(0.0, device=device)

        if (self.opt_dyn is not None) and cfg.dyn_use_sim and len(real_state_pool) > 0:
            pool = torch.stack(real_state_pool, dim=0)  # [T, K]
            M = cfg.dyn_sim_M  # e.g. 50

            idx = torch.randint(0, pool.shape[0], (M,), device=device)
            init_states = pool[idx]                     # [M, K]

            sim_logps, sim_returns, sim_values = simulate_rollouts_from_dynamics(
                policy=self.policy,
                value=self.value,
                dyn_model=self.dyn,
                init_states=init_states,
                T_sim=T,
                gamma=cfg.gamma,
                lam=cfg.lam,
                kappa_unc=cfg.kappa_unc,
                turn_coef=cfg.turn_coef,
                turn_target=cfg.turn_target,
                cost_coef=cfg.cost_coef,
                deterministic_next_state=cfg.dyn_sim_deterministic,
            )

            # ---- pick ONE trajectory ----
            j = torch.randint(0, len(sim_returns), (1,)).item()

            sim_logps_j   = sim_logps[j]      # [T]
            sim_returns_j = sim_returns[j]    # [T]
            sim_values_j  = sim_values[j]     # [T]

            sim_adv = sim_returns_j - sim_values_j.detach()
            if cfg.normalize_adv:
                sim_adv = (sim_adv - sim_adv.mean()) / (sim_adv.std(unbiased=False) + 1e-8)

            sim_policy_loss = -(sim_logps_j * sim_adv.detach()).mean() #if update > 200 else torch.tensor(0.0)
            sim_value_loss  = torch.tensor(0.0) #0.5 * (sim_returns_j - sim_values_j).pow(2).mean()


        total_loss = (
            cfg.actor_weight * policy_loss 
            + cfg.critic_weight * value_loss
            + cfg.kf_weight * kf_loss
            + cfg.dyn_weight * dyn_loss
            + cfg.dyn_sim_pl_weight * sim_policy_loss 
            + cfg.dyn_sim_vl_weight * sim_value_loss
        )



        # step
        self.opt_actor.zero_grad(set_to_none=True)
        if self.opt_critic is not None:
            self.opt_critic.zero_grad(set_to_none=True)
        if self.opt_kf is not None:
            self.opt_kf.zero_grad(set_to_none=True)

        if self.opt_dyn is not None:
            self.opt_dyn.zero_grad(set_to_none=True)

        total_loss.backward()

        if not warmupdyn and not warmupcrit and not warmupkf:
            self.opt_actor.step()

        
        torch.nn.utils.clip_grad_norm_(self.policy.parameters(), cfg.max_grad_norm_act)
        torch.nn.utils.clip_grad_norm_(self.value.parameters(), cfg.max_grad_norm_crit)
        if self.opt_kf is not None:
            torch.nn.utils.clip_grad_norm_(self.kf.parameters(), 10.0)

        if self.opt_dyn is not None:
            torch.nn.utils.clip_grad_norm_(self.dyn.parameters(), 10.0)


        
        if self.opt_critic is not None and True:
            self.opt_critic.step()
        if self.opt_kf is not None and True:
            self.opt_kf.step()
            #print(self.kf.A,self.kf.Q,self.kf.R)
            #with torch.no_grad():
                #self.kf.project_A_spectral(rho_max=0.98)

        if self.opt_dyn is not None and True:
            self.opt_dyn.step()

        if self.kf is not None:
            state["m_t"] = m_t.detach()
            state["V_t"] = V_t.detach()

        state["w_prev"] = w_prev.detach()
        state["cov_prev"] = cov_prev.detach()
        state["roll"].detach_()

        return {
            "total_loss": float(total_loss.detach().cpu()),
            "policy_loss": float(policy_loss.detach().cpu()),
            "value_loss": float(value_loss.detach().cpu()),
            "kf_loss": float(kf_loss.detach().cpu()),
            "dyn_loss": float(dyn_loss.detach().cpu()),
            "sim_policy_loss": float(sim_policy_loss.detach().cpu()),
            "sim_value_loss": float(sim_value_loss.detach().cpu()),
            "return0": float(returns[0].detach().cpu()),
        }, state

    @staticmethod
    @torch.no_grad()
    def evaluate_full_run(
        policy,
        kalman_filter,
        data_view,              # YahooReturnsDatasetView OR torch.Tensor [N,K]
        device="cpu",
        sample_policy=False,    # <- set True to match notebook stochastic behavior
        eps_weight=1e-4,
        window_size=10,
    ):
        policy.eval()
        if kalman_filter is not None:
            kalman_filter.eval()

        # get full returns tensor [N,K]
        if hasattr(data_view, "row_idx"):  # YahooReturnsDatasetView
            x = torch.from_numpy(data_view.parent.returns_np[data_view.row_idx]).to(device)
        else:
            x = data_view.to(device)

        N, K = x.shape
        eyeK = torch.eye(K, device=device)

        w_prev = torch.full((K,), 1.0 / K, device=device)
        cov_prev = eyeK.clone()

        if kalman_filter is not None:
            A, Q, R = kalman_filter.A, kalman_filter.Q, kalman_filter.R
            m_t = torch.zeros(K, device=device)
            V_t = eyeK.clone()
            I = eyeK

        rewards = torch.empty(N, device=device)
        pure = torch.empty(N, device=device)
        weights = []

        roll = RollingMean(window_size, K, device)


        for t in range(N):
            z_t = x[t]

            # KF update (stable: cholesky_solve)
            if kalman_filter is not None:
                m_pred = A @ m_t
                V_pred = A @ V_t @ A.T + Q
                y_t = z_t - m_pred
                S = V_pred + R
                S = 0.5 * (S + S.T)
                S_chol = torch.linalg.cholesky(S + 1e-6 * eyeK)
                X = torch.cholesky_solve(V_pred.T, S_chol)
                K_t = X.T

                m_t = m_pred + K_t @ y_t
                V_t = (I - K_t) @ V_pred @ (I - K_t).T + K_t @ R @ K_t.T

                feat = m_t
                cov_full = V_t + 1e-6 * eyeK
            else:
                feat = z_t
                cov_full = cov_prev

            base_feat = m_t if kalman_filter is not None else z_t
            feat = roll.update(base_feat)

            # policy
            concentration = policy(feat)
            if sample_policy:
                dist = torch.distributions.Dirichlet(concentration)
                w = dist.sample()
            else:
                w = concentration / concentration.sum(dim=-1, keepdim=True)

            w = w + eps_weight
            w = w / w.sum()

            pure[t] = torch.dot(w_prev, z_t)
            weights.append(w.detach().cpu())

            w_prev = w
            cov_prev = cov_full

        sharpe = pure.mean() / (pure.std() + 1e-8) * np.sqrt(252)
        cumret = 1 + pure.sum()

        return {
            "total_reward": float(cumret),
            "sharpe": float(sharpe),
            "mean_return": float(pure.mean()),
            "std_return": float(pure.std()),
            "pure_returns": pure.cpu().numpy(),
            "weights": weights,
        }

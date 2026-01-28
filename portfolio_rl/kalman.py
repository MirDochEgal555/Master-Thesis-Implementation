import torch
import torch.nn as nn

class LearnableKalman(nn.Module):
    def __init__(self, dim: int, Q=None, R=None):
        super().__init__()
        self.dim = dim

        # Fixed A (identity). Register as buffer so .to() moves it.
        self.register_buffer("A", torch.eye(dim))

        # Diagonal Q/R (simple + stable). Use log-params to keep them positive.
        init = torch.full((dim,), 1e-3)
        self.log_q = nn.Parameter(init.log())
        self.log_r = nn.Parameter(init.log())

        if Q is not None:
            self.fixed_Q = Q if torch.is_tensor(Q) else torch.as_tensor(Q, dtype=self.A.dtype)
        else:
            self.fixed_Q = None
        if R is not None:
            self.fixed_R = R if torch.is_tensor(R) else torch.as_tensor(R, dtype=self.A.dtype)
        else:
            self.fixed_R = None

    @property
    def Q(self):
        if self.fixed_Q is None:
            return torch.diag(torch.exp(self.log_q) + 1e-8)
        return self.fixed_Q.to(device=self.A.device, dtype=self.A.dtype)

    @property
    def R(self):
        if self.fixed_R is None:
            return torch.diag(torch.exp(self.log_r) + 1e-8)
        return self.fixed_R.to(device=self.A.device, dtype=self.A.dtype)

    @torch.no_grad()
    def project_A_spectral(self, rho_max: float = 0.98, iters: int = 15):
        # cheap spectral radius estimate (power iteration) then scale if needed
        A = self.A
        v = torch.randn(A.shape[0], device=A.device)
        v = v / (v.norm() + 1e-12)
        for _ in range(iters):
            v = A @ v
            v = v / (v.norm() + 1e-12)
        rho = (A @ v).norm().clamp_min(1e-12)
        if rho > rho_max:
            self.A.mul_(rho_max / rho)

    @torch.no_grad()
    def filter(self, observations):
        """
        observations: np.array of shape (T, N) - observed noisy returns
        returns: list of tuples (m_t, V_t) for each t
        """
        observations = torch.as_tensor(observations, device=self.A.device, dtype=self.A.dtype)
        T = observations.shape[0]
        m = torch.zeros(self.dim, device=self.A.device, dtype=self.A.dtype)  # initial mean
        V = torch.eye(self.dim, device=self.A.device, dtype=self.A.dtype)    # initial covariance (can be tuned)
        I = torch.eye(self.dim, device=self.A.device, dtype=self.A.dtype)

        results = []
        nis_list = []
        winv_list = []

        for t in range(T):
            # Prediction
            m_pred = self.A @ m
            V_pred = self.A @ V @ self.A.T + self.Q

            # Update
            z = observations[t]
            y = z - m_pred
            S = V_pred + self.R
            K = V_pred @ torch.linalg.solve(S, I)

            nis = float(y.T @ torch.linalg.solve(S, y))

            L = torch.linalg.cholesky(S)
            w = torch.linalg.solve(L, y)  # cov(w) ~ I


            m = m_pred + K @ y

            IK = I - K
            V = IK @ V_pred @ IK.T + K @ self.R @ K.T


            results.append((m.clone(), V.clone()))
            nis_list.append(nis)
            winv_list.append(w.clone())

        diagnostics = {
                "nis": torch.tensor(nis_list, device=self.A.device, dtype=self.A.dtype),  # shape (T,)
                "whitened_innovation": torch.vstack(winv_list),  # shape (T, dim)
            }

        return results, diagnostics

import torch
import torch.nn as nn

class LearnableKalman(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim

        # Parameterize A directly (you may later reparam for stability)
        self.A = torch.eye(dim)

        # Diagonal Q/R (simple + stable). Use log-params to keep them positive.
        init = torch.full((dim,), 1e-3)
        self.log_q = nn.Parameter(init.log())
        self.log_r = nn.Parameter(init.log())

    @property
    def Q(self):
        return 0.0032*1e-3*torch.eye(self.dim)#torch.diag(torch.exp(self.log_q) + 1e-8)

    @property
    def R(self):
        return 0.00038*1e-3*torch.eye(self.dim)#torch.diag(torch.exp(self.log_r) + 1e-8)

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

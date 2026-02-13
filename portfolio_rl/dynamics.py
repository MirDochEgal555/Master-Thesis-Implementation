# portfolio_rl/dynamics.py
import torch
import torch.nn as nn
import torch.nn.functional as F

class DynamicsModel(nn.Module):
    """
    Predict next_state ~ N(mu, diag(var)) given state.
    state: [B, K], action: [B, K]
    returns:
      mean: [B, K]
      var:  [B, K]  (positive)
    """
    def __init__(self, K: int, hidden: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(K, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
        )
        self.mean_head = nn.Linear(hidden, K)
        self.logvar_head = nn.Linear(hidden, K)

    def forward(self, state: torch.Tensor):
        x = torch.cat([state], dim=-1)
        h = self.net(x)
        mean = self.mean_head(h)
        logvar = self.logvar_head(h).clamp(-10.0, 5.0)  # keep sane
        var = torch.exp(logvar)
        return mean, var


def gaussian_nll(target, mean, var, eps=1e-8):
    # elementwise diag Gaussian NLL
    return 0.5 * (((target - mean) ** 2) / (var + eps) + torch.log(var + eps))

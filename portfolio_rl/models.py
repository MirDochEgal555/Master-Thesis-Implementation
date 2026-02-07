import torch.nn as nn

# models.py
class PolicyNet(nn.Module):
    def __init__(self, K, hidden=128):
        super().__init__()
        self.K = K
        self.net = nn.Sequential(
            nn.Linear(K, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
        )
        self.logit_head = nn.Linear(hidden, K)

    def forward(self, x):
        h = self.net(x)
        return self.logit_head(h)


class ValueNet(nn.Module):
    def __init__(self, K: int, hidden: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(K, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, 1),
        )

    def forward(self, x):
        return self.net(x).squeeze(-1)

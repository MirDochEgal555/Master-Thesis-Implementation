# portfolio_rl/features.py
import torch

class RollingMean:
    """
    Rolling mean over last W vectors (dim=K).
    O(1) update cost per step using a ring buffer.
    """
    def __init__(self, W: int, K: int, device: torch.device):
        self.W = W
        self.K = K
        self.device = device
        self.buf = torch.zeros(W, K, device=device)
        self.sum = torch.zeros(K, device=device)
        self.len = 0
        self.i = 0

    def reset(self):
        self.buf.zero_()
        self.sum.zero_()
        self.len = 0
        self.i = 0

    def update(self, x: torch.Tensor) -> torch.Tensor:
        x = x.detach()
        # x: [K]
        if self.len < self.W:
            self.buf[self.i] = x
            self.sum += x
            self.len += 1
        else:
            self.sum += x - self.buf[self.i]
            self.buf[self.i] = x
        self.i = (self.i + 1) % self.W
        return self.sum / self.len
    
    def detach_(self):
        self.buf = self.buf.detach()
        self.sum = self.sum.detach()

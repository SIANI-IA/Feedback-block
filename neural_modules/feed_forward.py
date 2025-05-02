import torch
import torch.nn as nn

from neural_modules.gelu import GELU


class FeedForward(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(cfg["emb_dim"], 4 * cfg["emb_dim"]),
            GELU(),
            nn.Linear(4 * cfg["emb_dim"], cfg["emb_dim"]),
        )

    def forward(self, x):
        return self.layers(x)
    
class AnisotropicFeedForward(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.up_layer = nn.Linear(cfg["emb_dim"], 4*cfg["emb_dim"])
        self.down_layer = nn.Linear(2*cfg["emb_dim"], cfg["emb_dim"])
        self.cfg = cfg

    def forward(self, x):
        x = self.up_layer(x)
        x1, x2 = torch.chunk(x, 2, dim=2)
        x = x1 * torch.tanh(x2)
        x = self.down_layer(x)
        return x
import torch
import torch.nn as nn

from neural_modules.gelu import GELU

class BlockSelector(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int):
        super().__init__()
        self.pool = nn.AdaptiveAvgPool1d(1)  # Reduce la dimensión de secuencia
        
        # MLP para clasificación
        self.fc = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            GELU(),
            nn.Linear(hidden_dim, output_dim),
            nn.Softmax(dim=-1),
        )

    def forward(self, x):
        x = x.permute(2, 0, 1)
        x = self.pool(x)
        x = x.view(1, -1) # flatten
        x = self.fc(x)
        return x

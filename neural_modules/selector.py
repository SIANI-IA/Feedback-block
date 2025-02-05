import torch
import torch.nn as nn

from neural_modules.gelu import GELU

"""class BlockSelector(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int):
        super().__init__()
        self.pool = nn.AdaptiveAvgPool2d((1,1))  # Reduce la dimensión de secuencia
        
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
        return x"""

import torch
import torch.nn as nn
import torch.nn.functional as F

class BlockSelector(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int, num_heads: int = 4, temperature: float = 0.0):
        super().__init__()

        # Capa de atención para capturar relaciones en las dimensiones variables
        self.attn = nn.MultiheadAttention(embed_dim=input_dim, num_heads=num_heads, batch_first=True)

        # MLP para clasificación
        self.fc = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            GELU(),
            nn.Linear(hidden_dim, output_dim),
        )

        self.temperature = temperature

    def forward(self, x):
        """
        Entrada esperada: x de tamaño (var1, var2, 768)
        """

        var1, var2, input_dim = x.shape  # (2, 256, 768)

        # Convertimos a secuencia: tratamos (var1, var2) como una secuencia de tokens
        x = x.view(var1 * var2, input_dim)  # (2*256, 768)

        # Atención sobre la secuencia
        x = x.unsqueeze(0)  # Agregamos dimensión de batch (1, 2*256, 768)
        x, _ = self.attn(x, x, x)  # (1, 2*256, 768)
        
        # Reducimos la secuencia promediando sobre la dimensión de secuencia
        x = x.mean(dim=1)  # (1, 768)

        # Pasamos al MLP
        x = self.fc(x) # (1, output_dim)

        # Aplicamos softmax
        if self.temperature > 0.0:
            x = x / self.temperature
            x = F.softmax(x, dim=-1)
        else:
            x = F.softmax(x, dim=-1)

        return x.squeeze(0)  # Eliminamos la dimensión de batch (output_dim,)
    
class SttoperNeuralModel(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, num_heads: int = 4):
        super().__init__()

        # Capa de atención para capturar relaciones en las dimensiones variables
        self.attn = nn.MultiheadAttention(embed_dim=input_dim, num_heads=num_heads, batch_first=True)

        # MLP para clasificación
        self.fc = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            GELU(),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        """
        Entrada esperada: x de tamaño (var1, var2, 768)
        """

        var1, var2, input_dim = x.shape  # (2, 256, 768)

        # Convertimos a secuencia: tratamos (var1, var2) como una secuencia de tokens
        x = x.view(var1 * var2, input_dim)  # (2*256, 768)

        # Atención sobre la secuencia
        x = x.unsqueeze(0)  # Agregamos dimensión de batch (1, 2*256, 768)
        x, _ = self.attn(x, x, x)  # (1, 2*256, 768)
        
        # Reducimos la secuencia promediando sobre la dimensión de secuencia
        x = x.mean(dim=1)  # (1, 768)

        # Pasamos al MLP
        x = self.fc(x)

        return x.squeeze(0)  # Eliminamos la dimensión de batch (output_dim,)"""
        



# main
import torch

from neural_modules.gpt import *
from neural_modules.selector import BlockSelector
from utils import count_parameters


if __name__ == "__main__":
    GPT_CONFIG_124M = {
        "vocab_size": 50257,   # Vocabulary size
        "context_length": 256, # Shortened context length (orig: 1024)
        "emb_dim": 768,        # Embedding dimension
        "n_heads": 12,         # Number of attention heads
        "n_layers": 12,        # Number of layers
        "drop_rate": 0.1,      # Dropout rate
        "qkv_bias": False,     # Query-key-value bias
        "n_iter": 3,           # Number of iterations
        "batch_size": 8,
    }
    SEED = 123
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    torch.manual_seed(SEED)
    model = FeedbackGPT(GPT_CONFIG_124M) #GPTModel(GPT_CONFIG_124M)
    model.to(device)

    #generate a tensro Size([1, 256, 768])

    x = torch.randn(64, 512, 768)
    print(x.shape)
    # transpose the tensor to Size([768, 2, 256])
    x = x.permute(2, 0, 1)
    print(x.shape)
    pool = nn.AdaptiveAvgPool2d((1,1)) 
    x = pool(x)
    print(x.shape)
    x = x.view(1, -1)
    print(x.shape)


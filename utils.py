import random
import numpy as np
import torch

import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator

def text_to_token_ids(text, tokenizer):
    encoded = tokenizer.encode(text, allowed_special={'<|endoftext|>'})
    encoded_tensor = torch.tensor(encoded).unsqueeze(0) # add batch dimension
    return encoded_tensor

def token_ids_to_text(token_ids, tokenizer):
    flat = token_ids.squeeze(0) # remove batch dimension
    return tokenizer.decode(flat.tolist())

def generate(model, idx, max_new_tokens, context_size, temperature=0.0, top_k=None, eos_id=None):
    for _ in range(max_new_tokens):
        idx_cond = idx[:, -context_size:]  # Extrae el contexto más reciente
        with torch.no_grad():
            logits = model(idx_cond)
            logits = logits[:, -1, :]  # Obtiene los logits del último token generado

            # Aplicar muestreo Top-K si está habilitado
            if top_k is not None:
                top_logits, _ = torch.topk(logits, top_k)
                min_val = top_logits[:, -1]
                logits = torch.where(
                    logits < min_val, 
                    torch.tensor(float('-inf')).to(logits.device), 
                    logits
                )

            # Aplicar escalado de temperatura si está habilitado
            if temperature > 0.0:
                logits = logits / temperature
                probs = torch.softmax(logits, dim=-1)
                idx_next = torch.multinomial(probs, num_samples=1)  # Muestreo probabilístico
            else:
                idx_next = torch.argmax(logits, dim=-1, keepdim=True)  # Decodificación codiciosa (greedy)

            # Verificar si se ha alcanzado un token de finalización (EOS)
            if eos_id is not None and idx_next == eos_id:
                break

            # Concatenar el nuevo token generado a la secuencia
            idx = torch.cat((idx, idx_next), dim=1)

    return idx

def plot_losses(epochs_seen, tokens_seen, train_losses, val_losses):
    fig, ax1 = plt.subplots(figsize=(5, 3))

    # Plot training and validation loss against epochs
    ax1.plot(epochs_seen, train_losses, label="Training loss")
    ax1.plot(epochs_seen, val_losses, linestyle="-.", label="Validation loss")
    ax1.set_xlabel("Epochs")
    ax1.set_ylabel("Loss")
    ax1.legend(loc="upper right")
    ax1.xaxis.set_major_locator(MaxNLocator(integer=True))  # only show integer labels on x-axis

    # Create a second x-axis for tokens seen
    ax2 = ax1.twiny()  # Create a second x-axis that shares the same y-axis
    ax2.plot(tokens_seen, train_losses, alpha=0)  # Invisible plot for aligning ticks
    ax2.set_xlabel("Tokens seen")

    fig.tight_layout()  # Adjust layout to make room
    plt.savefig("loss-plot.pdf")
    plt.show()

def format_number(num):
    """
    Converts large numbers into a human-readable format.
    Example: 
      - 1,000,000 -> '1M'
      - 1,000,000,000 -> '1B'
    """
    if num >= 1_000_000_000:
        return f"{num / 1_000_000_000:.1f}B"
    elif num >= 1_000_000:
        return f"{num / 1_000_000:.1f}M"
    elif num >= 1_000:
        return f"{num / 1_000:.1f}K"
    else:
        return str(num)

def count_parameters_per_module(model):
    """
    Prints the number of trainable parameters for each module in a PyTorch Lightning model.
    Uses `format_number` to display numbers in a compact format.
    """
    for module_name, module in model.named_modules():
        if len(list(module.children())) == 0:  # Ensures it's a leaf module
            num_params = sum(p.numel() for p in module.parameters() if p.requires_grad)
            print(f"Module '{module_name}' has {format_number(num_params)} trainable parameters.")

def count_parameters(model):
    """
    Returns the total number of trainable parameters in a model,
    formatted in a human-readable way.
    """
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model has {format_number(total_params)} trainable parameters.")


def plot_histogram(histogram_of_chosen_blocks: dict):
    # Extraer claves (capas) y valores (frecuencia de elección)
    layers = list(histogram_of_chosen_blocks.keys())
    counts = list(histogram_of_chosen_blocks.values())

    # Graficar
    plt.figure(figsize=(10, 5))
    plt.bar(layers, counts, color='skyblue', edgecolor='black')
    plt.xlabel("Layer Index")
    plt.ylabel("Count")
    plt.title("Histogram of Chosen Blocks")
    plt.xticks(layers)  # Asegurar que se marquen todas las capas en el eje X
    plt.grid(axis="y", linestyle="--", alpha=0.7)

    # Mostrar el gráfico
    plt.savefig("block-histogram.pdf")
    plt.show()

def seed_everything(seed):
    """
    Set the random seed for reproducibility.
    """
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(seed)
    random.seed(seed)

def transform_big_integers_to_human_reable(number: int) -> str:
    """
    Print large integers in a human-readable format.
    """
    if number >= 1_000_000_000:
        return f"{number / 1_000_000_000:.2f}B"
    elif number >= 1_000_000:
        return f"{number / 1_000_000:.2f}M"
    elif number >= 1_000:
        return f"{number / 1_000:.2f}K"
    else:
        return number
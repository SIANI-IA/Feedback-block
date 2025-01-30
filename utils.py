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

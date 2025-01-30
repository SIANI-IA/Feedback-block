import tiktoken
import torch

from dataset import create_dataloader_v1
from neural_modules.gpt import GPTModel, FeedbackGPT
from trainer import LanguageModelTrainer
from utils import plot_losses


GPT_CONFIG_124M = {
    "vocab_size": 50257,   # Vocabulary size
    "context_length": 256, # Shortened context length (orig: 1024)
    "emb_dim": 768,        # Embedding dimension
    "n_heads": 12,         # Number of attention heads
    "n_layers": 12,        # Number of layers
    "drop_rate": 0.1,      # Dropout rate
    "qkv_bias": False      # Query-key-value bias
}
SEED = 123
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

torch.manual_seed(SEED)
model = FeedbackGPT(GPT_CONFIG_124M) #GPTModel(GPT_CONFIG_124M)
model.to(device)
optimizer = torch.optim.AdamW(model.parameters(), lr=0.0004, weight_decay=0.1)

file_path = "data/pretrain/the-verdict.txt"
with open(file_path, "r", encoding="utf-8") as file:
        text_data = file.read()

# Train/validation ratio
train_ratio = 0.90
split_idx = int(train_ratio * len(text_data))
train_data = text_data[:split_idx]
val_data = text_data[split_idx:]


torch.manual_seed(123)

train_loader = create_dataloader_v1(
    train_data,
    batch_size=2,
    max_length=GPT_CONFIG_124M["context_length"],
    stride=GPT_CONFIG_124M["context_length"],
    drop_last=True,
    shuffle=True,
    num_workers=0
)

val_loader = create_dataloader_v1(
    val_data,
    batch_size=2,
    max_length=GPT_CONFIG_124M["context_length"],
    stride=GPT_CONFIG_124M["context_length"],
    drop_last=False,
    shuffle=False,
    num_workers=0
)
tokenizer = tiktoken.get_encoding("gpt2")

print("Training data size:", len(train_loader))
print("Validation data size:", len(val_loader))

trainer = LanguageModelTrainer(
    model=model,
    optimizer=optimizer,
    train_loader=train_loader,
    val_loader=val_loader,
    device=device,
    tokenizer=tokenizer,
    start_context="The verdict was",
)

EPOCHS = 15
train_losses, val_losses, tokens_seen = trainer.train(EPOCHS, eval_freq=5, eval_iter=5)

epochs_tensor = torch.linspace(0, EPOCHS, len(train_losses))
plot_losses(epochs_tensor, tokens_seen, train_losses, val_losses)

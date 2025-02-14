import tiktoken
import torch

from dataset import create_dataloader_v1
from neural_modules.gpt import GPTModel, FeedbackGPT, FeedbackGPT_concant, DynamicTransformer, DynamicTransformer2
from trainer import LanguageModelTrainer
from utils import plot_histogram, plot_losses, seed_everything


GPT_CONFIG_124M = {
    "vocab_size": 50257,   # Vocabulary size
    "context_length": 256, # Shortened context length (orig: 1024)
    "emb_dim": 768,        # Embedding dimension
    "n_heads": 12,         # Number of attention heads
    "n_layers": 5,        # Number of layers
    "drop_rate": 0.1,      # Dropout rate
    "qkv_bias": False,     # Query-key-value bias
    "n_iter": 3,           # Number of iterations
    "batch_size": 2,
    "temperature": 2,
}
EPOCHS = 15
SEED = 123
peak_lr = 0.001
weight_decay = 0.1
example_sentence = "The verdict was"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
use_wandb = True
project_name = "test_feedback"
run_name = "feed_2"

seed_everything(SEED)
model = DynamicTransformer(GPT_CONFIG_124M) #GPTModel(GPT_CONFIG_124M)
model.to(device)

optimizer = torch.optim.AdamW(
        model.parameters(), 
        lr=peak_lr, 
        weight_decay=weight_decay
    )

# TODO: extend this to use diferentes datasets
file_path = "data/pretrain/the-verdict.txt" #TODO: trasnform this to tiny dataset
with open(file_path, "r", encoding="utf-8") as file:
        text_data = file.read()

# Train/validation ratio
train_ratio = 0.90
split_idx = int(train_ratio * len(text_data))
train_data = text_data[:split_idx]
val_data = text_data[split_idx:]

train_loader = create_dataloader_v1(
    train_data,
    batch_size=GPT_CONFIG_124M["batch_size"],
    max_length=GPT_CONFIG_124M["context_length"],
    stride=GPT_CONFIG_124M["context_length"],
    drop_last=True,
    shuffle=True,
    num_workers=0
)

val_loader = create_dataloader_v1(
    val_data,
    batch_size=GPT_CONFIG_124M["batch_size"],
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
    start_context=example_sentence,
    use_wandb=use_wandb,
    project_name=project_name,
    run_name=run_name
)

total_steps = len(train_loader) * EPOCHS
warmup_steps = int(0.2 * total_steps) # 20% warmup

model_trained = trainer.train(
        EPOCHS, 
        eval_freq=5, 
        eval_iter=1, 
        warmup_steps=warmup_steps,
        initial_lr=1e-5,
        min_lr=1e-5
)

#TODO: Save the model
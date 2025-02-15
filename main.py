import tiktoken
import torch

from dataset import create_dataloader
from neural_modules.gpt import GPTModel, FeedbackGPT, FeedbackGPT_concant, DynamicTransformer, DynamicTransformer2
from trainer import LanguageModelTrainer
from utils import plot_histogram, plot_losses, seed_everything
from dataset_splitter.dataset_splitter import TxtDatasetSplitter, WikiDatasetSplitter


DATASETS = {
    "tiny": TxtDatasetSplitter("data/pretrain/the-verdict.txt"),
    "wikitext-2": WikiDatasetSplitter("data/pretrain/wikitext-2"),
    "wikitext-103": WikiDatasetSplitter("data/pretrain/wikitext-103"),
}

GPT_CONFIG_124M = {
    "vocab_size": 50257,   # Vocabulary size
    "context_length": 256, # Shortened context length (orig: 1024)
    "emb_dim": 768,        # Embedding dimension
    "n_heads": 12,         # Number of attention heads
    "n_layers": 12,        # Number of layers
    "drop_rate": 0.1,      # Dropout rate
    "qkv_bias": False,     # Query-key-value bias
    "n_iter": 3,           # Number of iterations
    "batch_size": 2,       # Batch size
    "temperature": 2,      # Temperature for selector module
}
EPOCHS = 3
SEED = 123

dataset_name = "tiny"
tokenizer_name = "gpt2"
peak_lr = 0.001
weight_decay = 0.1
example_sentence = "The verdict was"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
use_wandb = False
project_name = "test_feedback"
run_name = "feed_2"
num_workers = 5

seed_everything(SEED)
model = GPTModel(GPT_CONFIG_124M)
model.to(device)

optimizer = torch.optim.AdamW(
        model.parameters(), 
        lr=peak_lr, 
        weight_decay=weight_decay
    )

dataset = DATASETS[dataset_name]
train_data = dataset.get_train_data()
val_data = dataset.get_val_data()

train_loader = create_dataloader(
    train_data,
    batch_size=GPT_CONFIG_124M["batch_size"],
    max_length=GPT_CONFIG_124M["context_length"],
    stride=GPT_CONFIG_124M["context_length"],
    drop_last=True,
    shuffle=True,
    num_workers=num_workers
)

val_loader = create_dataloader(
    val_data,
    batch_size=GPT_CONFIG_124M["batch_size"],
    max_length=GPT_CONFIG_124M["context_length"],
    stride=GPT_CONFIG_124M["context_length"],
    drop_last=False,
    shuffle=False,
    num_workers=num_workers
)
tokenizer = tiktoken.get_encoding(tokenizer_name)

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
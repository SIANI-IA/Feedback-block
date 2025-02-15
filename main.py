import tiktoken
import torch
import os

from dataset import create_dataloader
from neural_modules.gpt import GPTModel, LoopTransformer, SFTFormer
from trainer import LanguageModelTrainer
from utils import seed_everything
from dataset_splitter.dataset_splitter import TxtDatasetSplitter, WikiDatasetSplitter


DATASETS = {
    "tiny": TxtDatasetSplitter("data/pretrain/the-verdict.txt"),
    "wikitext-2": WikiDatasetSplitter("data/pretrain/wikitext-2"),
    "wikitext-103": WikiDatasetSplitter("data/pretrain/wikitext-103"),
}

MODELS = {
    "gpt": GPTModel,
    "loop": LoopTransformer,
    "select": SFTFormer,
}

#Hyperparameters
config = {
    "vocab_size": 50257,   # Vocabulary size
    "context_length": 256, # Shortened context length (orig: 1024)
    "emb_dim": 768,        # Embedding dimension
    "n_heads": 12,         # Number of attention heads
    "n_layers": 12,        # Number of layers
    "drop_rate": 0.1,      # Dropout rate
    "qkv_bias": False,     # Query-key-value bias
    "batch_size": 2,       # Batch size
    # Feedback transformer specific
    "n_iter": 3,           # Number of iterations
    ## Selector module
    "select_dim": 512,     # Selector dimension
    "select_heads": 4,     # Selector heads
    "temperature": 2,      # Temperature for selector module
}
epochs = 3
seed = 123

dataset_name = "tiny"
transformer_type = "select"
tokenizer_name = "gpt2"
peak_lr = 0.001
initial_lr = 1e-5
min_lr = 1e-5
weight_decay = 0.1
example_sentence = "Every effort moves you"
use_wandb = False
project_name = "test_feedback"
folder_to_save = "checkpoints"
run_name = "feed_2"
num_workers = 0
warmup_portion = 0.2 # 20% of total steps
eval_freq = 5
eval_iter = 1
######

assert transformer_type in MODELS, f"Invalid transformer type: {transformer_type}"
assert dataset_name in DATASETS, f"Invalid dataset name: {dataset_name}"
assert num_workers >= 0, "Number of workers must be non-negative"
assert 0 <= warmup_portion <= 1, "Warmup portion must be between 0 and 1"

seed_everything(seed)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = MODELS[transformer_type](config)
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
    batch_size=config["batch_size"],
    max_length=config["context_length"],
    stride=config["context_length"],
    drop_last=True,
    shuffle=True,
    num_workers=num_workers
)

val_loader = create_dataloader(
    val_data,
    batch_size=config["batch_size"],
    max_length=config["context_length"],
    stride=config["context_length"],
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

total_steps = len(train_loader) * epochs
warmup_steps = int(warmup_portion * total_steps) # 20% warmup

model_trained = trainer.train(
        epochs, 
        eval_freq=eval_freq, 
        eval_iter=eval_iter, 
        warmup_steps=warmup_steps,
        initial_lr=initial_lr,
        min_lr=min_lr
)

# Save the model
# create the folder to save the model
folder_to_save = f"{folder_to_save}/{transformer_type}"
if not os.path.exists(folder_to_save):
    os.makedirs(folder_to_save)


torch.save(model_trained.state_dict(), f"{folder_to_save}/{run_name}.pt")
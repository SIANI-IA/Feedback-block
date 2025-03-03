import argparse
import json
import tiktoken
import torch
from distutils.util import strtobool
import os

from char_dataset import create_char_dataloader
from dataset import create_dataloader
from neural_modules.gpt import GPTModel, LoopTransformer
from trainer import LanguageModelTrainer
from utils import get_timestamp, seed_everything

from task.regular.cycle_navigation import CycleNavigation


DATASETS = {
    "cycle_navigation": CycleNavigation,
}

MODELS = {
    "gpt": GPTModel,
    "loop": LoopTransformer,
}

def parse_args():
    parser = argparse.ArgumentParser(description="Train a language model with hyperparameters from CLI.")
    parser.add_argument("--transformer_type", type=str, default="loop", choices=MODELS.keys())
    parser.add_argument("--context_length", type=int, default=13)
    parser.add_argument("--emb_dim", type=int, default=768)
    parser.add_argument("--n_heads", type=int, default=12)
    parser.add_argument("--n_layers", type=int, default=1)
    parser.add_argument("--drop_rate", type=float, default=0.1)
    parser.add_argument("--qkv_bias", type=lambda x: bool(strtobool(x)), default=False)
    parser.add_argument("--batch_size", type=int, default=2)
    # Feedback transformer hyperparameters
    parser.add_argument("--n_iter", type=int, default=6)
    # Task specific hyperparameters
    parser.add_argument("--task_name", type=str, choices=DATASETS.keys())
    parser.add_argument("--sample", type=int, default=100)

    # Training hyperparameters
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--peak_lr", type=float, default=0.001)
    parser.add_argument("--initial_lr", type=float, default=1e-5)
    parser.add_argument("--min_lr", type=float, default=1e-5)
    parser.add_argument("--weight_decay", type=float, default=0.1)
    parser.add_argument("--example_sentence", type=str, default="-110-10=")
    parser.add_argument("--use_wandb", type=lambda x: bool(strtobool(x)), default=False)
    parser.add_argument("--folder_to_save", type=str, default="checkpoints")
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--warmup_portion", type=float, default=0.2)
    parser.add_argument("--eval_freq", type=int, default=5)
    parser.add_argument("--eval_iter", type=int, default=1)
    parser.add_argument("--cosine_annealing", type=lambda x: bool(strtobool(x)), default=True)
    args = parser.parse_args()
    args.sample = int(args.sample)
    args.project_name = f"language-modeling-{args.task_name}"
    time_now = get_timestamp()

    args.run_name = f"{args.transformer_type}_" + \
    f"{args.epochs}epochs_" + \
    f"{args.context_length}context_" + \
    f"{args.n_heads}heads_" + \
    f"{args.n_layers}layers_" + \
    f"{args.emb_dim}emb"

    if args.transformer_type == "loop":
        args.run_name += f"{args.n_iter}iter_"
    if args.transformer_type == "select":
        args.run_name += f"{args.n_iter}iter_"
        args.run_name += f"{args.select_dim}sel_dim_" + \
        f"{args.select_heads}sel_heads_" + \
        f"{args.temperature}temp_"
    args.run_name += f"{time_now}"

    return args

#main
if __name__ == "__main__":
    args = parse_args()

    assert args.transformer_type in MODELS, f"Invalid transformer type: {args.transformer_type}"
    assert args.task_name in DATASETS, f"Invalid dataset name: {args.dataset_name}"
    assert args.num_workers >= 0, "Number of workers must be non-negative"
    assert 0 < args.warmup_portion <= 1, "Warmup portion must be between 0 and 1"

    seed_everything(args.seed)


    task_generator = DATASETS[args.task_name]()
    val_samples = args.sample // 10 + 1
    train_data = task_generator.sample_batch(args.sample, args.context_length)
    val_data = task_generator.sample_batch(val_samples, args.context_length)

    train_loader, tokenizer = create_char_dataloader(
        train_data, 
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers
    )

    val_loader, _ = create_char_dataloader(
        val_data, 
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers
    )

    args.vocab_size = tokenizer.vocab_size()
    print("Vocabulary size:", args.vocab_size)
    print("Training data size:", len(train_loader))
    print("Validation data size:", len(val_loader))

    config = vars(args)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model  = MODELS[args.transformer_type](config)
    model.to(device)

    optimizer = torch.optim.AdamW(
        model.parameters(), 
        lr=args.peak_lr, 
        weight_decay=args.weight_decay
    )

    trainer = LanguageModelTrainer(
        model=model,
        optimizer=optimizer,
        train_loader=train_loader,
        val_loader=val_loader,
        device=device,
        tokenizer=tokenizer,
        start_context=args.example_sentence,
        use_wandb=args.use_wandb,
        project_name=args.project_name,
        run_name=args.run_name
    )

    total_steps  = len(train_loader) * args.epochs
    warmup_steps = int(args.warmup_portion * total_steps) # 20% warmup

    model_trained = trainer.train(
        args.epochs, 
        eval_freq=args.eval_freq, 
        eval_iter=args.eval_iter, 
        warmup_steps=warmup_steps,
        initial_lr=args.initial_lr,
        min_lr=args.min_lr,
        cosine_annealing=args.cosine_annealing
    )

    # Save the model
    # create the folder to save the model
    folder_to_save = os.path.join(
        args.folder_to_save, 
        args.project_name,
        args.transformer_type,
        args.run_name
    )
    if not os.path.exists(folder_to_save):
        os.makedirs(folder_to_save)

    #save config in json file
    with open(f"{folder_to_save}/config.json", "w") as file:
        json.dump(config, file)

    torch.save(model_trained.state_dict(), f"{folder_to_save}/model.pth")
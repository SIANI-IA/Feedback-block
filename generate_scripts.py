import os
import itertools
import numpy as np

# Constants that remain the same for all scripts
MODELS  = ["gpt", "loop"]
LAYERS  = [1, 12]
TASK_NAME = ["cycle_navigation", "even_pairs"]

def get_log_distrubution(num_points: int, start: int, end: int) -> np.ndarray:
    log_seq = np.logspace(np.log10(start), np.log10(end), num=num_points)
    log_seq = np.round(log_seq).astype(int)
    return log_seq

CONTEXT_LENGTH = get_log_distrubution(10, 5, 100).tolist()

parms = {
    "context_length": 68,
    "transformer_type": "gpt",
    "emb_dim": 768,
    "n_heads": 12,
    "n_layers": 12,
    "drop_rate": 0.1,
    "n_iter": 0,
    "epochs": 10,
    "peak_lr": 0.001,
    "weight_decay": 0.1,
    "batch_size": 68,
    "use_wandb": True,
    "folder_to_save": "checkpoints",
    "num_workers": 0,
    "warmup_portion": 0.2,
    "eval_freq": 5,
    "eval_iter": 1,
    "sample": 1000,
}

def define_scripts(config: dict) -> str:
    bash_script = f"""#!/bin/bash
# Dataset
task_name="{config['task_name']}"
context_length={config['context_length']}
sample={config['sample']}

# transformer architecture
transformer_type="{config['transformer_type']}"
emb_dim={config['emb_dim']}
n_heads={config['n_heads']}
n_layers={config['n_layers']}
drop_rate={config['drop_rate']}
## Feedback hyperparameters
n_iter={config['n_iter']}

# training hyperparameters
epochs={config['epochs']}
peak_lr={config['peak_lr']}
weight_decay={config['weight_decay']}
batch_size={config['batch_size']}
use_wandb={config['use_wandb']}
folder_to_save={config['folder_to_save']}
num_workers={config['num_workers']}
warmup_portion={config['warmup_portion']}
eval_freq={config['eval_freq']}
eval_iter={config['eval_iter']}

python3 task_experiments.py \
    --sample $sample \
    --context_length $context_length \
    --emb_dim $emb_dim \
    --n_heads $n_heads \
    --n_layers $n_layers \
    --drop_rate $drop_rate \
    --batch_size $batch_size \
    --n_iter $n_iter \
    --epochs $epochs \
    --task_name $task_name \
    --transformer_type $transformer_type \
    --peak_lr $peak_lr \
    --weight_decay $weight_decay \
    --use_wandb $use_wandb \
    --folder_to_save $folder_to_save \
    --num_workers $num_workers \
    --warmup_portion $warmup_portion \
    --eval_freq $eval_freq \
    --eval_iter $eval_iter
            """
    return bash_script

def save_script(script_filename: str, bash_script: str) -> None:
    # Save the script
    with open(script_filename, "w") as file:
        file.write(bash_script)

    # Make the script executable
    os.chmod(script_filename, 0o755)


# Create an output directory for the scripts
output_dir = "scripts"
os.makedirs(output_dir, exist_ok=True)
iterations = itertools.product(LAYERS, MODELS, CONTEXT_LENGTH, TASK_NAME)

# Generate a script for each combination of model, PEFT type, and dataset
for i, (layer, model, context, task) in enumerate(iterations):
    # Define the script
    script_filename  = os.path.join(output_dir, f"train_{i+1}.sh")
    parms["task_name"] = task
    parms["context_length"] = context
    parms["transformer_type"] = model
    if model == "gpt":
        parms["n_layers"] = layer
    else:
        if layer == 1:
            continue
        parms["n_layers"] = 1
        parms["n_iter"] = layer
    bash_script = define_scripts(parms)

    # Save the script
    with open(script_filename, "w") as file:
        file.write(bash_script)

    # Make the script executable
    os.chmod(script_filename, 0o755)

    print(f"✅ Script '{script_filename}' generated.")

print("\n🚀 All scripts have been generated in the 'generated_scripts' folder.")
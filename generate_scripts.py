import os
import itertools

# Constants that remain the same for all scripts
EMB_DIM = [32, 64]
N_HEADS = [1, 2, 4, 8]
LAYERS  = [1, 3, 6, 12]

parms = {
    "context_length": 256,
    "dataset_name":"wikitext-103",
    "vocab_size": 50257,
    "transformer_type": "gpt",
    "emb_dim": 768,
    "n_heads": 12,
    "n_layers": 12,
    "drop_rate": 0.1,
    "n_iter": 0,
    "select_dim": 0,
    "select_heads": 0,
    "temperature": 0,
    "epochs": 1,
    "peak_lr": 0.001,
    "weight_decay": 0.1,
    "batch_size": 68,
    "use_wandb": True,
    "folder_to_save": "checkpoints",
    "num_workers": 5,
    "warmup_portion": 0.2,
    "eval_freq": 5,
    "eval_iter": 1
}

def define_scripts(config: dict) -> str:
    bash_script = f"""#!/bin/bash
# Dataset
dataset_name="{config['dataset_name']}"
context_length={config['context_length']}
vocab_size={config['vocab_size']}

# transformer architecture
transformer_type="{config['transformer_type']}"
emb_dim={config['emb_dim']}
n_heads={config['n_heads']}
n_layers={config['n_layers']}
drop_rate={config['drop_rate']}
## Feedback hyperparameters
n_iter={config['n_iter']}
## SFTransformer hyperparameters
select_dim={config['select_dim']}
select_heads={config['select_heads']}
temperature={config['temperature']}

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

python3 main.py \
    --vocab_size $vocab_size \
    --context_length $context_length \
    --emb_dim $emb_dim \
    --n_heads $n_heads \
    --n_layers $n_layers \
    --drop_rate $drop_rate \
    --batch_size $batch_size \
    --n_iter $n_iter \
    --select_dim $select_dim \
    --select_heads $select_heads \
    --temperature $temperature \
    --epochs $epochs \
    --dataset_name $dataset_name \
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

# Generate a script for each combination of model, PEFT type, and dataset
for i, (dim, head, layer) in enumerate(itertools.product(EMB_DIM, N_HEADS, LAYERS)):
    # Define the script
    script_filename  = os.path.join(output_dir, f"train_{i+1}.sh")
    parms["emb_dim"] = dim
    parms["n_heads"] = head
    parms["n_layers"] = layer
    bash_script = define_scripts(parms)

    # Save the script
    with open(script_filename, "w") as file:
        file.write(bash_script)

    # Make the script executable
    os.chmod(script_filename, 0o755)

    print(f"✅ Script '{script_filename}' generated.")

print("\n🚀 All scripts have been generated in the 'generated_scripts' folder.")
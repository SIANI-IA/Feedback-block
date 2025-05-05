from datasets import load_dataset
import os

datasets = {
    "wikitext-2": {
        "name_repo": "Salesforce/wikitext",
        "name_dataset": "wikitext-2-v1",
    },
    "wikitext-103": {
        "name_repo": "Salesforce/wikitext",
        "name_dataset": "wikitext-103-v1",
    }
}

name_dataset = "wikitext-103"
assert name_dataset in datasets, f"Dataset {name_dataset} not found"
folder_cache = os.path.join("data", "pretrain", "cache")
folder_data = os.path.join("data", "pretrain", name_dataset)
num_proc = 16

ds = load_dataset(
    datasets[name_dataset]["name_repo"], 
    datasets[name_dataset]["name_dataset"],
    num_proc=num_proc,
    cache_dir=folder_cache,
)

print(ds)

# save the dataset
ds.save_to_disk(folder_data)

# delete the cache is a folder
import shutil
shutil.rmtree(folder_cache)






from datasets import load_dataset
import os

# Cargar el dataset
ds = load_dataset("HuggingFaceTB/smollm-corpus", "cosmopedia-v2", split="train", num_proc=16)

# Imprimir el primer elemento del dataset
print(ds[0])

# Crear el directorio donde se guardará el dataset
ds_dir = "data/pretrain/cosmopedia-v2"
os.makedirs(ds_dir, exist_ok=True)

# Guardar el dataset en formato JSON o Parquet
ds.save_to_disk(ds_dir)







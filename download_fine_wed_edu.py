from datasets import load_dataset

ds = load_dataset("HuggingFaceTB/smollm-corpus", "fineweb-edu-dedup")

print(ds[0])

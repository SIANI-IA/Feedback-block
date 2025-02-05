from datasets import load_dataset

ds = load_dataset("HuggingFaceTB/smollm-corpus", "fineweb-edu-dedup", split="train", num_proc=16)

print(ds[0])

from datasets import load_dataset

# Download and cache the dataset
# dataset_name = "allenai/llama-3.1-tulu-3-8b-preference-mixture"
# dataset_name = "ezosa/tulu-3-sft-mixture-commercial" # Filtered SFT dataset
dataset_name = "allenai/Dolci-Instruct-SFT"
print(f"Downloading {dataset_name}...")

dataset = load_dataset(dataset_name, split="train")
print(f"Dataset downloaded successfully!")
print(f"Number of examples: {len(dataset)}")
print(f"Dataset cached at: {dataset.cache_files}")

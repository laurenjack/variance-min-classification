from datasets import load_dataset
from transformers import AutoTokenizer
import torch
import torch.nn as nn
from torch.utils.data import DataLoader


def download_data(c):
    # Load the dataset (download if not already cached)
    dataset = load_dataset(c.dataset_name, cache_dir=c.cache_dir)
    train_data = dataset["train"]
    val_data   = dataset.get("test", dataset.get("validation", None))  # use 'test' split as validation if available

    # Initialize tokenizer
    tokenizer = AutoTokenizer.from_pretrained(c.model_name, cache_dir=c.cache_dir)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token  # ensure pad_token is defined

    # Preprocessing: tokenize prompt+chosen and prompt+rejected for each example
    def tokenize_pair(example):
        prompt = example["prompt"]
        chosen = example["chosen"]
        rejected = example["rejected"]
        # Concatenate prompt with each response; include EOS token to mark end of response
        chosen_text = prompt + " " + chosen + (tokenizer.eos_token or "")
        rejected_text = prompt + " " + rejected + (tokenizer.eos_token or "")
        chosen_enc = tokenizer(chosen_text, max_length=c.max_length, padding=False, truncation=True)
        rejected_enc = tokenizer(rejected_text, max_length=c.max_length, padding=False, truncation=True)
        return {
            "chosen_ids": chosen_enc["input_ids"],
            "chosen_attention_mask": chosen_enc["attention_mask"],
            "rejected_ids": rejected_enc["input_ids"],
            "rejected_attention_mask": rejected_enc["attention_mask"],
        }

    # Apply tokenization to the entire dataset (this will cache results for reuse)
    train_data = train_data.map(tokenize_pair, batched=False, remove_columns=train_data.column_names)
    if val_data:
        val_data = val_data.map(tokenize_pair, batched=False, remove_columns=val_data.column_names)

    # Create PyTorch DataLoaders with padding in collate function
    from torch.utils.data import DataLoader

    def collate_fn(batch):
        # Batch is a list of examples (dicts) with keys: chosen_ids, chosen_attention_mask, rejected_ids, rejected_attention_mask
        # We need to pad chosen and rejected sequences separately and then combine.
        chosen_list = [torch.tensor(ex["chosen_ids"], dtype=torch.long) for ex in batch]
        reject_list = [torch.tensor(ex["rejected_ids"], dtype=torch.long) for ex in batch]
        # Pad sequences
        chosen_padded = nn.utils.rnn.pad_sequence(chosen_list, batch_first=True, padding_value=tokenizer.pad_token_id)
        reject_padded = nn.utils.rnn.pad_sequence(reject_list, batch_first=True, padding_value=tokenizer.pad_token_id)
        # Attention masks (pad with 0s accordingly)
        chosen_mask = nn.utils.rnn.pad_sequence([torch.tensor(ex["chosen_attention_mask"], dtype=torch.long) for ex in batch],
                                            batch_first=True, padding_value=0)
        reject_mask = nn.utils.rnn.pad_sequence([torch.tensor(ex["rejected_attention_mask"], dtype=torch.long) for ex in batch],
                                            batch_first=True, padding_value=0)
        # Stack chosen and rejected together into one batch dimension of size 2*batch_size for efficient forward pass
        input_ids = torch.cat([chosen_padded, reject_padded], dim=0)
        attention_mask = torch.cat([chosen_mask, reject_mask], dim=0)
        return input_ids, attention_mask

    train_loader = DataLoader(train_data, batch_size=c.train_batch_size, shuffle=True, collate_fn=collate_fn)
    val_loader = DataLoader(val_data, batch_size=c.eval_batch_size, shuffle=False, collate_fn=collate_fn) if val_data else None

    print(f"Loaded dataset with {len(train_data)} training examples and {len(val_data) if val_data else 0} validation examples.")
    return train_loader, val_loader

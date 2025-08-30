import torch
from learned_dropout.llm.data import prepare_data
from learned_dropout.llm.model import create_model
from learned_dropout.llm.train import train_model

# Hyperparameters and settings
model_size = "tiny"         # choose from "tiny", "mini", "small", "medium", "large", "xl"
seq_len = 64                 # sequence length (context window)
batch_size = 32               # sequences per batch (micro-batch)
grad_accum_steps = 1         # gradient accumulation steps to form an effective batch
learning_rate = 1e-4         # initial learning rate for AdamW optimizer
num_train_steps = 1000       # total training steps
eval_interval = 100          # evaluate on validation set every 100 steps
subset = "sample-10BT"       # subset of FineWeb to use (use smaller sample for demo)

# Prepare data
print("Loading data...")
tokenizer, train_seq_gen, val_sequences = prepare_data(seq_len=seq_len, num_val_sequences=1000, subset=subset)

# Create model
print(f"Initializing a {model_size} GPT-2 style model...")
model = create_model(model_size=model_size, vocab_size=len(tokenizer), seq_len=seq_len)

# Train model
train_model(model, train_seq_gen, val_sequences, tokenizer, 
            num_train_steps=num_train_steps, batch_size=batch_size, grad_accum_steps=grad_accum_steps,
            learning_rate=learning_rate, device=("cuda" if torch.cuda.is_available() else "cpu"), 
            eval_interval=eval_interval)

# (Optional) Save the trained model
model.save_pretrained(f"fineweb-{model_size}-model")
print("Model training and evaluation done.")

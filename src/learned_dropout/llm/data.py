from datasets import load_dataset
from transformers import GPT2TokenizerFast

def load_fineweb_dataset(subset="sample-10BT"):
    """
    Load the FineWeb dataset in streaming mode. 
    `subset` can be 'default' for full data or one of 'sample-350BT', 'sample-100BT', 'sample-10BT' etc.:contentReference[oaicite:4]{index=4}.
    """
    return load_dataset("HuggingFaceFW/fineweb", name=subset, split="train", streaming=True)

def get_tokenizer():
    """
    Initialize a GPT-2 tokenizer. We use GPT-2's byte-level BPE tokenizer with the standard vocabulary (50257 tokens).
    """
    tokenizer = GPT2TokenizerFast.from_pretrained("gpt2")
    # Use the end-of-text token as padding (though we will mostly avoid padding by using fixed-length sequences)
    tokenizer.pad_token = tokenizer.eos_token
    return tokenizer

def sequence_generator(dataset_iter, tokenizer, seq_len):
    """
    Generate sequences of length `seq_len` tokens from an iterable of documents.
    Concatenates documents with an EOS token and yields contiguous token sequences.
    """
    eos_id = tokenizer.eos_token_id
    buffer = []
    for doc in dataset_iter:
        text = doc["text"]
        # Tokenize the document text to token IDs
        tokens = tokenizer.encode(text)
        # Append an EOS token at the end of each document
        tokens.append(eos_id)
        # Extend the buffer with the new tokens
        buffer.extend(tokens)
        # Yield sequences from the buffer
        while len(buffer) >= seq_len:
            # Take the first seq_len tokens as one sequence
            seq = buffer[:seq_len]
            # Remove them from the buffer
            buffer = buffer[seq_len:]
            yield seq  # yields a list of token IDs of length seq_len
    # (We drop any remaining tokens in the buffer smaller than seq_len to avoid partial sequences)
    
def prepare_data(seq_len=64, num_val_sequences=1000, subset="sample-10BT"):
    """
    Load dataset and tokenizer, and prepare training data generator and validation data.
    Returns (tokenizer, train_generator, val_sequences).
    """
    # Load dataset in streaming mode
    dataset = load_fineweb_dataset(subset=subset)
    tokenizer = get_tokenizer()
    # Create an iterator over the dataset
    dataset_iter = iter(dataset)
    # Initialize the sequence generator
    seq_gen = sequence_generator(dataset_iter, tokenizer, seq_len)
    # Collect a fixed number of validation sequences from the generator
    val_sequences = []
    for _ in range(num_val_sequences):
        try:
            seq = next(seq_gen)
            val_sequences.append(seq)
        except StopIteration:
            break
    # At this point, `val_sequences` contains num_val_sequences token sequences (for validation),
    # and the seq_gen generator is poised to continue yielding training sequences from where it left off.
    return tokenizer, seq_gen, val_sequences

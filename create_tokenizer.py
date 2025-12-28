"""
Modal script for creating the SQL tokenizer.

This trains a tiny 512-token SentencePiece BPE tokenizer specialized for SQL syntax.
The small vocabulary keeps the embedding table small for tiny models.

Usage:
    # Train tokenizer from data on the volume and save to volume
    modal run create_tokenizer.py::train_tokenizer

    # Download the trained tokenizer locally
    modal run create_tokenizer.py::download_tokenizer_local

    # Test the tokenizer on a sample
    modal run create_tokenizer.py::test_tokenizer
"""

import modal

app = modal.App("sqlpilot-tokenizer")

# Use the same data volume as training
data_volume = modal.Volume.from_name("sqlpilot-data", create_if_missing=True)

DATA_DIR = "/data"
RAW_DATA_SUBDIR = "raw"
TOKENIZER_SUBDIR = "tokenizer"

image = (
    modal.Image.debian_slim(python_version="3.11")
    .uv_pip_install(
        "tokenizers",
        "sentencepiece",
    )
)

VOCAB_SIZE = 512


@app.function(
    image=image,
    volumes={DATA_DIR: data_volume},
    timeout=30 * 60,  # 30 min
)
def train_tokenizer():
    """Train a BPE tokenizer on the raw training data."""
    import os
    import json
    from tokenizers import SentencePieceBPETokenizer
    
    raw_data_path = os.path.join(DATA_DIR, RAW_DATA_SUBDIR)
    tokenizer_path = os.path.join(DATA_DIR, TOKENIZER_SUBDIR)
    
    # Check for raw data
    if not os.path.exists(raw_data_path) or not os.listdir(raw_data_path):
        raise RuntimeError(
            f"No raw data found at {raw_data_path}. "
            "Run 'modal run modal_train.py::upload_data_local' first"
        )
    
    def load_json_files():
        files = sorted(os.listdir(raw_data_path))
        for filename in files:
            if filename.endswith(".json"):
                file_path = os.path.join(raw_data_path, filename)
                with open(file_path, 'r', encoding='utf-8') as file:
                    data = json.load(file)
                    for example in data:
                        # Split into parts to help tokenizer learn structure
                        example = example.replace('### TABLEDATA', '')
                        parts = example.split('### STATEMENT')
                        parts = [part.strip() for part in parts]
                        yield '### TABLEDATA'
                        yield parts[0]
                        yield '### STATEMENT'
                        if len(parts) > 1:
                            yield parts[1]
    
    print("Creating tokenizer...")
    tokenizer = SentencePieceBPETokenizer()
    
    # Define special tokens
    special_tokens = [
        "<unk>",
        "<pad>",
        "<s>",
        "</s>",
    ]
    tokenizer.add_special_tokens(special_tokens)
    
    print(f"Training tokenizer with vocab_size={VOCAB_SIZE}...")
    tokenizer.train_from_iterator(
        load_json_files(),
        vocab_size=VOCAB_SIZE - len(special_tokens)
    )
    
    # Save the tokenizer
    os.makedirs(tokenizer_path, exist_ok=True)
    output_file = os.path.join(tokenizer_path, "tokenizer.json")
    tokenizer.save(output_file)
    
    print(f"Tokenizer saved to {output_file}")
    
    # Commit to persist
    data_volume.commit()
    print("Tokenizer committed to volume!")
    
    # Test it
    from tokenizers import Tokenizer
    loaded = Tokenizer.from_file(output_file)
    sample = "SELECT * FROM users WHERE id = 1"
    encoded = loaded.encode(sample)
    print(f"\nTest encoding: '{sample}'")
    print(f"Tokens: {encoded.tokens}")
    print(f"Token count: {len(encoded.tokens)}")


@app.function(
    image=image,
    volumes={DATA_DIR: data_volume},
    timeout=60,
)
def get_tokenizer_file() -> bytes:
    """Get the tokenizer file contents."""
    import os
    
    tokenizer_file = os.path.join(DATA_DIR, TOKENIZER_SUBDIR, "tokenizer.json")
    
    if not os.path.exists(tokenizer_file):
        raise ValueError(
            f"Tokenizer not found at {tokenizer_file}. "
            "Run 'modal run create_tokenizer.py::train_tokenizer' first"
        )
    
    with open(tokenizer_file, 'rb') as f:
        return f.read()


@app.local_entrypoint()
def download_tokenizer_local(output_dir: str = "tokenizer"):
    """Download the trained tokenizer locally."""
    import os
    
    content = get_tokenizer_file.remote()
    
    os.makedirs(output_dir, exist_ok=True)
    output_file = os.path.join(output_dir, "tokenizer.json")
    
    with open(output_file, 'wb') as f:
        f.write(content)
    
    print(f"Tokenizer saved to {output_file}")


@app.function(
    image=image,
    volumes={DATA_DIR: data_volume},
    timeout=60,
)
def test_tokenizer():
    """Test the tokenizer on sample SQL."""
    import os
    from tokenizers import Tokenizer
    
    tokenizer_file = os.path.join(DATA_DIR, TOKENIZER_SUBDIR, "tokenizer.json")
    
    if not os.path.exists(tokenizer_file):
        raise ValueError("Tokenizer not found. Run train_tokenizer first.")
    
    tokenizer = Tokenizer.from_file(tokenizer_file)
    
    samples = [
        "SELECT * FROM users WHERE id = 1",
        "INSERT INTO orders (user_id, total) VALUES (1, 99.99)",
        "CREATE TABLE products (id INT PRIMARY KEY, name VARCHAR(255))",
        "### TABLEDATA",
        "### STATEMENT",
    ]
    
    print(f"Vocab size: {tokenizer.get_vocab_size()}")
    print()
    
    for sample in samples:
        encoded = tokenizer.encode(sample)
        print(f"Input: {sample}")
        print(f"Tokens ({len(encoded.tokens)}): {encoded.tokens}")
        print()


if __name__ == "__main__":
    print("Use 'modal run create_tokenizer.py::train_tokenizer' to train the tokenizer")


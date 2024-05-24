import os
import json
from tokenizers import SentencePieceBPETokenizer, Tokenizer
from tokenizers.pre_tokenizers import Split

directory = '../data/diverse_training_data'
tokenizer_dir = 'smol_tokenizer'

VOCAB_SIZE = 512

def load_json_files():
    for filename in os.listdir(directory):
        if filename.endswith(".json"):
            file_path = os.path.join(directory, filename)
            with open(file_path, 'r', encoding='utf-8') as file:
                data = json.load(file)
                for example in data:
                    example = example.replace('### TABLEDATA', '')
                    parts = example.split('### STATEMENT')
                    parts = [part.strip() for part in parts]
                    yield '### TABLEDATA'
                    yield parts[0]
                    yield '### STATEMENT'
                    yield parts[1]

# Create a new SentencePiece BPE tokenizer
tokenizer = SentencePieceBPETokenizer()

# Define special tokens
special_tokens = [
    # terminators
    "<unk>",
    "<pad>",
    "<s>",
    "</s>",
]
tokenizer.add_special_tokens(special_tokens)

# Train the tokenizer on your data
tokenizer.train_from_iterator(load_json_files(), vocab_size=VOCAB_SIZE - len(special_tokens))

# Create the tokenizer directory if it doesn't exist
os.makedirs(tokenizer_dir, exist_ok=True)

# Save the trained tokenizer
tokenizer_path = os.path.join(tokenizer_dir, "tokenizer.json")
tokenizer.save(tokenizer_path)

# Load the trained tokenizer
tokenizer = Tokenizer.from_file(tokenizer_path)

# Tokenize a sample sentence
encoded = tokenizer.encode("""
### TABLEDATA

CREATE TABLE sendy_stats (
    powerful UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    ind INT,
    forreal VARCHAR(50) NOT NULL,
    eggplant JSONB
);

### STATEMENT

SELECT hi
""")

# Print the tokenized output
print(encoded.tokens)
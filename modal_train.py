"""
Modal training script for SQLPilot.

Volumes:
- sqlpilot-checkpoints: Stores training checkpoints (persistent)
- sqlpilot-data: Stores training data and tokenized datasets

Usage:
    # First time: upload training data to the volume
    modal run modal_train.py::upload_data_local

    # Tokenize the data (run once, or after uploading new data)
    modal run modal_train.py::tokenize

    # Train (will auto-resume from latest checkpoint)
    modal run modal_train.py::train

    # Train specific model size (260K, 15M, or 42M)
    modal run modal_train.py::train --model-size 42M

    # List available checkpoints
    modal run modal_train.py::list_checkpoints

    # Download checkpoint locally
    modal run modal_train.py::download_checkpoint_local --model-size 15M --checkpoint final

    # Evaluate a trained model
    modal run modal_train.py::evaluate --model-size 15M
"""

import modal

# Create the Modal app
app = modal.App("sqlpilot")

# Persistent volumes
checkpoint_volume = modal.Volume.from_name("sqlpilot-checkpoints", create_if_missing=True)
data_volume = modal.Volume.from_name("sqlpilot-data", create_if_missing=True)

CHECKPOINT_DIR = "/checkpoints"
DATA_DIR = "/data"
TOKENIZER_DIR = "/tokenizer"

# Paths within DATA_DIR
RAW_DATA_SUBDIR = "raw"  # JSON files go here
TOKENIZED_DATA_SUBDIR = "tokenized"  # Pre-tokenized dataset

# Container image with all dependencies
image = (
    modal.Image.debian_slim(python_version="3.11")
    .uv_pip_install(
        "torch",
        "transformers>=4.39.0",
        "datasets",
        "accelerate",
        "wandb",
        "tokenizers",
        "sentencepiece",
    )
    .add_local_file(
        "hugtrain/smol_tokenizer/tokenizer.json",
        "/tokenizer/tokenizer.json",
    )
)


# Model configurations
MODEL_CONFIGS = {
    "260K": {
        "hidden_size": 64,
        "num_hidden_layers": 5,
        "num_attention_heads": 8,
        "num_key_value_heads": 4,
        "intermediate_size": 4 * 64,
        "max_position_embeddings": 512,
    },
    "15M": {
        "hidden_size": 288,
        "num_hidden_layers": 6,
        "num_attention_heads": 6,
        "num_key_value_heads": 6,
        "intermediate_size": 4 * 288,
        "max_position_embeddings": 512,
    },
    "42M": {
        "hidden_size": 512,
        "num_hidden_layers": 8,
        "num_attention_heads": 8,
        "num_key_value_heads": 8,
        "intermediate_size": 4 * 512,
        "max_position_embeddings": 512,
    },
}


def upload_data():
    """Upload training data from local directory to Modal volume using batch_upload."""
    import os
    
    local_data_dir = "data/diverse_training_data"
    
    if not os.path.exists(local_data_dir):
        print(f"Local data directory not found: {local_data_dir}")
        print("Run this from the project root directory")
        return
    
    files = sorted([f for f in os.listdir(local_data_dir) if f.endswith(".json")])
    print(f"Found {len(files)} JSON files to upload")
    
    # Use Modal's batch_upload to upload files directly to the volume
    with data_volume.batch_upload() as batch:
        for filename in files:
            local_path = os.path.join(local_data_dir, filename)
            remote_path = f"/{RAW_DATA_SUBDIR}/{filename}"  # Put in raw/ subdirectory
            batch.put_file(local_path, remote_path)
            print(f"Queued {filename}")
    
    print(f"Uploaded {len(files)} files to sqlpilot-data volume")
    print(f"Now run: modal run modal_train.py::tokenize")


@app.local_entrypoint()
def upload_data_local():
    """Local entrypoint to upload training data to Modal volume."""
    upload_data()


@app.function(
    image=image,
    volumes={DATA_DIR: data_volume},
    timeout=60 * 60,  # 1 hour max for tokenization
    cpu=4,
)
def tokenize():
    """Tokenize raw JSON data and save to volume for fast training."""
    import os
    import json
    from transformers import LlamaTokenizerFast
    from datasets import Dataset
    
    raw_data_path = os.path.join(DATA_DIR, RAW_DATA_SUBDIR)
    tokenized_data_path = os.path.join(DATA_DIR, TOKENIZED_DATA_SUBDIR)
    
    # Check for raw data
    if not os.path.exists(raw_data_path) or not os.listdir(raw_data_path):
        raise RuntimeError(
            f"No raw data found at {raw_data_path}. "
            "Run 'modal run modal_train.py::upload_data_local' first"
        )
    
    print(f"Loading tokenizer from {TOKENIZER_DIR}")
    tokenizer = LlamaTokenizerFast(
        tokenizer_file=f"{TOKENIZER_DIR}/tokenizer.json",
        unk_token="<unk>",
        pad_token="<pad>",
        bos_token="<s>",
        eos_token="</s>",
        padding_side="right",
    )
    
    # Load raw JSON files
    def load_json_files():
        files = sorted(os.listdir(raw_data_path))
        for filename in files:
            if filename.endswith(".json"):
                file_path = os.path.join(raw_data_path, filename)
                with open(file_path, 'r', encoding='utf-8') as file:
                    data = json.load(file)
                    for example in data:
                        yield {'text': example}
    
    print("Loading raw dataset...")
    dataset = Dataset.from_generator(load_json_files)
    print(f"Loaded {len(dataset)} examples")
    
    # Split into train/test
    print("Splitting into train/test...")
    dataset = dataset.train_test_split(test_size=0.1, seed=42)
    
    # Tokenize
    def tokenize_function(examples):
        inputs = tokenizer(examples["text"], padding="max_length", truncation=True, max_length=512)
        labels = inputs["input_ids"].copy()
        inputs["labels"] = labels
        return inputs
    
    print("Tokenizing dataset...")
    tokenized_dataset = dataset.map(
        tokenize_function,
        batched=True,
        num_proc=4,
        remove_columns=["text"],
        desc="Tokenizing",
    )
    
    print(f"Train samples: {len(tokenized_dataset['train'])}")
    print(f"Test samples: {len(tokenized_dataset['test'])}")
    
    # Save to volume
    print(f"Saving tokenized dataset to {tokenized_data_path}...")
    tokenized_dataset.save_to_disk(tokenized_data_path)
    
    # Commit to persist
    data_volume.commit()
    print("Tokenized dataset saved and committed!")
    print(f"Ready to train: modal run modal_train.py::train")


@app.function(
    image=image,
    gpu="H100",
    volumes={
        CHECKPOINT_DIR: checkpoint_volume,
        DATA_DIR: data_volume,
    },
    timeout=6 * 60 * 60,  # 6 hours max
    secrets=[modal.Secret.from_name("wandb-secret")],
)
def train(
    model_size: str = "15M",
    num_epochs: int = 3,
    batch_size: int = 512,  # H100 has 80GB VRAM, can handle large batches
    learning_rate: float = 1e-3,
    gradient_accumulation_steps: int = 1,
    save_steps: int = 200,
    eval_steps: int = 200,
    resume: bool = True,
):
    """Train the SQL autocomplete model on Modal with persistent checkpoints."""
    import os
    import re
    import torch
    from transformers import (
        LlamaForCausalLM,
        LlamaConfig,
        TrainingArguments,
        Trainer,
        default_data_collator,
    )
    from datasets import load_from_disk

    print(f"Training {model_size} model")
    print(f"GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'}")
    
    tokenized_data_path = os.path.join(DATA_DIR, TOKENIZED_DATA_SUBDIR)
    
    # Check for tokenized data
    if not os.path.exists(tokenized_data_path):
        raise RuntimeError(
            f"No tokenized data found at {tokenized_data_path}. "
            "Run 'modal run modal_train.py::tokenize' first"
        )
    
    # Load model config
    if model_size not in MODEL_CONFIGS:
        raise ValueError(f"Unknown model size: {model_size}. Choose from: {list(MODEL_CONFIGS.keys())}")
    
    model_config = MODEL_CONFIGS[model_size]
    
    # Load pre-tokenized dataset
    print(f"Loading tokenized dataset from {tokenized_data_path}...")
    tokenized_dataset = load_from_disk(tokenized_data_path)
    
    print(f"Train samples: {len(tokenized_dataset['train'])}")
    print(f"Test samples: {len(tokenized_dataset['test'])}")
    
    # Check for existing checkpoint
    checkpoint_path = None
    model_checkpoint_dir = os.path.join(CHECKPOINT_DIR, model_size)
    os.makedirs(model_checkpoint_dir, exist_ok=True)
    
    if resume and os.path.isdir(model_checkpoint_dir):
        checkpoints = [d for d in os.listdir(model_checkpoint_dir) if d.startswith("checkpoint-")]
        if checkpoints:
            latest = max(checkpoints, key=lambda x: int(re.findall(r'\d+', x)[0]))
            checkpoint_path = os.path.join(model_checkpoint_dir, latest)
            print(f"Resuming from checkpoint: {checkpoint_path}")
    
    # Initialize model
    if checkpoint_path:
        print(f"Loading model from checkpoint: {checkpoint_path}")
        model = LlamaForCausalLM.from_pretrained(checkpoint_path)
    else:
        print(f"Initializing new {model_size} model")
        model = LlamaForCausalLM(LlamaConfig(
            **model_config,
            vocab_size=512,
        ))
    
    print(f"Model parameters: {model.num_parameters():,}")
    
    # Training arguments
    training_args = TrainingArguments(
        output_dir=model_checkpoint_dir,
        num_train_epochs=num_epochs,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        warmup_steps=500,
        weight_decay=0.01,
        logging_dir=os.path.join(model_checkpoint_dir, "logs"),
        logging_steps=10,
        gradient_accumulation_steps=gradient_accumulation_steps,
        learning_rate=learning_rate,
        save_strategy="steps",
        save_steps=save_steps,
        save_total_limit=5,  # Keep last 5 checkpoints
        eval_strategy="steps",
        eval_steps=eval_steps,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        report_to="wandb" if os.environ.get("WANDB_API_KEY") else "none",
        run_name=f"sqlpilot-{model_size}",
        bf16=True,
        torch_compile=True,
        dataloader_num_workers=4,
    )
    
    # Create trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset["train"],
        eval_dataset=tokenized_dataset["test"],
        data_collator=default_data_collator,
    )
    
    # Train
    print("Starting training...")
    trainer.train(resume_from_checkpoint=checkpoint_path)
    
    # Save final model
    final_model_path = os.path.join(model_checkpoint_dir, "final")
    trainer.save_model(final_model_path)
    print(f"Saved final model to {final_model_path}")
    
    # Commit volume to persist checkpoints
    checkpoint_volume.commit()
    print("Checkpoints committed to volume")
    
    # Evaluate
    eval_results = trainer.evaluate()
    print(f"Final eval loss: {eval_results['eval_loss']:.4f}")
    
    return eval_results


@app.function(
    image=image,
    volumes={CHECKPOINT_DIR: checkpoint_volume},
    timeout=300,
)
def list_checkpoints():
    """List all available checkpoints."""
    import os
    
    if not os.path.exists(CHECKPOINT_DIR):
        print("No checkpoints found")
        return
    
    for model_size in os.listdir(CHECKPOINT_DIR):
        model_dir = os.path.join(CHECKPOINT_DIR, model_size)
        if os.path.isdir(model_dir):
            checkpoints = [d for d in os.listdir(model_dir) if d.startswith("checkpoint-") or d == "final"]
            print(f"\n{model_size}:")
            for cp in sorted(checkpoints):
                cp_path = os.path.join(model_dir, cp)
                size = sum(
                    os.path.getsize(os.path.join(cp_path, f))
                    for f in os.listdir(cp_path)
                    if os.path.isfile(os.path.join(cp_path, f))
                ) / (1024 * 1024)
                print(f"  - {cp} ({size:.1f} MB)")


@app.function(
    image=image,
    volumes={CHECKPOINT_DIR: checkpoint_volume},
    timeout=600,
)
def download_checkpoint(model_size: str = "15M", checkpoint: str = "final") -> dict:
    """Download a checkpoint. Returns the model files as a dict."""
    import os
    
    checkpoint_path = os.path.join(CHECKPOINT_DIR, model_size, checkpoint)
    
    if not os.path.exists(checkpoint_path):
        raise ValueError(f"Checkpoint not found: {checkpoint_path}")
    
    files = {}
    for filename in os.listdir(checkpoint_path):
        filepath = os.path.join(checkpoint_path, filename)
        if os.path.isfile(filepath):
            with open(filepath, 'rb') as f:
                files[filename] = f.read()
    
    return files


@app.local_entrypoint()
def download_checkpoint_local(model_size: str = "15M", checkpoint: str = "final", output_dir: str = "downloaded_model"):
    """Download checkpoint to local directory."""
    import os
    
    files = download_checkpoint.remote(model_size, checkpoint)
    
    os.makedirs(output_dir, exist_ok=True)
    for filename, content in files.items():
        filepath = os.path.join(output_dir, filename)
        with open(filepath, 'wb') as f:
            f.write(content)
        print(f"Downloaded: {filename}")
    
    print(f"\nModel saved to: {output_dir}")


@app.function(
    image=image,
    gpu="H100",
    volumes={CHECKPOINT_DIR: checkpoint_volume},
    timeout=300,
)
def evaluate(model_size: str = "15M", checkpoint: str = "final"):
    """Evaluate a trained model."""
    import os
    import torch
    from transformers import LlamaForCausalLM, LlamaTokenizerFast
    
    checkpoint_path = os.path.join(CHECKPOINT_DIR, model_size, checkpoint)
    
    if not os.path.exists(checkpoint_path):
        raise ValueError(f"Checkpoint not found: {checkpoint_path}")
    
    print(f"Loading model from {checkpoint_path}")
    model = LlamaForCausalLM.from_pretrained(checkpoint_path)
    model.eval()
    
    tokenizer = LlamaTokenizerFast(
        tokenizer_file=f"{TOKENIZER_DIR}/tokenizer.json",
        unk_token="<unk>",
        pad_token="<pad>",
        bos_token="<s>",
        eos_token="</s>",
        padding_side="right",
    )
    
    print(f"Model parameters: {model.num_parameters():,}")
    
    # Test prompts
    prompts = [
        """### TABLEDATA

CREATE TABLE users (
    id INT PRIMARY KEY,
    name VARCHAR(100),
    email VARCHAR(255)
);

### STATEMENT

SELECT * FROM users WHERE""",
        """### TABLEDATA

CREATE TABLE orders (
    id INT PRIMARY KEY,
    user_id INT REFERENCES users(id),
    total DECIMAL(10, 2),
    created_at TIMESTAMP
);

### STATEMENT

INSERT INTO orders""",
    ]
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device)
    
    for i, prompt in enumerate(prompts):
        print(f"\n--- Prompt {i+1} ---")
        print(prompt)
        
        input_ids = tokenizer.encode(prompt, return_tensors="pt").to(device)
        output = model.generate(
            input_ids,
            max_length=len(input_ids[0]) + 50,
            num_return_sequences=3,
            num_beams=5,
            eos_token_id=tokenizer.pad_token_id,
            pad_token_id=tokenizer.pad_token_id,
        )
        
        print("\nCompletions:")
        for j, seq in enumerate(output):
            completion = tokenizer.decode(seq, skip_special_tokens=True)[len(prompt):]
            print(f"  {j+1}: {completion.strip()}")


if __name__ == "__main__":
    # For local testing
    print("Use 'modal run modal_train.py::train' to run on Modal")

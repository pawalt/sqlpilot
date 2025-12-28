# SQLPilot

SQLPilot is a synthetic data generation and training pipeline for extremely small LLaMA-architecture models that perform SQL statement autocomplete. The models are trained to predict SQL statement completions given database table schemas.

## Project Purpose

Train tiny language models (260K to 42M parameters) that can autocomplete SQL statements when given CREATE TABLE schemas as context. The use case is real-time SQL autocomplete in IDEs or database tools.

## Architecture Overview

### Data Generation Pipeline (`datagen/main.py`)

Generates synthetic training data using OpenAI's GPT-3.5/GPT-4:

1. **Topic Generation**: Creates ~75 database application topics (e-commerce, healthcare, etc.)
2. **Schema Generation**: For each topic, generates CockroachDB CREATE TABLE schemas with 1-5 tables
3. **Statement Generation**: Generates SQL statements (SELECT, INSERT, UPDATE, DELETE, TRUNCATE, etc.) for each schema
4. **Data Augmentation**: Randomizes table/column names to prevent memorization and improve generalization

Training data format:
```
### TABLEDATA

CREATE TABLE users (
    id INT PRIMARY KEY,
    name VARCHAR(100)
);

### STATEMENT

SELECT * FROM users WHERE
```

### Custom Tokenizer (`hugtrain/create_tokenizer.py`)

Creates a tiny 512-token SentencePiece BPE tokenizer specialized for SQL syntax. Small vocabulary enables smaller embedding tables and faster inference.

### Training (`modal_train.py`)

Trains LLaMA-architecture models on Modal using HuggingFace Transformers. Model size configurations:

| Config | Params | Hidden | Layers | Heads |
|--------|--------|--------|--------|-------|
| 260K   | ~260K  | 64     | 5      | 8     |
| 15M    | ~15M   | 288    | 6      | 6     |
| 42M    | ~42M   | 512    | 8      | 8     |

Uses Weights & Biases for experiment tracking (optional).

### Inference (`hugtrain/autocomplete.py`, `hugtrain/test_eval.py`)

- `autocomplete.py`: Interactive curses-based demo that shows real-time SQL completions as you type
- `test_eval.py`: Evaluation script that runs the model on test prompts

## Directory Structure

```
sqlpilot/
├── modal_train.py           # Modal training driver (main entry point)
├── requirements.txt         # Local dependencies (just modal)
├── datagen/
│   └── main.py              # Data generation pipeline (legacy)
├── hugtrain/
│   ├── main.py              # Local training script (legacy)
│   ├── create_tokenizer.py  # Custom tokenizer training
│   ├── autocomplete.py      # Interactive demo
│   ├── test_eval.py         # Evaluation script
│   └── smol_tokenizer/      # Trained tokenizer files
└── data/
    ├── topics.json          # Generated topics
    ├── topic_detail.json    # Expanded topic details
    ├── tons_of_tables/      # Generated table schemas
    ├── diverse_statements/  # Generated SQL statements
    └── diverse_training_data/ # Final training data (JSON)
```

## Key Concepts

1. **Tiny Models**: The goal is models small enough to run on-device with low latency for real-time autocomplete
2. **Name Randomization**: Table/column names are randomized during data generation so models learn SQL structure, not specific names
3. **Custom Tokenizer**: A 512-token vocabulary keeps the embedding table small
4. **CockroachDB Focus**: Schemas are formatted as CockroachDB SHOW CREATE TABLE output

## Running

### Setup

```bash
pip install modal
modal setup  # Authenticate with Modal
```

### Training (Modal)

```bash
# First time: upload training data to Modal volume
modal run modal_train.py::upload_data_local

# Train (auto-resumes from latest checkpoint)
modal run modal_train.py::train

# Train specific model size (260K, 15M, or 42M)
modal run modal_train.py::train --model-size 42M

# List available checkpoints
modal run modal_train.py::list_checkpoints

# Download trained model locally
modal run modal_train.py::download_checkpoint_local --model-size 15M --checkpoint final

# Evaluate a trained model
modal run modal_train.py::evaluate --model-size 15M
```

Modal volumes used:
- `sqlpilot-checkpoints`: Persistent checkpoint storage (survives container restarts)
- `sqlpilot-data`: Training data storage

To use Weights & Biases for tracking, create a Modal secret named `wandb-secret` with your `WANDB_API_KEY`.

### Interactive Demo
```bash
cd hugtrain
python autocomplete.py
```
Enter table schemas, then type SQL to see completions.

## Dependencies

Locally: just `modal` (see `requirements.txt`)

Remote dependencies are handled via Modal's image definition in `modal_train.py`.

# SQLPilot

SQLPilot is a synthetic data generation and training pipeline for extremely small LLaMA-architecture models that perform SQL statement autocomplete. The models are trained to predict SQL statement completions given database table schemas.

## Project Purpose

Train tiny language models (260K to 42M parameters) that can autocomplete SQL statements when given CREATE TABLE schemas as context. The use case is real-time SQL autocomplete in IDEs or database tools.

## Architecture Overview

### Data Generation Pipeline (Modal)

Generates synthetic training data using a self-hosted Qwen3-235B model on Modal:

**Files:**
- `datagen/vllm_server.py` - vLLM inference server serving Qwen3-235B-A22B-Instruct-2507-FP8
- `datagen/modal_datagen.py` - Data generation functions using instructor for structured outputs
- `datagen/main.py` - Legacy local generation script (uses OpenAI API)

**Pipeline Steps:**
1. **Topic Generation**: Creates ~75 database application topics (e-commerce, healthcare, etc.)
2. **Topic Details**: Expands each topic into 100+ specific use case examples
3. **Schema Generation**: For each example, generates CockroachDB CREATE TABLE schemas with 1-5 tables
4. **Statement Generation**: Generates SQL statements (SELECT, INSERT, UPDATE, DELETE, TRUNCATE) for each schema
5. **Data Augmentation**: Randomizes table/column names to prevent memorization and improve generalization

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

### Custom Tokenizer (`create_tokenizer.py`)

Creates a tiny 512-token SentencePiece BPE tokenizer specialized for SQL syntax. Small vocabulary enables smaller embedding tables and faster inference.

### Training (`modal_train.py`)

Trains LLaMA-architecture models on Modal using HuggingFace Transformers. Model size configurations:

| Config | Params | Hidden | Layers | Heads |
|--------|--------|--------|--------|-------|
| 260K   | ~260K  | 64     | 5      | 8     |
| 15M    | ~15M   | 288    | 6      | 6     |
| 42M    | ~42M   | 512    | 8      | 8     |

Uses Weights & Biases for experiment tracking (optional).

### Inference (`autocomplete.py`)

Interactive curses-based demo that shows real-time SQL completions as you type. Uses uv for dependency management.

## Directory Structure

```
sqlpilot/
├── modal_train.py           # Modal training driver (main entry point)
├── create_tokenizer.py      # Modal tokenizer training
├── autocomplete.py          # Interactive terminal demo (uv script)
├── requirements.txt         # Local dependencies (just modal)
├── tokenizer/
│   └── tokenizer.json       # Trained tokenizer
├── web/
│   ├── index.html           # Static web UI (Transformers.js)
│   ├── model/               # ONNX model for web inference
│   └── README.md            # Web deployment instructions
├── datagen/
│   ├── vllm_server.py       # Modal vLLM server (Qwen3-235B)
│   ├── modal_datagen.py     # Modal data generation functions
│   └── main.py              # Legacy local generation (OpenAI API)
└── data/                    # Local data (also on Modal volume)
    ├── topics.json          # Generated topics
    ├── topic_detail.json    # Expanded topic details
    ├── tons_of_tables/      # Generated table schemas
    ├── diverse_statements/  # Generated SQL statements
    └── raw/                 # Final training data (JSON)
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

### Data Generation (Modal)

```bash
# Deploy the vLLM inference server (Qwen3-235B, requires 2x H100)
modal deploy datagen/vllm_server.py
# Note the URL from the output (e.g., https://your-app.modal.run)

# Run the full data generation pipeline
modal run datagen/modal_datagen.py::generate_all --vllm-url https://YOUR-VLLM-URL.modal.run/v1

# Or run individual steps
modal run datagen/modal_datagen.py::generate_topics --vllm-url YOUR_URL
modal run datagen/modal_datagen.py::generate_topic_details --vllm-url YOUR_URL
modal run datagen/modal_datagen.py::generate_schemas --vllm-url YOUR_URL
modal run datagen/modal_datagen.py::generate_statements --vllm-url YOUR_URL
modal run datagen/modal_datagen.py::generate_training_data
```

### Training (Modal)

```bash
# First time: upload training data to Modal volume
modal run modal_train.py::upload_data_local

# Tokenize the data (run once after uploading)
modal run modal_train.py::tokenize

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

### Tokenizer (Modal)

```bash
# Train a new tokenizer from the data
modal run create_tokenizer.py::train_tokenizer

# Download the tokenizer locally
modal run create_tokenizer.py::download_tokenizer_local

# Test the tokenizer
modal run create_tokenizer.py::test_tokenizer
```

### Interactive Demo (Terminal)

```bash
# Run with uv (handles dependencies automatically)
uv run autocomplete.py

# Or specify a model path
uv run autocomplete.py ./downloaded_model
```

Enter table schemas, then type SQL to see completions.

### Web UI

A static HTML/JS interface using [Transformers.js](https://huggingface.co/docs/transformers.js) for in-browser inference:

```bash
# Serve locally
cd web
python -m http.server 8080
# Open http://localhost:8080
```

To use with your trained model:

1. Export to ONNX: `optimum-cli export onnx --model ./downloaded_model --task text-generation ./onnx_model`
2. Upload to HuggingFace Hub
3. Update `MODEL_ID` in `web/index.html`

See `web/README.md` for full instructions.

## Dependencies

Locally: just `modal` and `uv` (see `requirements.txt`)

Remote dependencies are handled via Modal's image definition in `modal_train.py`.

Modal volumes used:
- `sqlpilot-checkpoints`: Persistent checkpoint storage (survives container restarts)
- `sqlpilot-data`: Training data, tokenized datasets, and tokenizer storage
- `sqlpilot-model-cache`: Cached Qwen3 model weights for vLLM server

To use Weights & Biases for tracking, create a Modal secret named `wandb-secret` with your `WANDB_API_KEY`.

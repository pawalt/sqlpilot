from transformers import LlamaForCausalLM, LlamaTokenizer, LlamaConfig, TrainingArguments, Trainer, LlamaTokenizerFast, TrainerCallback, AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
import re
from datasets import load_dataset, Dataset
import os
import json
from transformers import default_data_collator
import torch
from tokenizers import SentencePieceBPETokenizer, Tokenizer
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from bitsandbytes.functional import quantize_blockwise, dequantize_blockwise
from trl import SFTTrainer

os.environ["WANDB_PROJECT"] = "sqlpilot"  # name your W&B project

directory = '../data/diverse_training_data'
def load_json_files():
    files = os.listdir(directory)
    files.sort()
    for filename in files:
        if filename.endswith(".json"):
            file_path = os.path.join(directory, filename)
            with open(file_path, 'r', encoding='utf-8') as file:
                data = json.load(file)
                for example in data:
                    yield {'text': example}

class CustomTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False):
        labels = inputs.pop("labels")
        outputs = model(**inputs)
        logits = outputs.logits
        loss_fct = torch.nn.CrossEntropyLoss()
        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()
        loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
        return (loss, outputs) if return_outputs else loss

def get_tokenizer():
    # Load the trained tokenizer
    tokenizer = LlamaTokenizerFast(
        tokenizer_file="smol_tokenizer/tokenizer.json",

        unk_token="<unk>",
        pad_token="<pad>",
        bos_token="<s>",
        eos_token="</s>",

        padding_side="right"
    )
    # tokenizer = LlamaTokenizer.from_pretrained("KoboldAI/llama2-tokenizer")
    # tokenizer.pad_token = tokenizer.eos_token
    return tokenizer

def get_dataset():
    # THIS IS SO FUCKED UP YOU NEED TO MANUALLY BUST THIS CACHE FUCKING OFTEN
    dataset = Dataset.from_generator(load_json_files)

    # Split the dataset into train and validation sets
    dataset = dataset.train_test_split(test_size=0.1)  # Adjust the test_size as needed

    tokenizer = get_tokenizer()

    def tokenize_function(examples):
        inputs = tokenizer(examples["text"], padding="max_length", truncation=True, max_length=512)
        labels = inputs["input_ids"].copy()
        inputs["labels"] = labels
        return inputs

    return dataset.map(
        tokenize_function,
        batched=True,
        num_proc=4,
        remove_columns=["text"],
        load_from_cache_file=True,
        cache_file_names={
            "train": "train_dataset.arrow",
            "test": "test_dataset.arrow",
        },
    )

class SaveBestModelCallback(TrainerCallback):
    def on_evaluate(self, args, state, control, metrics, **kwargs):
        print(f"VALIDATION: current eval loss: {metrics['eval_loss']}, best eval loss: {state.best_metric}")
        if state.best_metric is None or metrics["eval_loss"] < state.best_metric:
            output_dir = "output/best_model"
            self.model.save_pretrained(output_dir)
            state.best_metric = metrics["eval_loss"]

def run_full_train():
    modelargs_42M = {
        "hidden_size": 512,
        "num_hidden_layers": 8,
        "num_attention_heads": 8,
        "num_key_value_heads": 8,
        "intermediate_size": 4 * 512,
        "max_position_embeddings": 512,
    }

    modelargs_15M_long = {
        "hidden_size": 288,
        "num_hidden_layers": 12,
        "num_attention_heads": 6,
        "num_key_value_heads": 6,
        "intermediate_size": 4 * 288,
        "max_position_embeddings": 512,
    }

    modelargs_15M = {
        "hidden_size": 288,
        "num_hidden_layers": 6,
        "num_attention_heads": 6,
        "num_key_value_heads": 6,
        "intermediate_size": 4 * 288,
        "max_position_embeddings": 512,
    }

    modelargs_260k = {
        "hidden_size": 64,
        "num_hidden_layers": 5,
        "num_attention_heads": 8,
        "num_key_value_heads": 4,
        "intermediate_size": 4 * 64,
        "max_position_embeddings": 512,
    }

    modelargs_260k_lengthened = {
        "hidden_size": 64,
        "num_hidden_layers": 8,
        "num_attention_heads": 8,
        "num_key_value_heads": 4,
        "intermediate_size": 4 * 64,
        "max_position_embeddings": 512,
    }

    # Initialize the LLaMA model
    model = LlamaForCausalLM(LlamaConfig(
        **modelargs_15M,
        vocab_size=512,
    ))

    # Specify the directory containing the checkpoints
    output_dir = "output"

    # Define the training arguments
    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=3,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=16,
        warmup_steps=500,
        weight_decay=0.01,
        logging_dir="logs",
        logging_steps=10,
        gradient_accumulation_steps=8,
        learning_rate=1e-3,

        report_to="wandb",

        # eval strat
        # evaluation_strategy="steps",
        # eval_steps=1000,
        # save_steps=1000,
        # save_total_limit=2,
        # load_best_model_at_end=True,
        # metric_for_best_model="eval_loss",
    )

    tokenized_dataset = get_dataset()

    # Create the Trainer
    trainer = CustomTrainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset["train"],
        eval_dataset=tokenized_dataset["test"],
        data_collator=default_data_collator,
        # callbacks=[SaveBestModelCallback],
    )

    # Start training
    trainer.train(
        resume_from_checkpoint=False,
    )

def get_latest_checkpoint():
    # Specify the directory containing the checkpoints
    checkpoint_dir = "output"

    # Get a list of all checkpoint directories
    checkpoint_dirs = [dir for dir in os.listdir(checkpoint_dir) if dir.startswith("checkpoint-")]

    # Find the checkpoint with the highest index
    latest_checkpoint = max(checkpoint_dirs, key=lambda x: int(re.findall(r'\d+', x)[0]))

    print(f"using checkpoint {latest_checkpoint}")

    # Construct the path to the latest checkpoint
    return os.path.join(checkpoint_dir, latest_checkpoint)

def run_eval():
    # Load the model from the checkpoint
    model = LlamaForCausalLM.from_pretrained(get_latest_checkpoint())

    print(f"num params: {model.num_parameters()}")

    # Set the model to evaluation mode
    model.eval()

    tokenized_dataset = get_dataset()

    # Create the Trainer
    trainer = CustomTrainer(
        model=model,
        args=TrainingArguments(output_dir="temp_output"),
        train_dataset=tokenized_dataset["train"],
        eval_dataset=tokenized_dataset["test"],
        data_collator=default_data_collator,
    )

    # Evaluate the model
    eval_results = trainer.evaluate()

    print(eval_results)

def run_peft_train_mistral():
    # Load the Mistral model and tokenizer
    nf4_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.bfloat16
    )

    model = AutoModelForCausalLM.from_pretrained(
        "mistralai/Mistral-7B-Instruct-v0.1",
        device_map='auto',
        quantization_config=nf4_config,
        use_cache=False
    )
    tokenizer = AutoTokenizer.from_pretrained("mistralai/Mistral-7B-v0.1")
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    # Prepare the model for 4-bit training
    model = prepare_model_for_kbit_training(model)

    # Define the LoRA configuration
    peft_config = LoraConfig(
        lora_alpha=16,
        lora_dropout=0.1,
        r=64,
        bias="none",
        task_type="CAUSAL_LM"
    )

    # Apply LoRA to the model
    model = get_peft_model(model, peft_config)

    dataset = Dataset.from_generator(load_json_files)

    # Split the dataset into train and validation sets
    dataset = dataset.train_test_split(
        test_size=0.1,
        seed=42,
    )  # Adjust the test_size as needed

    # Define the training arguments
    args = TrainingArguments(
        max_steps=100000,

        output_dir="mistral_sql_gen",
        per_device_train_batch_size=8,
        warmup_steps=0.03,
        # num_train_epochs=2,
        logging_steps=10,

        save_strategy="steps",
        save_steps=100,

        learning_rate=2e-4,
        bf16=True,
        lr_scheduler_type='constant',

        report_to="wandb",
    )

    max_seq_length = 512
    trainer = SFTTrainer(
        model=model,
        peft_config=peft_config,
        max_seq_length=max_seq_length,
        tokenizer=tokenizer,
        packing=False,
        args=args,
        train_dataset=dataset["train"],
        eval_dataset=dataset["test"],
        dataset_text_field="text",
    )

    # Start training
    trainer.train()

    # Save the model
    trainer.save_model("mistral_sql_gen")

def find_best_model():
    checkpoint_dir = "output"

    # Load the saved model checkpoints
    checkpoints = [os.path.join(checkpoint_dir, d) for d in os.listdir(checkpoint_dir) if d.startswith("checkpoint-")]

    tokenized_dataset = get_dataset()

    # Evaluate each checkpoint on the test set
    best_checkpoint = None
    best_eval_loss = float("inf")
    for checkpoint in checkpoints:
        model = LlamaForCausalLM.from_pretrained(checkpoint)
        trainer = CustomTrainer(
            model=model,
            args=TrainingArguments(output_dir="temp_output"),
            train_dataset=tokenized_dataset["train"],
            eval_dataset=tokenized_dataset["test"],
            data_collator=default_data_collator,
        )
        eval_results = trainer.evaluate()
        if eval_results["eval_loss"] < best_eval_loss:
            best_eval_loss = eval_results["eval_loss"]
            best_checkpoint = checkpoint

    # Save the best model
    output_dir = "tosave/270k_long_customtok"
    model = LlamaForCausalLM.from_pretrained(best_checkpoint)
    model.save_pretrained(output_dir)

def inspect_training_data():
    gen = load_json_files()
    for i in range(10):
        print(next(gen)["text"])
        print("--------------")

if __name__ == "__main__":
    run_full_train()

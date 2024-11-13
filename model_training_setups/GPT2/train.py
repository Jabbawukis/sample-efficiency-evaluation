import torch
from transformers import (
    AutoConfig,
    GPT2LMHeadModel,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
    DataCollatorForLanguageModeling,
)
from datasets import load_dataset, DatasetDict, load_from_disk

# torch.cuda.set_device('cuda:0')

# Load the dataset
dataset = load_dataset("wikimedia/wikipedia", "20231101.en")

# Initialize the tokenizer
context_length = 128
tokenizer = AutoTokenizer.from_pretrained("gpt2")
tokenizer.pad_token = tokenizer.eos_token  # Set pad token


def tokenize(element):
    # Tokenize input, handling long sequences with `return_overflowing_tokens`
    outputs = tokenizer(
        element["text"],
        truncation=True,
        max_length=context_length,
        padding="max_length",  # Pad to max length
        return_overflowing_tokens=True,
        return_length=True,
    )

    # Ensure each example has a consistent number of tokens
    input_batch = []
    for length, input_ids in zip(outputs["length"], outputs["input_ids"]):
        if length == context_length:  # Only keep sequences of the expected length
            input_batch.append(input_ids)

    # Return the list as a dictionary
    return {"input_ids": input_batch}


# Tokenize the entire dataset and remove unnecessary columns
tokenized_datasets = dataset.map(tokenize, batched=True, remove_columns=dataset["train"].column_names)

# Load the previously saved tokenized dataset
# tokenized_datasets = load_from_disk('wikipedia_20231101_en/tokenized_ds')

# Save the tokenized dataset (optional)
tokenized_datasets.save_to_disk("wikipedia_20231101_en/tokenized_ds")

tokenized_datasets.set_format("torch")

# Split into train and eval (e.g., 90% train, 10% eval)
train_test_split = tokenized_datasets["train"].train_test_split(test_size=0.1, seed=42)
tokenized_datasets = DatasetDict({"train": train_test_split["train"], "eval": train_test_split["test"]})

# Data collator for causal language modeling
data_collator = DataCollatorForLanguageModeling(tokenizer, mlm=False)

# Configure the model
config = AutoConfig.from_pretrained(
    "gpt2",
    vocab_size=len(tokenizer),
    n_ctx=context_length,
    bos_token_id=tokenizer.bos_token_id,
    eos_token_id=tokenizer.eos_token_id,
)

# Initialize the model from scratch
model = GPT2LMHeadModel(config)

# Training arguments
training_args = TrainingArguments(
    output_dir="./gpt2-wikipedia",  # Output directory
    per_device_train_batch_size=32,
    gradient_accumulation_steps=8,
    num_train_epochs=3,
    weight_decay=0.1,
    warmup_steps=1_000,
    lr_scheduler_type="cosine",
    learning_rate=5e-4,
    save_steps=5_000,
    logging_steps=5_000,
    evaluation_strategy="steps",  # Enable evaluation during training
    eval_steps=2_500,  # Evaluate every 2,500 steps
    fp16=True,  # Use mixed precision for faster training and reduced memory
)

# Initialize the Trainer with both train and eval datasets
trainer = Trainer(
    model=model,
    args=training_args,
    data_collator=data_collator,
    train_dataset=tokenized_datasets["train"],
    eval_dataset=tokenized_datasets["eval"],
    tokenizer=tokenizer,
)

# Train the model
trainer.train()

# Save the final model and tokenizer
model.save_pretrained("./gpt2-wikipedia")
tokenizer.save_pretrained("./gpt2-wikipedia")
print("Training complete. Model saved at ./gpt2-wikipedia")

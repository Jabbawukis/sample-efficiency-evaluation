from transformers import (
    AutoModelForCausalLM,
    Mamba2Config,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
    DataCollatorForLanguageModeling,
)
from datasets import load_dataset, DatasetDict, load_from_disk

# import os
# os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
# os.environ["CUDA_VISIBLE_DEVICES"] = "2"
output = "./mamba2-wikipedia"

# Initialize the tokenizer
context_length = 128
tokenizer = AutoTokenizer.from_pretrained("gpt2") # use GPT2 tokenizer for mamba2
tokenizer.pad_token = tokenizer.eos_token  # Set pad token


############################################################################################
# Load the dataset
dataset = load_dataset("wikimedia/wikipedia", "20231101.en").shuffle(seed=42)
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
tokenized_dataset = dataset.map(tokenize, batched=True, remove_columns=dataset["train"].column_names)

# Save the tokenized dataset (optional)
tokenized_dataset.save_to_disk("wikipedia_20231101_en/tokenized_ds")
############################################################################################

# Load the previously saved tokenized dataset and ignore the above steps encased in # for the dataset processing
# tokenized_dataset = load_from_disk('wikipedia_20231101_en/tokenized_ds')

# Split into train and eval (e.g., 90% train, 10% eval)
# train_test_split = tokenized_dataset["train"].train_test_split(test_size=0.1, seed=42)
# tokenized_dataset = DatasetDict({"train": train_test_split["train"], "eval": train_test_split["test"]})

# Data collator for causal language modeling
data_collator = DataCollatorForLanguageModeling(tokenizer, mlm=False)

# Configure the model
config = Mamba2Config(
    vocab_size=len(tokenizer),
    bos_token_id=tokenizer.bos_token_id,
    eos_token_id=tokenizer.eos_token_id,
    pad_token_id=tokenizer.pad_token_id,
    hidden_size=768,
    num_heads=24,
    num_hidden_layers=24,
    state_size=32
)

# Initialize the model from scratch
model = AutoModelForCausalLM.from_config(config)

# Training arguments
training_args = TrainingArguments(
    output_dir=output,  # Output directory
    per_device_train_batch_size=32,
    # per_device_eval_batch_size=32,
    # eval_strategy="steps",
    # eval_steps=5_000,
    logging_steps=3650,
    gradient_accumulation_steps=8,
    num_train_epochs=1,
    weight_decay=0.1,
    warmup_steps=1_000,
    lr_scheduler_type="cosine",
    learning_rate=5e-4,
    save_steps=3650,
    fp16=True,
)

# Initialize the Trainer with both train and eval datasets
trainer = Trainer(
    model=model,
    args=training_args,
    data_collator=data_collator,
    train_dataset=tokenized_dataset["train"],
    # eval_dataset=tokenized_dataset["eval"],
    processing_class=tokenizer,
)

# Train the model
trainer.train()

# Save the final model and tokenizer
model.save_pretrained(output)
tokenizer.save_pretrained(output)
print(f"Training complete. Model saved at {output}")

from transformers import GPT2Tokenizer,GPT2LMHeadModel,Trainer,TrainingArguments
from datasets import load_dataset
tokenizer=GPT2Tokenizer.from_pretrained("gpt2")
tokenizer.pad_token=tokenizer.eos_token  
model=GPT2LMHeadModel.from_pretrained("gpt2")  
dataset=load_dataset("wikitext","wikitext-2-raw-v1",split="train[:5%]")
def tokenize_function(examples):
    tokenized=tokenizer(examples["text"],truncation=True,padding="max_length",max_length=128)
    tokenized["labels"]=tokenized["input_ids"].copy()
    return tokenized
tokenized_datasets=dataset.map(tokenize_function,batched=True)
training_args=TrainingArguments(
    output_dir="../models/gpt_model",
    eval_strategy="no", 
    save_strategy="epoch",
    logging_dir="../logs",
    per_device_train_batch_size=32,  
    num_train_epochs=1, 
    max_steps=200,  
    fp16=True,  
    warmup_steps=100,
    weight_decay=0.01,
)
trainer=Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_datasets,
)
trainer.train()
model.save_pretrained("../models/gpt_model")
tokenizer.save_pretrained("../models/gpt_model")
print("GPT-2 model saved successfully.")
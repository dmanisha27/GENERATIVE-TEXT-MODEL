from transformers import GPT2Tokenizer,GPT2LMHeadModel
import torch
model_path = r"C:\Users\chari\Desktop\Codetech\GenerativeTextModel\models\gpt_model"
tokenizer = GPT2Tokenizer.from_pretrained(model_path)
model = GPT2LMHeadModel.from_pretrained(model_path)
tokenizer.pad_token = tokenizer.eos_token  
def generate_text(prompt, max_new_tokens=100):
    inputs = tokenizer(prompt, return_tensors="pt", padding=True, truncation=True, max_length=128)
    with torch.no_grad():
        outputs = model.generate(
            inputs.input_ids, 
            attention_mask=inputs.attention_mask, 
            max_new_tokens=max_new_tokens, 
            temperature=1.1,  
            top_k=40,  
            top_p=0.9, 
            repetition_penalty=1.3,  
            do_sample=True,  
            num_return_sequences=1, 
            early_stopping=True,  
            eos_token_id=tokenizer.eos_token_id 
        )
    return tokenizer.decode(outputs[0], skip_special_tokens=True)
print(generate_text("The future of AI is"))
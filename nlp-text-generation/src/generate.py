from transformers import GPT2Tokenizer, GPT2LMHeadModel

def generate_text(prompt, model_path, max_length=50):
    tokenizer = GPT2Tokenizer.from_pretrained(model_path)
    model = GPT2LMHeadModel.from_pretrained(model_path)

    inputs = tokenizer.encode(prompt, return_tensors="pt")
    outputs = model.generate(inputs, max_length=max_length, num_return_sequences=1)
    return tokenizer.decode(outputs[0], skip_special_tokens=True)

if __name__ == "__main__":
    model_path = "../models/gpt2-finetuned"
    prompt = "Once upon a time"
    generated_text = generate_text(prompt, model_path)
    print("Generated Text:")
    print(generated_text)
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

# Use a smaller Llama 2 model for better CPU performance
model_name = "mistralai/Mistral-7B-v0.1"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name, device_map="auto", torch_dtype=torch.float32)  # Ensuring compatibility with CPU

# Function to Generate Response
def generate_response(prompt):
    inputs = tokenizer(prompt, return_tensors="pt").to("cpu")  # Ensuring execution on CPU
    outputs = model.generate(**inputs, max_length=200)
    return tokenizer.decode(outputs[0], skip_special_tokens=True)

# Example Usage
user_prompt = "Tell me about dream interpretation."
print(generate_response(user_prompt))

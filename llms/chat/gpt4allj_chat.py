import transformers

model_name = 'nomic-ai/gpt4all-j'

model = transformers.AutoModelForCausalLM.from_pretrained(model_name, revision="v1.2-jazzy")

tokenizer = transformers.AutoTokenizer.from_pretrained(model_name)

def generate_response(system_input, user_input):
    # Format the input using the provided template
    prompt = f"### System:\n{system_input}\n### User:\n{user_input}\n### Assistant:\n"

    # Tokenize and encode the prompt
    inputs = tokenizer.encode(prompt, return_tensors="pt", add_special_tokens=False)

    # Generate a response
    outputs = model.generate(inputs, max_length=1000, num_return_sequences=1)
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)

    # Extract only the assistant's response
    return response.split("### Assistant:\n")[-1]

if __name__ == '__main__':
    # Example usage
    system_input = "You are a math expert assistant. Your mission is to help users understand and solve various math problems. You should provide step-by-step solutions, explain reasonings and give the correct answer."
    user_input = "calculate 100 + 520 + 60"
    response = generate_response(system_input, user_input)
    print(response)
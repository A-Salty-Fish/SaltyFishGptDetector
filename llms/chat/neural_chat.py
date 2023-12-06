import transformers


def init_model_and_tokenizer():


    model_name = 'Intel/neural-chat-7b-v3-1'
    model = transformers.AutoModelForCausalLM.from_pretrained(model_name).cuda()
    tokenizer = transformers.AutoTokenizer.from_pretrained(model_name)

    return model, tokenizer



def generate_with_system_input_response(model, tokenizer, system_input, user_input):
    # Format the input using the provided template
    prompt = f"### System:\n{system_input}\n### User:\n{user_input}\n### Assistant:\n"

    # Tokenize and encode the prompt
    inputs = tokenizer.encode(prompt, return_tensors="pt", add_special_tokens=False).cuda()

    # Generate a response
    outputs = model.generate(inputs, max_length=1000, num_return_sequences=1).cuda()
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)

    # Extract only the assistant's response
    return response.split("### Assistant:\n")[-1]


def chat(model, tokenizer, context):
    return generate_with_system_input_response(model, tokenizer,
                                        "You are a math expert assistant. Your mission is to help users understand and solve various math problems. You should provide step-by-step solutions, explain reasonings and give the correct answer.", context)


if __name__ == '__main__':
    # Example usage
    user_input = "calculate 100 + 520 + 60"
    model, tokenizer = init_model_and_tokenizer()
    print(chat(model, tokenizer, user_input))
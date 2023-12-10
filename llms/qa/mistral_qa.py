from transformers import AutoModelForCausalLM, AutoTokenizer

device = "cuda" # the device to load the model onto

def init_model_and_tokenizer():

    model = AutoModelForCausalLM.from_pretrained("mistralai/Mistral-7B-Instruct-v0.1")
    tokenizer = AutoTokenizer.from_pretrained("mistralai/Mistral-7B-Instruct-v0.1")

    return model, tokenizer


def chat(model, tokenizer, context):


    messages = [
        {"role": "user", "content": context}
        # {"role": "assistant", "content": "Well, I'm quite partial to a good squeeze of fresh lemon juice. It adds just the right amount of zesty flavour to whatever I'm cooking up in the kitchen!"},
        # {"role": "user", "content": "Do you have mayonnaise recipes?"}
    ]

    encodeds = tokenizer.apply_chat_template(messages, return_tensors="pt")

    model_inputs = encodeds.to(device)
    model.to(device)

    generated_ids = model.generate(model_inputs, max_new_tokens=1000, do_sample=True)
    decoded = tokenizer.batch_decode(generated_ids)
    return decoded[0]


if __name__ == '__main__':
    model, tokenizer = init_model_and_tokenizer()
    print(chat(model, tokenizer, "Do you have mayonnaise recipes?"))
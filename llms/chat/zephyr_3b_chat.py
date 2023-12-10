from transformers import AutoModelForCausalLM, AutoTokenizer


def init_model_and_tokenizer():

    tokenizer = AutoTokenizer.from_pretrained('stabilityai/stablelm-zephyr-3b')
    model = AutoModelForCausalLM.from_pretrained(
        'stabilityai/stablelm-zephyr-3b',
        trust_remote_code=True,
        device_map="auto"
    )

    return model, tokenizer

def chat(model ,tokenizer, context):
    prompt = [{'role': 'user', 'content': context}]
    inputs = tokenizer.apply_chat_template(
        prompt,
        add_generation_prompt=True,
        return_tensors='pt'
    )

    tokens = model.generate(
        inputs.to(model.device),
        max_new_tokens=1024,
        temperature=0.8,
        do_sample=True
    )

    return (tokenizer.decode(tokens[0], skip_special_tokens=False))


if __name__ == '__main__':
    model, tokenizer = init_model_and_tokenizer()
    print(chat(model, tokenizer, "please tell a story of about 500 words about a child and a dog."))
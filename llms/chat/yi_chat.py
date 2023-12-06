from transformers import AutoModelForCausalLM, AutoTokenizer


def init_model_and_token():

    model_path = '01-ai/Yi-6b-Chat'

    tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=False)

    # Since transformers 4.35.0, the GPT-Q/AWQ model can be loaded using AutoModelForCausalLM.
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        device_map="auto",
        torch_dtype='auto'
    ).eval()

    return model, tokenizer


def chat(model, tokenizer, context):

    # Prompt content: "hi"
    messages = [
        {"role": "user", "content": context}
    ]

    input_ids = tokenizer.apply_chat_template(conversation=messages, tokenize=True, add_generation_prompt=True, return_tensors='pt')
    output_ids = model.generate(input_ids.to('cuda'))
    response = tokenizer.decode(output_ids[0][input_ids.shape[1]:], skip_special_tokens=True)
    return response


if __name__ == '__main__':
    model, tokenizer = init_model_and_token()
    print(chat(model, tokenizer, "hi"))
    # Model response: "Hello! How can I assist you today?"

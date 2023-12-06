import torch
from transformers import LlamaTokenizer, LlamaForCausalLM



def init_model_and_tokenizer():
    ## v2 models
    model_path = 'openlm-research/open_llama_7b_v2'
    ## v1 models
    # model_path = 'openlm-research/open_llama_3b'
    # model_path = 'openlm-research/open_llama_7b'
    # model_path = 'openlm-research/open_llama_13b'
    tokenizer = LlamaTokenizer.from_pretrained(model_path)
    model = LlamaForCausalLM.from_pretrained(
        model_path, torch_dtype=torch.float16, device_map='auto',
    ).cuda()
    return model, tokenizer


def chat(model, tokenizer, context):
    input_ids = tokenizer(context, return_tensors="pt").input_ids.cuda()

    generation_output = model.generate(
        input_ids=input_ids, max_new_tokens=32
    ).cuda()
    return tokenizer.decode(generation_output[0])


if __name__ == '__main__':
    prompt = 'Q: What is the largest animal?\nA:'
    model,tokenizer = init_model_and_tokenizer()
    print(chat(model, tokenizer, prompt))






import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.generation.utils import GenerationConfig

def init_model_and_tokenizer():

    tokenizer = AutoTokenizer.from_pretrained("baichuan-inc/Baichuan2-7B-Chat", use_fast=False, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained("baichuan-inc/Baichuan2-7B-Chat", device_map="auto", torch_dtype=torch.bfloat16, trust_remote_code=True).cuda()
    model.generation_config = GenerationConfig.from_pretrained("baichuan-inc/Baichuan2-7B-Chat")
    return model, tokenizer

def chat(model, tokenizer, context):
    messages = []
    messages.append({"role": "user", "content": context})
    response = model.chat(tokenizer, messages)
    return (response)


if __name__ == '__main__':
    model, tokenizer = init_model_and_tokenizer()
    print(chat(model, tokenizer, "hello, how are you."))

# bug
# https://github.com/baichuan-inc/Baichuan2/issues/204
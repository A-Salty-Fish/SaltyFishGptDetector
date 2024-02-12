import json
import random
import time

import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

device = "cuda"  # the device to load the model onto


def init_model_and_tokenizer():
    start_time = time.time()
    tokenizer = AutoTokenizer.from_pretrained("mistralai/Mistral-7B-Instruct-v0.2")
    model = AutoModelForCausalLM.from_pretrained("mistralai/Mistral-7B-Instruct-v0.2", torch_dtype=torch.float16)

    end_time = time.time()
    print("load model successful: " + str(end_time - start_time))
    return model, tokenizer


def chat(model, tokenizer, context):
    # start_time = time.time()
    messages = [
        {"role": "user", "content": context}
        # {"role": "assistant", "content": "Well, I'm quite partial to a good squeeze of fresh lemon juice. It adds just the right amount of zesty flavour to whatever I'm cooking up in the kitchen!"},
        # {"role": "user", "content": "Do you have mayonnaise recipes?"}
    ]

    encodeds = tokenizer.apply_chat_template(messages, return_tensors="pt")

    model_inputs = encodeds.to(device)
    model.to(device)

    generated_ids = model.generate(model_inputs, max_new_tokens=512, do_sample=True,
                                   pad_token_id=tokenizer.eos_token_id)
    decoded = tokenizer.batch_decode(generated_ids)
    # end_time = time.time()
    # print("generate response successful: " + str(end_time - start_time))
    return decoded[0].split('[/INST]')[1].replace('</s>', '')


def batch_generate_contents(model, tokenizer,
                            in_file='../../data_collector/test_data/hc3_english_mix_multi/wiki_csai.mix.jsonl',
                            out_file='./wiki_csai.mix.human.jsonl',
                            nums=10):
    prompt_template = 'Please rewrite the following AI-generated text to make it more like human text, {without any useless content}: '
    in_jsons = []
    index = 0
    with open(in_file, 'r', encoding='utf-8') as in_f:
        for line in in_f:
            in_jsons.append(json.loads(line))
    with open(out_file, 'a', encoding='utf-8') as out_f:
        for in_json in in_jsons:
            index += 1
            print(str(index))
            out_json = in_json
            for i in range(0, nums):
                prompt = prompt_template + in_json['ai']
                out_json['prompt'] = prompt
                out_json['ai_rewrite'] = chat(model, tokenizer, prompt)
                out_f.write(json.dumps(out_json) + '\n')


if __name__ == '__main__':
    model, tokenizer = init_model_and_tokenizer()
    print('open_qa')
    batch_generate_contents(model, tokenizer,
                            '../../data_collector/test_data/hc3_english_mix_multi/open_qa.mix.jsonl',
                            './open_qa.mix.human.jsonl'
                            )
    print('finance')
    batch_generate_contents(model, tokenizer,
                            '../../data_collector/test_data/hc3_english_mix_multi/finance.mix.jsonl',
                            './finance.mix.human.jsonl'
                            )
    print('medicine')
    batch_generate_contents(model, tokenizer,
                            '../../data_collector/test_data/hc3_english_mix_multi/medicine.mix.jsonl',
                            './medicine.mix.human.jsonl'
                            )

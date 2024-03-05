import json
import random
import time

from datasets import load_dataset
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

device = "cuda"  # the device to load the model onto


def init_model_and_tokenizer():
    start_time = time.time()
    tokenizer = AutoTokenizer.from_pretrained("mistralai/Mistral-7B-Instruct-v0.2")
    model = AutoModelForCausalLM.from_pretrained("mistralai/Mistral-7B-Instruct-v0.2")

    end_time = time.time()
    print("load model successful: " + str(end_time - start_time))
    return model, tokenizer


def chat(model, tokenizer, context):
    # start_time = time.time()
    messages = [
        {"role": "user", "content": context}
    ]

    encodeds = tokenizer.apply_chat_template(messages, return_tensors="pt")

    model_inputs = encodeds.to(device)
    model.to(device)

    generated_ids = model.generate(model_inputs, max_new_tokens=1000, do_sample=True,
                                   pad_token_id=tokenizer.eos_token_id)
    decoded = tokenizer.batch_decode(generated_ids)
    return decoded[0].split('[/INST]')[1].replace('</s>', '')


def load_arxiv_datas(file_path):
    with open(file_path, 'r', encoding='utf-8') as in_f:
        json_objs = []
        for line in in_f:
            json_obj = json.loads(line)
            json_objs.append(json_obj)

        return json_objs


def generate_datas(file_path, model, tokenizer):
    prompt_template = 'please continue to write the following content: '
    json_objs = load_arxiv_datas(file_path)
    print(file_path)
    existed_lines = 0
    try:
        with open('./data/' + file_path.split('/')[-1], 'r', encoding='utf-8') as out_f:
            for line in out_f:
                existed_lines += 1
    except Exception as e:
        pass
    with open('./data/' + file_path.split('/')[-1], 'a', encoding='utf-8') as out_f:
        i = 0
        for json_obj in tqdm(json_objs):
            i+=1
            if i < existed_lines:
                continue
            json_obj['ai_rewrite'] = chat(model, tokenizer, prompt_template + json_obj['content'])
            out_f.write(json.dumps(json_obj) + '\n')


if __name__ == '__main__':
    model, tokenizer = init_model_and_tokenizer()
    generate_datas('../../data_collector/row_data/arxiv_paras/7.jsonl', model, tokenizer)
    generate_datas('../../data_collector/row_data/arxiv_paras/8.jsonl', model, tokenizer)
    generate_datas('../../data_collector/row_data/arxiv_paras/9.jsonl', model, tokenizer)
    generate_datas('../../data_collector/row_data/arxiv_paras/10.jsonl', model, tokenizer)
    generate_datas('../../data_collector/row_data/arxiv_paras/11.jsonl', model, tokenizer)



import json
import random
import time

from datasets import load_dataset
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoModel

device = "cuda"  # the device to load the model onto


def init_mix_model_and_tokenizer():
    start_time = time.time()
    tokenizer = AutoTokenizer.from_pretrained("mistralai/Mistral-7B-Instruct-v0.2")
    model = AutoModelForCausalLM.from_pretrained("mistralai/Mistral-7B-Instruct-v0.2")

    end_time = time.time()
    print("load model successful: " + str(end_time - start_time))
    return model, tokenizer


def mix_chat(model, tokenizer, context):
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


def generate_mix_datas(file_path, model, tokenizer):
    # prompt_template = 'please continue to write the following content: '
    prompt_template = 'Please rewrite the following paragraph so that it is as different as possible from the original text without changing its meaning'
    json_objs = load_arxiv_datas(file_path)
    print(file_path)
    existed_lines = 0
    out_file_name = file_path.split('/')[-1] + '.rewrite'
    try:
        with open('./data/' + out_file_name, 'r', encoding='utf-8') as out_f:
            for line in out_f:
                existed_lines += 1
    except Exception as e:
        pass
    with open('./data/' + out_file_name, 'a', encoding='utf-8') as out_f:
        i = 0
        for json_obj in tqdm(json_objs):
            i+=1
            if i < existed_lines:
                continue
            json_obj['ai_rewrite'] = mix_chat(model, tokenizer, prompt_template + json_obj['content'])
            out_f.write(json.dumps(json_obj) + '\n')


def generate_mix_paraphase_datas(file_path, model, tokenizer):
    prompt_template = "Please rewrite the following AI-generated text to make it more like human text, {without any useless content}:  "

    json_objs = load_arxiv_datas(file_path)
    print(file_path)
    existed_lines = 0
    out_file_name = file_path.split('/')[-1] + '.paraphase'
    try:
        with open('./data/' + out_file_name, 'r', encoding='utf-8') as out_f:
            for line in out_f:
                existed_lines += 1
    except Exception as e:
        pass
    with open('./data/' + out_file_name, 'a', encoding='utf-8') as out_f:
        i = 0
        for json_obj in tqdm(json_objs):
            i += 1
            if i < existed_lines:
                continue
            json_obj['ai_paraphase'] = mix_chat(model, tokenizer, prompt_template + json_obj['content'])
            out_f.write(json.dumps(json_obj) + '\n')


def init_qwen_model_and_tokenizer():
    model = AutoModelForCausalLM.from_pretrained(
        "Qwen/Qwen1.5-7B-Chat",
        torch_dtype="auto",
        device_map="auto"
    )
    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen1.5-7B-Chat")

    return model, tokenizer


def chat_qwen(model, tokenizer, content):
    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": content}
    ]
    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )
    model_inputs = tokenizer([text], return_tensors="pt").to(device)

    generated_ids = model.generate(
        model_inputs.input_ids,
        max_new_tokens=512
    )
    generated_ids = [
        output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
    ]

    response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]

    return response


def generate_qwen_datas(file_path, model, tokenizer):
    # prompt_template = 'please continue to write the following content: '
    prompt_template = 'Please rewrite the following paragraph so that it is as different as possible from the original text without changing its meaning'
    json_objs = load_arxiv_datas(file_path)
    print(file_path)
    existed_lines = 0
    out_file_name = file_path.split('/')[-1] + '.qwen' + '.rewrite'
    try:
        with open('./data/' + out_file_name, 'r', encoding='utf-8') as out_f:
            for line in out_f:
                existed_lines += 1
    except Exception as e:
        pass
    with open('./data/' + out_file_name , 'a', encoding='utf-8') as out_f:
        i = 0
        for json_obj in tqdm(json_objs):
            i+=1
            if i < existed_lines:
                continue
            json_obj['ai_rewrite'] = chat_qwen(model, tokenizer, prompt_template + json_obj['content'])
            out_f.write(json.dumps(json_obj) + '\n')


def init_glm_model_and_tokenizer():
    tokenizer = AutoTokenizer.from_pretrained("THUDM/chatglm3-6b", trust_remote_code=True)
    model = AutoModel.from_pretrained("THUDM/chatglm3-6b", trust_remote_code=True).half().cuda()
    model = model.eval()

    return model, tokenizer


def chat_glm(model, tokenizer, message):
    response, history = model.chat(tokenizer, message, history=[])
    return response



def generate_glm_datas(file_path, model, tokenizer):
    # prompt_template = 'please continue to write the following content: '
    prompt_template = 'Please rewrite the following paragraph so that it is as different as possible from the original text without changing its meaning'
    json_objs = load_arxiv_datas(file_path)
    print(file_path)
    existed_lines = 0
    out_file_name = file_path.split('/')[-1] + '.glm' + '.rewrite'
    try:
        with open('./data/' + out_file_name, 'r', encoding='utf-8') as out_f:
            for line in out_f:
                existed_lines += 1
    except Exception as e:
        pass
    with open('./data/' + out_file_name , 'a', encoding='utf-8') as out_f:
        i = 0
        for json_obj in tqdm(json_objs):
            i+=1
            if i < existed_lines:
                continue
            json_obj['ai_rewrite'] = chat_glm(model, tokenizer, prompt_template + json_obj['content'])
            out_f.write(json.dumps(json_obj) + '\n')



if __name__ == '__main__':
    # model, tokenizer = init_mix_model_and_tokenizer()
    # generate_mix_datas('../../data_collector/row_data/arxiv_paras/7.jsonl', model, tokenizer)
    # generate_mix_datas('../../data_collector/row_data/arxiv_paras/8.jsonl', model, tokenizer)
    # generate_mix_datas('../../data_collector/row_data/arxiv_paras/9.jsonl', model, tokenizer)
    # generate_mix_datas('../../data_collector/row_data/arxiv_paras/10.jsonl', model, tokenizer)
    # generate_mix_datas('../../data_collector/row_data/arxiv_paras/11.jsonl', model, tokenizer)

    # generate_mix_paraphase_datas('./data/7.jsonl', model, tokenizer)
    # generate_mix_paraphase_datas('./data/8.jsonl', model, tokenizer)
    # generate_mix_paraphase_datas('./data/9.jsonl', model, tokenizer)
    # generate_mix_paraphase_datas('./data/10.jsonl', model, tokenizer)
    #
    # generate_mix_paraphase_datas('./data/7.jsonl.rewrite', model, tokenizer)
    # generate_mix_paraphase_datas('./data/8.jsonl.rewrite', model, tokenizer)
    # generate_mix_paraphase_datas('./data/9.jsonl.rewrite', model, tokenizer)
    # generate_mix_paraphase_datas('./data/10.jsonl.rewrite', model, tokenizer)

    # model, tokenizer = init_qwen_model_and_tokenizer()
    #
    # generate_qwen_datas('./data/7.jsonl', model, tokenizer)
    # generate_qwen_datas('./data/8.jsonl', model, tokenizer)
    # generate_qwen_datas('./data/9.jsonl', model, tokenizer)
    # generate_qwen_datas('./data/10.jsonl', model, tokenizer)


    model, tokenizer = init_glm_model_and_tokenizer()
    generate_glm_datas('./data/7.jsonl', model, tokenizer)
    generate_glm_datas('./data/8.jsonl', model, tokenizer)
    generate_glm_datas('./data/9.jsonl', model, tokenizer)
    generate_glm_datas('./data/10.jsonl', model, tokenizer)


    pass







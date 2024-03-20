import gc
import json
import os
import random
import time

from datasets import load_dataset
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoModel

import json
import os
import time

import torch
from nltk import sent_tokenize
from tqdm import tqdm
from transformers import T5Tokenizer, T5ForConditionalGeneration


class DipperParaphraser(object):
    def __init__(self, model="kalpeshk2011/dipper-paraphraser-xxl", verbose=True):
        time1 = time.time()
        self.tokenizer = T5Tokenizer.from_pretrained('google/t5-v1_1-xxl')
        self.model = T5ForConditionalGeneration.from_pretrained(model, load_in_4bit=True, low_cpu_mem_usage=True)
        if verbose:
            print(f"{model} model loaded in {time.time() - time1}")
        # self.model.cuda()
        self.model.eval()

    def paraphrase(self, input_text, lex_diversity, order_diversity, prefix="", sent_interval=3, **kwargs):
        """Paraphrase a text using the DIPPER model.

        Args:
            input_text (str): The text to paraphrase. Make sure to mark the sentence to be paraphrased between <sent> and </sent> blocks, keeping space on either side.
            lex_diversity (int): The lexical diversity of the output, choose multiples of 20 from 0 to 100. 0 means no diversity, 100 means maximum diversity.
            order_diversity (int): The order diversity of the output, choose multiples of 20 from 0 to 100. 0 means no diversity, 100 means maximum diversity.
            **kwargs: Additional keyword arguments like top_p, top_k, max_length.
        """
        assert lex_diversity in [0, 20, 40, 60, 80, 100], "Lexical diversity must be one of 0, 20, 40, 60, 80, 100."
        assert order_diversity in [0, 20, 40, 60, 80, 100], "Order diversity must be one of 0, 20, 40, 60, 80, 100."

        lex_code = int(100 - lex_diversity)
        order_code = int(100 - order_diversity)

        input_text = " ".join(input_text.split())
        sentences = sent_tokenize(input_text)
        prefix = " ".join(prefix.replace("\n", " ").split())
        output_text = ""

        for sent_idx in range(0, len(sentences), sent_interval):
            curr_sent_window = " ".join(sentences[sent_idx:sent_idx + sent_interval])
            final_input_text = f"lexical = {lex_code}, order = {order_code}"
            if prefix:
                final_input_text += f" {prefix}"
            final_input_text += f" <sent> {curr_sent_window} </sent>"

            final_input = self.tokenizer([final_input_text], return_tensors="pt")
            final_input = {k: v.cuda() for k, v in final_input.items()}

            with torch.inference_mode():
                outputs = self.model.generate(**final_input, **kwargs)
            outputs = self.tokenizer.batch_decode(outputs, skip_special_tokens=True)
            prefix += " " + outputs[0]
            output_text += " " + outputs[0]

        return output_text


def dp_paraphase(dp, input_text, lex_diversity=20):
    prompt = "Please rewrite the following AI-generated text to make it more like human text, {without any useless content}:"
    output_l60_sample = dp.paraphrase(input_text, lex_diversity=lex_diversity, order_diversity=0, prefix=prompt, do_sample=True,
                                      top_p=0.75, top_k=None, max_length=512)

    return output_l60_sample

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


def my_dpo_chat(model, tokenizer, context):
    messages = [
        {"role": "user", "content": context}
    ]

    encodeds = tokenizer.apply_chat_template(messages, return_tensors="pt")

    model_inputs = encodeds.to(device)

    generated_ids = model.generate(model_inputs, max_new_tokens=1000, do_sample=True,
                                   pad_token_id=tokenizer.eos_token_id)
    decoded = tokenizer.batch_decode(generated_ids)
    return decoded[0].split('[/INST]')[1].replace('</s>', '')



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
            json_obj['ai'] = mix_chat(model, tokenizer, prompt_template + json_obj['content'])
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
            json_obj['ai'] = mix_chat(model, tokenizer, prompt_template + json_obj['content'])
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
            json_obj['ai'] = chat_qwen(model, tokenizer, prompt_template + json_obj['content'])
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


# 对抗生成如下：
def generate_qwen_paraphase_datas(file_path, model, tokenizer):
    print(file_path)
    prompt_template = "Please rewrite the following AI-generated text to make it more like human text, {without any useless content}: "
    with open(file_path, 'r', encoding='utf-8') as in_f, open(file_path + '.qwen.paraphase.jsonl', 'w', encoding='utf-8') as out_f:
        json_objs = []
        for line in in_f:
            json_objs.append(json.loads(line))
        for json_obj in tqdm(json_objs):
            json_obj['ai_rewrite'] = chat_qwen(model, tokenizer, prompt_template + json_obj['ai'])
            out_f.write(json.dumps(json_obj) + '\n')


def generate_dp_paraphase_datas(file_path, dp):
    print(file_path)
    with open(file_path, 'r', encoding='utf-8') as in_f, open(file_path + '.dp.paraphase.jsonl', 'w',
                                                              encoding='utf-8') as out_f:
        json_objs = []
        for line in in_f:
            json_objs.append(json.loads(line))
        for json_obj in tqdm(json_objs):
            json_obj['ai_rewrite'] = dp_paraphase(dp, json_obj['ai'])
            out_f.write(json.dumps(json_obj) + '\n')


def load_my_paraphase_model(model_name="mistralai/Mistral-7B-Instruct-v0.2", peft_path='./tmp.pt/checkpoint-1600'):
    all_begin_time = time.time()

    model = AutoModelForCausalLM.from_pretrained(peft_path,
                                                 # quantization_config=bnb_config,
                                                 low_cpu_mem_usage=True,
                                                 torch_dtype=torch.float16,
                                                 load_in_4bit=True,
                                                 trust_remote_code=True)
    print("load model success: " + str(time.time() - all_begin_time))

    begin_time = time.time()
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    print("load tokenizer success: " + str(time.time() - begin_time))

    print("load all success: " + str(time.time() - all_begin_time))
    return model, tokenizer

def generate_my_paraphase_datas(file_path, model, tokenizer):
    print(file_path)
    prompt_template = "Please rewrite the following AI-generated text to make it more like human text, {without any useless content}: "
    with open(file_path, 'r', encoding='utf-8') as in_f, open(file_path + '.dpo.paraphase.jsonl', 'w', encoding='utf-8') as out_f:
        json_objs = []
        for line in in_f:
            json_objs.append(json.loads(line))
        for json_obj in tqdm(json_objs):
            json_obj['ai_rewrite'] = my_dpo_chat(model, tokenizer, prompt_template + json_obj['ai'])
            out_f.write(json.dumps(json_obj) + '\n')

    pass


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


    # model, tokenizer = init_glm_model_and_tokenizer()
    # generate_glm_datas('./data/7.jsonl', model, tokenizer)
    # generate_glm_datas('./data/8.jsonl', model, tokenizer)
    # generate_glm_datas('./data/9.jsonl', model, tokenizer)
    # generate_glm_datas('./data/10.jsonl', model, tokenizer)

    # qwen_model, qwen_tokenizer = init_qwen_model_and_tokenizer()
    #
    # generate_qwen_paraphase_datas('./data/nature/qwen/7.jsonl.qwen.rewrite.jsonl', qwen_model, qwen_tokenizer)
    # generate_qwen_paraphase_datas('./data/nature/qwen/8.jsonl.qwen.rewrite.jsonl', qwen_model, qwen_tokenizer)
    # generate_qwen_paraphase_datas('./data/nature/qwen/9.jsonl.qwen.rewrite.jsonl', qwen_model, qwen_tokenizer)
    # generate_qwen_paraphase_datas('./data/nature/qwen/10.jsonl.qwen.rewrite.jsonl', qwen_model, qwen_tokenizer)
    # del qwen_tokenizer, qwen_model
    # gc.collect()  # 执行垃圾回收
    # torch.cuda.empty_cache()  # 清空CUDA缓存，释放GPU内存
    #
    # dp = DipperParaphraser()
    # generate_dp_paraphase_datas('./data/nature/qwen/7.jsonl.qwen.rewrite.jsonl', dp)
    # generate_dp_paraphase_datas('./data/nature/qwen/8.jsonl.qwen.rewrite.jsonl', dp)
    # generate_dp_paraphase_datas('./data/nature/qwen/9.jsonl.qwen.rewrite.jsonl', dp)
    # generate_dp_paraphase_datas('./data/nature/qwen/10.jsonl.qwen.rewrite.jsonl', dp)
    # del dp
    # gc.collect()  # 执行垃圾回收
    # torch.cuda.empty_cache()  # 清空CUDA缓存，释放GPU内存

    my_model, my_tokenizer = load_my_paraphase_model(peft_path='../dpo_test/dpo_1/1/final_checkpoint/')
    generate_my_paraphase_datas('./data/nature/qwen/7.jsonl.qwen.rewrite.jsonl', my_model, my_tokenizer)
    generate_my_paraphase_datas('./data/nature/qwen/8.jsonl.qwen.rewrite.jsonl', my_model, my_tokenizer)
    generate_my_paraphase_datas('./data/nature/qwen/9.jsonl.qwen.rewrite.jsonl', my_model, my_tokenizer)
    generate_my_paraphase_datas('./data/nature/qwen/10.jsonl.qwen.rewrite.jsonl', my_model, my_tokenizer)

    pass







import datetime
import json
import os
import random
import time
import math

import pandas as pd
from sklearn.model_selection import train_test_split
from torch import nn
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM
from torch.utils.data import Dataset, DataLoader

import torch

from test_base import init_test_model_and_tokenizer
from train_base import MyClassifier


def init_model_and_tokenizer(model_name="mistralai/Mistral-7B-Instruct-v0.2"):
    start_time = time.time()
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.float16)

    end_time = time.time()
    print("load adversal generator model successful: " + str(end_time - start_time))
    return model, tokenizer


def chat(model, tokenizer, context, max_new_tokens=512, device='cuda'):
    messages = [
        {"role": "user", "content": context}
    ]

    encodeds = tokenizer.apply_chat_template(messages, return_tensors="pt")
    model_inputs = encodeds.to(device)
    model.to(device)

    generated_ids = model.generate(model_inputs, max_new_tokens=max_new_tokens, do_sample=True,
                                   pad_token_id=tokenizer.eos_token_id)
    decoded = tokenizer.batch_decode(generated_ids)
    return decoded[0].split('[/INST]')[1].replace('</s>', '')


class MyAdversaryDataset(Dataset):
    def __init__(self, datas: list, tokenizer):

        texts = [x['content'] for x in datas]
        # print("begin tokenize datas")
        self.texts = [tokenizer(text, padding='max_length',
                                max_length=256,
                                truncation=True,
                                return_tensors="pt")
                      for text in texts]

        # print("end tokenize datas")
        self.labels = [x['label'] for x in datas]

        self.domains = []
        self.prompts = []
        for x in datas:
            try:
                if x['domain'] is None:
                    self.domains.append('default')
                else:
                    self.domains.append(x['domain'])
            except Exception as e:
                self.domains.append('default')
            try:
                if x['prompt'] is None:
                    self.prompts.append('default')
                else:
                    self.prompts.append(x['prompt'])
            except Exception as e:
                self.prompts.append('default')

    def __getitem__(self, idx):
        text = self.texts[idx]
        label = self.labels[idx]

        return text, label

    def __len__(self):
        return min(len(self.texts), len(self.labels))


def load_local_file_data_by_prompt_map(prompt_file_map):
    prompt_jsons_map = {}
    for prompt in prompt_file_map:
        prompt_file = prompt_file_map[prompt]
        with open(prompt_file, 'r', encoding='utf-8') as input_file:
            prompt_jsons = []
            for line in input_file:
                json_obj = json.loads(line)
                prompt_jsons.append({
                    'label': 0,
                    'content': json_obj['human']
                })
                prompt_jsons.append({
                    'label': 1,
                    'content': json_obj['ai']
                })
            prompt_jsons_map[prompt] = prompt_jsons

    return prompt_jsons_map


def get_adversary_data_loader(adversary_data: MyAdversaryDataset, batch_size=16, shuffle=False):
    dataloader = DataLoader(adversary_data, batch_size=batch_size, shuffle=shuffle)
    return dataloader


def get_adversary_prompts():
    prompts_map = {
        'rewrite': 'Please rewrite the following content, {without any useless content}:',
        'continue': 'Please continue to write the following content, {without any useless content}:',
        'easy': 'Please change the following content to make it easier to understand, {without any useless content}:',
        'academic': 'Please change the following content to be more academic and professional, {without any useless content}:',
        'difficult': 'Please change the following content to make it more difficult to understand, {without any useless content}:',
    }
    return prompts_map


def adversary_val(train_model: MyClassifier, prompt_dataloader_map, device='cuda'):
    criterion = nn.BCELoss()
    prompt_loss_map = {}
    for prompt in prompt_dataloader_map:
        prompt_loss = 0
        for train_input, train_label in tqdm(prompt_dataloader_map[prompt]):
            attention_mask = train_input['attention_mask'].to(device)
            input_ids = train_input['input_ids'].squeeze(1).to(device)
            train_label = train_label.to(device)
            output = train_model(input_ids, attention_mask)
            loss = criterion(output, train_label.float().unsqueeze(1))

            prompt_loss += loss.item()
        prompt_loss_map[prompt] = prompt_loss
    return prompt_loss_map




class AdversaryGenerator:

    def init_adversary_model_and_tokenizer(self, model_name="mistralai/Mistral-7B-Instruct-v0.2"):
        start_time = time.time()
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.float16)

        end_time = time.time()
        print("load adversal generator model successful: " + str(end_time - start_time))
        return model, tokenizer

    # 默认的文件加载函数
    def default_file_load_func(self, file):
        file_objs = []
        print("begin load local file: " + file)
        with open(file, 'r', encoding='utf-8') as in_f:
            for line in in_f:
                json_obj = json.loads(line)
                human_obj = {
                    'label': 0,
                    'content': json_obj['human']
                }
                ai_obj = {
                    'label': 1,
                    'content': json_obj['ai']
                }
                file_objs.append(human_obj)
                file_objs.append(ai_obj)
        print("load local file successful: " + file)
        print("data nums: " + str(len(file_objs)))
        return file_objs

    # 初始化分类好的验证集，模型等参数
    def __init__(self,  local_val_files_map, prompts_map, train_df, file_load_func=default_file_load_func, tmp_data_path='./tmp_ad/', lazy_init=False):
        print("begin init adversary generator")
        begin_time = time.time()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.lazy_init = lazy_init
        if not lazy_init:
            chat_model, chat_tokenizer = self.init_adversary_model_and_tokenizer()
            self.chat_model = chat_model
            self.chat_tokenizer = chat_tokenizer
        else:
            self.chat_model = None
            self.chat_tokenizer = None

        self.tmp_data_path = tmp_data_path
        # Check if the directory exists
        if not os.path.exists(tmp_data_path):
            # Create the directory
            os.makedirs(tmp_data_path)
            print("Directory created successfully: " + tmp_data_path)
        else:
            print("Directory already exists: " + tmp_data_path)

        self.prompts_map = prompts_map

        train_texts = [x['content'] for i, x in train_df.iterrows()]
        train_labels = [x['label'] for i, x in train_df.iterrows()]
        self.train_datas = []
        for i in range(0, len(train_texts)):
            self.train_datas.append({
                'label': train_labels[i],
                'content': train_texts[i]
            })

        self.val_data_map = {}
        for key in local_val_files_map:
            self.val_data_map[key] = []

        if file_load_func is not None:
            for key in local_val_files_map:
                file = local_val_files_map[key]
                file_objs = file_load_func(self, file)
                self.val_data_map[key] += file_objs

        end_time = time.time()
        print("end init adversary generator: " + str(end_time - begin_time))

    # 将当前训练的模型进行评估，评估不同类的分类loss
    def val_cur_train_model_by_local_datas(self, cur_train_model: MyClassifier, cur_train_tokenizer, eval_nums=16):
        criterion = nn.BCELoss()
        key_loss_map = {}
        device = self.device
        for key in self.val_data_map:
            key_loss = 0
            key_eval_datas = self.val_data_map[key][0: eval_nums]
            key_adversary_dataset = MyAdversaryDataset(key_eval_datas, cur_train_tokenizer)
            key_adversary_dataloader = get_adversary_data_loader(adversary_data=key_adversary_dataset, batch_size=eval_nums)
            for train_input, train_label in tqdm(key_adversary_dataloader):
                attention_mask = train_input['attention_mask'].to(device)
                input_ids = train_input['input_ids'].squeeze(1).to(device)
                train_label = train_label.to(device)
                output = cur_train_model(input_ids, attention_mask)
                loss = criterion(output, train_label.float().unsqueeze(1))

                key_loss += loss.item()
            key_loss_map[key] = key_loss
        return key_loss_map

    def calculate_adversary_data_num_by_loss(self, loss_map):
        batch_num = 16

        log_loss_map = {}
        log_loss_sum = 0
        log_loss_percent_map = {}

        for k in loss_map:
            log_loss_map[k] = (-math.log10(1 - loss_map[k]))
            log_loss_sum += log_loss_map[k]

        for k in log_loss_map:
            log_loss_percent_map[k] = log_loss_map[k] / log_loss_sum

        sorted_dict = dict(sorted(log_loss_percent_map.items(), key=lambda item: item[1], reverse=True))
        sorted_list = [[k, sorted_dict[k]] for k in sorted_dict]

        adversary_nums_map = {}

        for key in loss_map:
            adversary_nums_map[key] = 1

        for kv in sorted_list:
            adversary_nums_map[kv[0]] += int(0.5 + kv[1] * (batch_num - len(loss_map)))

        return adversary_nums_map

    def generate_adversary_train_data_by_val_result(self, key_loss_map, batch_size=16):
        begin_time = time.time()

        if self.lazy_init:
            if self.chat_model is None or self.chat_tokenizer is None:
                self.chat_model, self.chat_tokenizer = self.init_adversary_model_and_tokenizer()
                self.lazy_init = False

        adversary_nums_map = self.calculate_adversary_data_num_by_loss(key_loss_map)

        prompts_map = self.prompts_map
        tmp_data_path = self.tmp_data_path
        save_file_name = 'adversary_train_data' + str(datetime.datetime.now()) + '.jsonl'

        row_train_datas = [x for x in self.train_datas if x['label'] == 1][0: int(batch_size*1.5)]
        row_data_index = 0
        random.shuffle(row_train_datas)

        adversary_train_datas = []

        for key in adversary_nums_map:
            print('generate process : [%d/%d]' % (row_data_index, batch_size), end='\r')
            prompt_template = prompts_map[key]
            for i in range(0, adversary_nums_map[key]):
                row_train_data = row_train_datas[row_data_index]
                chat_text = prompt_template + row_train_data['content']
                chat_result = self.generate_chat_context(chat_text)
                adversary_train_datas.append({
                    'label': 1,
                    'content': chat_result,
                    'prompt': key
                })
                row_data_index += 1

        print('generate finish: ' + str(time.time() - begin_time))

        random.shuffle(adversary_train_datas)
        # 截断
        adversary_train_datas = adversary_train_datas[0: batch_size]

        with open(tmp_data_path + save_file_name, 'w', encoding='utf-8') as out_f:
            for data in adversary_train_datas:
                out_f.write(json.dumps(data) + '\n')

        return adversary_train_datas


    def generate_chat_context(self, text):
        return chat(self.chat_model, self.chat_tokenizer, text)


if __name__ == '__main__':
    local_val_file_map = {
        'rewrite': '../../data_collector/test_data/hc3_english_mix_multi/open_qa.rewrite.mix.jsonl',
        'continue': '../../data_collector/test_data/hc3_english_mix_multi/open_qa.continue.mix.jsonl',
        'academic': '../../data_collector/test_data/hc3_english_mix_multi/open_qa.academic.mix.jsonl',
        'difficult': '../../data_collector/test_data/hc3_english_mix_multi/open_qa.difficult.mix.jsonl',
        'easy': '../../data_collector/test_data/hc3_english_mix_multi/open_qa.easy.mix.jsonl',
        'qa': '../../data_collector/test_data/hc3_english_mix_multi/open_qa.mix.jsonl',
    }
    prompts_map = {
        'rewrite': 'Please rewrite the following content, {without any useless content}:',
        'continue': 'Please continue to write the following content, {without any useless content}:',
        'easy': 'Please change the following content to make it easier to understand, {without any useless content}:',
        'academic': 'Please change the following content to be more academic and professional, {without any useless content}:',
        'difficult': 'Please change the following content to make it more difficult to understand, {without any useless content}:',
        'qa': 'The following is a response to a question, please re-answer the question based on this response, {without any useless content}:'
    }

    train_file = pd.read_json("../Deberta_test/data/hc3_all.jsonl.train")
    train_df, val_df = train_test_split(train_file, test_size=0.2, random_state=0)

    adversary_generator = AdversaryGenerator(
        local_val_files_map=local_val_file_map,
        prompts_map=prompts_map,
        train_df=train_df
    )

    train_model, train_tokenizer = init_test_model_and_tokenizer(test_model_path='hc3_row.pt')

    loss_map = adversary_generator.val_cur_train_model_by_local_datas(cur_train_model=train_model, cur_train_tokenizer=train_tokenizer)
    print(loss_map)

    data_num_map = adversary_generator.calculate_adversary_data_num_by_loss(loss_map)
    print(data_num_map)

    print(adversary_generator.generate_adversary_train_data_by_val_result(loss_map))



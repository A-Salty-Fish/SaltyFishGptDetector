import json
import time

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
        print("begin tokenize datas")
        self.texts = [tokenizer(text, padding='max_length',
                                max_length=256,
                                truncation=True,
                                return_tensors="pt")
                      for text in texts]

        print("end tokenize datas")
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
    def __init__(self, chat_model, chat_tokenizer, local_val_files_map, file_load_func=default_file_load_func):
        print("begin init adversary generator")
        begin_time = time.time()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.chat_model = chat_model
        self.chat_tokenizer = chat_tokenizer

        self.val_data_map = {}
        for key in local_val_files_map:
            self.val_data_map[key] = []

        if file_load_func is not None:
            for key in local_val_files_map:
                file = local_val_files_map[key]
                file_objs = file_load_func(file)
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

    def generate_chat_context(self, text):
        return chat(self.chat_model, self.chat_tokenizer, text)


if __name__ == '__main__':
    model, tokenizer = init_model_and_tokenizer()
    local_val_file_map = {
        'rewrite': '../../data_collector/test_data/hc3_english_mix_multi/open_qa.rewrite.mix.jsonl',
        'continue': '../../data_collector/test_data/hc3_english_mix_multi/open_qa.continue.mix.jsonl',
        'academic': '../../data_collector/test_data/hc3_english_mix_multi/open_qa.academic.mix.jsonl',
        'difficult': '../../data_collector/test_data/hc3_english_mix_multi/open_qa.difficult.mix.jsonl',
        'easy': '../../data_collector/test_data/hc3_english_mix_multi/open_qa.easy.mix.jsonl',
        'qa': '../../data_collector/test_data/hc3_english_mix_multi/open_qa.mix.jsonl',
    }
    adversary_generator = AdversaryGenerator(model, tokenizer, local_val_file_map)

    train_model, train_tokenizer = init_test_model_and_tokenizer(test_model_path='hc3_row.pt')

    print(adversary_generator.val_cur_train_model_by_local_datas(cur_train_model=train_model, cur_train_tokenizer=train_tokenizer))

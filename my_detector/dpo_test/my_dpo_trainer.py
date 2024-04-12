import functools
import gc
import json
import os
import random
import time
from typing import Dict

import datasets
import numpy as np
import pandas as pd
import torch
from bleurt_pytorch import BleurtForSequenceClassification, BleurtTokenizer, BleurtConfig
from peft import LoraConfig
from sklearn.model_selection import train_test_split
from torch import nn
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments, BitsAndBytesConfig, AutoModel

from trl import DPOTrainer
from torch.optim import Adam
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence

peft_config = LoraConfig(
    target_modules=[
        "q_proj",
        "v_proj",
        "k_proj",
        "out_proj",
        "fc_in",
        "fc_out",
        "wte",
    ],
    bias="none",
    task_type="CAUSAL_LM",
)

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
)


def load_trainer_args(output_dir='./tmp'):
    # CUDA_VISIBLE_DEVICES=5,6 accelerate launch --multi_gpu DPO_trl.py \
    # CUDA_VISIBLE_DEVICES=5 python DPO_trl.py \
    # --per_device_train_batch_size 1 \
    # --gradient_accumulation_steps 1 \
    # --num_train_epochs 1 \
    # --save_steps 500 \
    # --save_total_limit 5 \
    # --learning_rate 5e-7 \
    # --seed 42 \
    # --ddp_find_unused_parameters=False \
    # --remove_unused_columns false \
    # --logging_steps 10 \
    # --output_dir ./weights/DPO_BC
    train_args = TrainingArguments(output_dir=output_dir)
    # train_args.gradient_accumulation_steps = 1
    train_args.num_train_epochs = 5
    train_args.save_steps = 200
    train_args.save_total_limit = 2
    # train_args.learning_rate = 5e-4
    train_args.seed = 42
    train_args.ddp_find_unused_parameters = False
    train_args.remove_unused_columns = False
    train_args.logging_steps = 500
    train_args.per_device_train_batch_size = 2
    train_args.per_device_eval_batch_size = 1
    return train_args


def load_generator_train_model(model_name="mistralai/Mistral-7B-Instruct-v0.2",
                               model_path='./hc3_all_1/final_checkpoint', quantization_config=bnb_config):
    all_begin_time = time.time()

    begin_time = time.time()
    model = AutoModelForCausalLM.from_pretrained(model_path,
                                                 # quantization_config=quantization_config,
                                                 low_cpu_mem_usage=True,
                                                 torch_dtype=torch.float16,
                                                 load_in_4bit=True,
                                                 # is_trainable=True,
                                                 trust_remote_code=True)
    print("load model success: " + str(time.time() - begin_time))

    begin_time = time.time()
    # ref_model = AutoModelForCausalLM.from_pretrained(model_name,
    #                                                  # quantization_config = quantization_config,
    #                                                  low_cpu_mem_usage=True,
    #                                                  torch_dtype=torch.float16,
    #                                                  load_in_4bit=True,
    #                                                  trust_remote_code=True)
    print("load ref_model success: " + str(time.time() - begin_time))

    begin_time = time.time()
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token
    print("load tokenizer success: " + str(time.time() - begin_time))

    print("load all success: " + str(time.time() - all_begin_time))
    return model, None, tokenizer


def tokenize_batch_element(prompt: str, chosen: str, rejected: str, truncation_mode: str, tokenizer, max_length: int,
                           max_prompt_length: int):
    chosen_tokens = tokenizer(chosen, add_special_tokens=False)
    rejected_tokens = tokenizer(rejected, add_special_tokens=False)
    prompt_tokens = tokenizer(prompt, add_special_tokens=False)

    # print(tokenizer.eos_token_id)
    # print(tokenizer.im_end_id)

    end_id = tokenizer.eos_token_id if tokenizer.eos_token_id is not None else tokenizer.im_end_id
    # Or, depending on your model.

    assert end_id not in prompt_tokens['input_ids'], f"Prompt contains EOS token: {prompt}"
    assert end_id not in chosen_tokens['input_ids'], f"Chosen response contains EOS token: {chosen}"
    assert end_id not in rejected_tokens['input_ids'], f"Rejected response contains EOS token: {rejected}"

    chosen_tokens['input_ids'].append(end_id)
    chosen_tokens['attention_mask'].append(1)

    rejected_tokens['input_ids'].append(end_id)
    rejected_tokens['attention_mask'].append(1)

    longer_response_length = max(len(chosen_tokens['input_ids']), len(rejected_tokens['input_ids']))

    # if combined sequence is too long, truncate the prompt
    if len(prompt_tokens['input_ids']) + longer_response_length > max_length:
        if truncation_mode == 'keep_start':
            prompt_tokens = {k: v[:max_prompt_length] for k, v in prompt_tokens.items()}
        elif truncation_mode == 'keep_end':
            prompt_tokens = {k: v[-max_prompt_length:] for k, v in prompt_tokens.items()}
        else:
            raise ValueError(f'Unknown truncation mode: {truncation_mode}')

    # if that's still too long, truncate the response
    if len(prompt_tokens['input_ids']) + longer_response_length > max_length:
        chosen_tokens = {k: v[:max_length - max_prompt_length] for k, v in chosen_tokens.items()}
        rejected_tokens = {k: v[:max_length - max_prompt_length] for k, v in rejected_tokens.items()}

    # Create labels
    chosen_sequence_tokens = {k: prompt_tokens[k] + chosen_tokens[k] for k in chosen_tokens}
    rejected_sequence_tokens = {k: prompt_tokens[k] + rejected_tokens[k] for k in rejected_tokens}
    chosen_sequence_tokens['labels'] = chosen_sequence_tokens['input_ids'][:]
    chosen_sequence_tokens['labels'][:len(prompt_tokens['input_ids'])] = [-100] * len(prompt_tokens['input_ids'])
    rejected_sequence_tokens['labels'] = rejected_sequence_tokens['input_ids'][:]
    rejected_sequence_tokens['labels'][:len(prompt_tokens['input_ids'])] = [-100] * len(prompt_tokens['input_ids'])

    batch = {}

    batch['prompt'] = prompt
    batch['chosen'] = prompt + chosen
    batch['rejected'] = prompt + rejected
    batch['chosen_response_only'] = chosen
    batch['rejected_response_only'] = rejected

    for k, toks in {'chosen': chosen_sequence_tokens, 'rejected': rejected_sequence_tokens,
                    'prompt': prompt_tokens}.items():
        for type_key, tokens in toks.items():
            if type_key == 'token_type_ids':
                continue
            batch[f'{k}_{type_key}'] = tokens

    return batch


def load_dataset(data_path, tokenizer, prompt_key='prompt', accept_key='accept', reject_key='reject'):
    data = []
    with open(data_path, 'r', encoding='utf-8') as in_f:
        for line in in_f:
            obj = json.loads(line)
            data.append(obj)

    prompt = [D[prompt_key] for D in data]
    accept = [D[accept_key] for D in data]
    reject = [D[reject_key] for D in data]

    for P, A, R in zip(prompt, accept, reject):
        tokenize_data = tokenize_batch_element(P, chosen=A, rejected=R, truncation_mode="keep_start",
                                               tokenizer=tokenizer, max_length=512, max_prompt_length=512)
        # tokenize_data = collate_fn([tokenize_data], tokenizer)
        # print(tokenize_data)
        yield tokenize_data


# 以下为获取对抗样本所需的能力
def get_text_predictions(model, tokenizer, texts, bar=0.5):
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")

    model = model.to(device)

    results_predictions = []

    data_inputs = []
    for text in texts:
        data_inputs.append(
            tokenizer(text, padding='max_length',
                      max_length=512,
                      truncation=True,
                      return_tensors="pt")
        )

    with torch.no_grad():
        model.eval()
        for data_input in data_inputs:
            # print(data_input)
            attention_mask = data_input['attention_mask'].to(device)
            input_ids = data_input['input_ids'].squeeze(1).to(device)

            output = model(input_ids, attention_mask)

            output = (output > bar).int()
            results_predictions.append(output)

    npmpy_results = torch.cat(results_predictions).cpu().detach().numpy()
    return [x[0] == 0 for x in npmpy_results]


def load_predict_model(model_name="roberta-base", model_path='best_model.pt'):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    # base_model = AutoModel.from_pretrained(model_name)
    model = (torch.load(model_path))
    # model = MyClassifier(base_model)
    model.eval()

    return model, tokenizer


def predict_jsonl(model, tokenizer, jsonl_file, bar=0.5):
    json_objs = []
    with open(jsonl_file, 'r', encoding='utf-8') as in_f:
        for line in in_f:
            json_obj = json.loads(line)
            json_objs.append(json_obj)
    texts = [json_obj['ai_rewrite'] for json_obj in json_objs]
    predict_results = get_text_predictions(model, tokenizer, texts, bar)
    prompt_map = {}
    for i in range(0, len(json_objs)):
        json_obj = json_objs[i]
        prompt = json_obj['prompt']
        if prompt_map.get(prompt) is None:
            prompt_map[prompt] = {
                'chosen': [],
                'rejected': []
            }
        if predict_results[i]:
            prompt_map[prompt]['chosen'].append(json_obj['ai_rewrite'])
        else:
            prompt_map[prompt]['rejected'].append(json_obj['ai_rewrite'])

    results = []
    for prompt in prompt_map:
        prompt_obj = prompt_map[prompt]
        chosens = prompt_obj['chosen']
        rejecteds = prompt_obj['rejected']
        if len(chosens) > 0 and len(rejecteds) > 0:
            if len(chosens) == len(rejecteds):
                for i in range(0, len(chosens)):
                    results.append({
                        'prompt': prompt,
                        'chosen': chosens[i],
                        'rejected': rejecteds[i]
                    })
            elif len(chosens) > len(rejecteds):
                for i in range(0, len(rejecteds)):
                    results.append({
                        'prompt': prompt,
                        'chosen': chosens[i],
                        'rejected': rejecteds[i]
                    })
                # for i in range(len(rejecteds), len(chosens)):
                #     results.append({
                #         'prompt': prompt,
                #         'chosen': chosens[i],
                #         'rejected': random.sample(rejecteds, 1)[0]
                #     })
            else:
                for i in range(0, len(chosens)):
                    results.append({
                        'prompt': prompt,
                        'chosen': chosens[i],
                        'rejected': rejecteds[i]
                    })
                # for i in range(len(chosens), len(rejecteds)):
                #     results.append({
                #         'prompt': prompt,
                #         'chosen': random.sample(chosens, 1)[0],
                #         'rejected': rejecteds[i]
                #     })
        else:
            continue
    # print(len(texts))
    # print(len(predict_results))
    # print(len([x for x in predict_results if x]))
    print(len(results))
    with open(jsonl_file + '.all', 'w', encoding='utf-8') as out_f:
        out_f.write(json.dumps(results))


def prepare_train_data():
    json_objs = []
    with open('../../data_collector/test_data/hc3_english_mix_multi/wiki_csai.mix.jsonl', 'r',
              encoding='utf-8') as in_f:
        for line in in_f:
            json_objs.append(json.loads(line))
    out_json_objs = [
        {
            'prompt': x['question'],
            'chosen': x['human'],
            'rejected': x['ai']
        }
        for x in json_objs
    ]
    with open('./data/wiki_csai.mix.train', 'w', encoding='utf-8') as out_f:
        out_f.write(json.dumps(out_json_objs))


def convert_dataset(file_path):
    new_file_path = file_path + '.conv'
    new_jsons = []
    rejected_set = []
    chosen_set = []
    with open(file_path, 'r', encoding='utf-8') as in_f:
        json_objs = json.load(in_f)
        for json_obj in json_objs:
            if json_obj['rejected'] in rejected_set:
                continue
            if json_obj['chosen'] in chosen_set:
                continue
            rejected_set.append(json_obj['rejected'])
            chosen_set.append(json_obj['chosen'])
            new_jsons.append(
                {
                    'prompt': '<s>[INST] ' + json_obj['prompt'] + ' [/INST]',
                    'chosen': '<s>[INST] ' + json_obj['prompt'] + ' [/INST]' + json_obj['chosen'] + '</s>',
                    'rejected': '<s>[INST] ' + json_obj['prompt'] + ' [/INST]' + json_obj['rejected'] + '</s>'
                }
            )
    with open(new_file_path, 'w', encoding='utf-8') as out_f:
        out_f.write(json.dumps(new_jsons))


# 以下为分类器和生成器模块
# ——————————————————————————————————————————————————————

# 需要声明判别器模型结构
class MyClassifier(nn.Module):
    def __init__(self, base_model):
        super(MyClassifier, self).__init__()

        self.bert = base_model
        self.fc1 = nn.Linear(768, 32)
        self.fc2 = nn.Linear(32, 1)

        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

    def forward(self, input_ids, attention_mask):
        bert_out = self.bert(input_ids=input_ids,
                             attention_mask=attention_mask)[0][:, 0]
        x = self.fc1(bert_out)
        x = self.relu(x)

        x = self.fc2(x)
        x = self.sigmoid(x)

        return x


# 声明生成器
class MyGenerator:
    def __init__(self, model_name, perf_path, device='cuda'):
        all_begin_time = time.time()

        model = AutoModelForCausalLM.from_pretrained(perf_path,
                                                     # quantization_config=bnb_config,
                                                     trust_remote_code=True, torch_dtype=torch.float16)
        print("load model success: " + str(time.time() - all_begin_time))

        begin_time = time.time()
        tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        print("load tokenizer success: " + str(time.time() - begin_time))

        print("load all success: " + str(time.time() - all_begin_time))
        self.model = model.to(device)
        self.tokenizer = tokenizer
        self.device = device
        self.prompt_template = "Please rewrite the following AI-generated text to make it more like human text, {without any useless content}:  "

    def chat(self, context):
        messages = [
            {"role": "user", "content": context}
        ]
        encodeds = self.tokenizer.apply_chat_template(messages, return_tensors="pt")
        model_inputs = encodeds.to(self.device)
        generated_ids = self.model.generate(model_inputs, max_new_tokens=512, do_sample=True,
                                            pad_token_id=self.tokenizer.eos_token_id)
        decoded = self.tokenizer.batch_decode(generated_ids)
        return decoded[0].split('[/INST]')[1].replace('</s>', '')

    def adversary_chat(self, row_train_text):
        return self.chat(self.prompt_template + row_train_text)

    # 实时生成
    def generate_adversary_train_data(self, row_train_texts):
        results = []
        for row_train_text in tqdm(row_train_texts):
            adversary_text = self.chat(self.prompt_template + row_train_text)
            results.append({
                'label': 1,
                'content': adversary_text,
                'row': row_train_text
            })
        return results

    # 从文件中加载
    def load_adversary_train_data(self, file_path, file_type='jsonl'):
        results = []
        if file_type == 'jsonl':
            with open(file_path, 'r', encoding='utf-8') as in_f:
                for line in in_f:
                    results.append(json.loads(line))
            return [{
                'label': 1,
                'content': x['ai_rewrite']
            } for x in results]
        elif file_type == 'json_arr':
            with open(file_path, 'r', encoding='utf-8') as in_f:
                results = json.load(in_f)
            return [{
                'label': 1,
                'content': x['ai_rewrite']
            } for x in results]
        else:
            return []


# 对生成器生成的文本进行相似度打分，避免过于不相似
import json

import nltk
import numpy as np
# nltk.download('stopwords')
# nltk.download('punkt')
# https://huggingface.co/lucadiliello/BLEURT-20
# pip install git+https://github.com/lucadiliello/bleurt-pytorch.git
# pip install rouge
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from tqdm import tqdm


class GenerateTextScorer:
    def __init__(self, need_cosine=True, need_euclidean=True, need_edit_distance=True, need_bleu=True,
                 need_rouge=False):
        import transformers
        transformers.logging.set_verbosity_error()
        if need_bleu:
            self.blue_config = BleurtConfig.from_pretrained('lucadiliello/BLEURT-20')
            self.blue_model = BleurtForSequenceClassification.from_pretrained('lucadiliello/BLEURT-20')
            self.blue_tokenizer = BleurtTokenizer.from_pretrained('lucadiliello/BLEURT-20')
            self.blue_model.eval()
            print("load bleu model")
        if need_rouge:
            from rouge import Rouge
            self.rouge = Rouge()
            print("load rouge model")
        self.need_cosine = need_cosine
        self.need_euclidean = need_euclidean
        self.need_edit_distance = need_edit_distance
        self.need_bleu = need_bleu
        self.need_rouge = need_rouge

    def preprocess_text(self, text):
        stop_words = set(stopwords.words('english'))
        word_tokens = word_tokenize(text.lower())
        filtered_text = [word for word in word_tokens if word.isalnum() and word not in stop_words]
        return " ".join(filtered_text)

    def calculate_cosine_similarity(self, text1, text2):
        preprocessed_text1 = self.preprocess_text(text1)
        preprocessed_text2 = self.preprocess_text(text2)

        vectorizer = CountVectorizer().fit_transform([preprocessed_text1, preprocessed_text2])
        vectors = vectorizer.toarray()

        cosine_sim = cosine_similarity(vectors)

        return cosine_sim[0][1]

    def calculate_euclidean_distance(self, text1, text2):
        preprocessed_text1 = self.preprocess_text(text1)
        preprocessed_text2 = self.preprocess_text(text2)

        tfidf_vectorizer = TfidfVectorizer()
        tfidf_matrix = tfidf_vectorizer.fit_transform([preprocessed_text1, preprocessed_text2])

        tfidf_array = tfidf_matrix.toarray()
        euclidean_dist = np.linalg.norm(tfidf_array[0] - tfidf_array[1])

        return euclidean_dist

    def calculate_edit_distance(self, text1, text2):
        m = len(text1)
        n = len(text2)

        # 初始化动态规划矩阵
        dp = [[0 for j in range(n + 1)] for i in range(m + 1)]

        # 初始化第一行和第一列
        for i in range(1, m + 1):
            dp[i][0] = i
        for j in range(1, n + 1):
            dp[0][j] = j

        # 计算动态规划矩阵
        for i in range(1, m + 1):
            for j in range(1, n + 1):
                if text1[i - 1] == text2[j - 1]:
                    dp[i][j] = dp[i - 1][j - 1]
                else:
                    dp[i][j] = 1 + min(dp[i - 1][j], dp[i][j - 1], dp[i - 1][j - 1])

        return dp[m][n]

    def get_bleu_score(self, text1, text2):
        model = self.blue_model
        tokenizer = self.blue_tokenizer
        references = [text1]
        candidates = [text2]
        with torch.no_grad():
            inputs = tokenizer(references, candidates, truncation=True, return_tensors='pt', max_length=512)
            res = model(**inputs).logits.flatten().tolist()
            return res[0]

    def get_rouge_score(self, text1, text2):
        return self.rouge.get_scores(text1, text2)[0]['rouge-1']['f']

    # 原始分数
    def row_score(self, text1, text2):
        # need_cosine = True, need_euclidean = True, need_edit_distance = True, need_bleu = True, need_rouge = False
        result = {}
        if self.need_cosine:
            result['cosine'] = self.calculate_cosine_similarity(text1, text2)
        if self.need_euclidean:
            result['euclidean'] = self.calculate_euclidean_distance(text1, text2)
        if self.need_edit_distance:
            result['edit_distance'] = self.calculate_edit_distance(text1, text2)
        if self.need_bleu:
            result['bleu'] = self.get_bleu_score(text1, text2)
        if self.need_rouge:
            result['rouge'] = self.get_rouge_score(text1, text2)
        result['text1'] = text1
        result['text2'] = text2
        return result

    # 加权分数
    def weighted_score(self, row_score_result):
        result = 0
        total = 0
        if self.need_cosine:
            total += 100
            result += row_score_result['cosine'] * 100
        if self.need_euclidean:
            total += 100
            euclidean_score = 100 - 100 * row_score_result['euclidean'] / 1.5
            result += min(100, euclidean_score)
        if self.need_edit_distance:
            total += 100
            edit_distance_score = 100 - (row_score_result['edit_distance']) / (
                    len(row_score_result['text1']) + len(row_score_result['text2']))
            result += min(100, edit_distance_score)
        if self.need_bleu:
            total += 100
            result += row_score_result['bleu'] * 100
        if self.need_rouge:
            total += 100
            result += row_score_result['rouge'] * 100
        return 100 * result / total


# 初始化 生成器的训练模型
def init_generator_train_model_and_tokenizer(model_name="mistralai/Mistral-7B-Instruct-v0.2"):
    all_begin_time = time.time()
    begin_time = time.time()
    model = AutoModelForCausalLM.from_pretrained(model_name,
                                                 # quantization_config=quantization_config,
                                                 low_cpu_mem_usage=True,
                                                 torch_dtype=torch.float16,
                                                 load_in_4bit=True,
                                                 trust_remote_code=True)
    print("load generator train model success: " + str(time.time() - begin_time))

    begin_time = time.time()
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    print("load generator train tokenizer success: " + str(time.time() - begin_time))

    print("load generator train all success: " + str(time.time() - all_begin_time))
    return model, None, tokenizer


# 初始化 分类器的训练模型
def init_classifier_train_model_and_tokenizer(base_model_name='roberta-base'):
    begin_time = time.time()
    torch.manual_seed(0)
    np.random.seed(0)

    # BERT_MODEL = "roberta-base"
    tokenizer = AutoTokenizer.from_pretrained(base_model_name)
    base_model = AutoModel.from_pretrained(base_model_name)
    model = MyClassifier(base_model)

    print("load classifier train all success: " + str(time.time() - begin_time))
    return model, tokenizer


# 训练集
class MyTrainDataset(Dataset):
    def __init__(self, dataframe, tokenizer, max_nums=None):
        texts = [x['content'] for i, x in dataframe.iterrows()][0: max_nums]

        self.labels = [x['label'] for i, x in dataframe.iterrows()][0: max_nums]

        self.texts = [tokenizer(text, padding='max_length',
                                max_length=512,
                                truncation=True,
                                return_tensors="pt")
                      for text in texts]

        print("end tokenize datas")

    def __getitem__(self, idx):
        text = self.texts[idx]
        label = self.labels[idx]

        return text, label

    def __len__(self):
        return min(len(self.texts), len(self.labels))


# 对抗训练集
class MyAdversaryDataset(Dataset):
    def __init__(self, datas: list, tokenizer):
        texts = [x['content'] for x in datas]
        # print("begin tokenize datas")
        self.texts = [tokenizer(text, padding='max_length',
                                max_length=512,
                                truncation=True,
                                return_tensors="pt")
                      for text in texts]

        # print("end tokenize datas")
        self.labels = [x['label'] for x in datas]

    def __getitem__(self, idx):
        text = self.texts[idx]
        label = self.labels[idx]

        return text, label

    def __len__(self):
        return min(len(self.texts), len(self.labels))

def load_train_and_val_df(train_data_path="../Deberta_test/data/hc3_all.jsonl.train", val_size=0.2, random_state=0):
    train_file = pd.read_json(train_data_path)
    train_df, val_df = train_test_split(train_file, test_size=val_size, random_state=random_state)
    return train_df, val_df


def get_train_and_val_dataloader(train_df, val_df, tokenizer, batch_size=16, shuffle=False):
    train_dataloader = DataLoader(MyTrainDataset(train_df, tokenizer), batch_size=batch_size, shuffle=shuffle)
    val_dataloader = DataLoader(MyTrainDataset(val_df, tokenizer), batch_size=batch_size, shuffle=shuffle)

    return train_dataloader, val_dataloader

# 训练分类器的逻辑
def train_classifier(base_model_name, test_model_path, train_df, val_df, learning_rate, epochs,
                     batch_size=16,
                     save_name="best_model.pt", adversary_generator: MyGenerator = None, adversary_file_path=None,
                     adversary_data_rate=10):
    tokenizer = AutoTokenizer.from_pretrained(base_model_name)
    if test_model_path == 'roberta-base':
        model = MyClassifier(AutoModel.from_pretrained(base_model_name))
    else:
        model = (torch.load(test_model_path))
        model.eval()
    best_val_loss = float('inf')
    early_stopping_threshold_count = 0

    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")

    criterion = nn.BCELoss()
    optimizer = Adam(model.parameters(), lr=learning_rate)

    model = model.to(device)
    criterion = criterion.to(device)

    train_dataloader, val_dataloader = get_train_and_val_dataloader(train_df, val_df, tokenizer, batch_size)

    for epoch in range(epochs):
        total_acc_train = 0
        total_loss_train = 0

        model.train()

        for train_input, train_label in tqdm(train_dataloader):
            attention_mask = train_input['attention_mask'].to(device)
            input_ids = train_input['input_ids'].squeeze(1).to(device)

            train_label = train_label.to(device)

            output = model(input_ids, attention_mask)

            loss = criterion(output, train_label.float().unsqueeze(1)) * 0.5

            total_loss_train += loss.item()

            acc = ((output >= 0.5).int() == train_label.unsqueeze(1)).sum().item()
            total_acc_train += acc

            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

        total_loss_adversary_train = 0
        if adversary_generator is not None:
            if adversary_file_path is not None:
                adversary_train_data = adversary_generator.load_adversary_train_data(file_path=adversary_file_path)
            else:
                row_train_data_samples = [x['content'] for i, x in train_df.iterrows()]
                # random.shuffle(row_train_data_samples)
                row_train_data_samples = row_train_data_samples[0: adversary_data_rate * batch_size]
                adversary_train_data = adversary_generator.generate_adversary_train_data(row_train_data_samples)
            with open('./tmp/' + save_name + '.adversary.' + str(epoch), 'w', encoding='utf-8') as tmp_adversary_file:
                tmp_adversary_file.write(json.dumps(adversary_train_data))
            adversary_train_dataloader = DataLoader(MyAdversaryDataset(adversary_train_data, tokenizer),
                                                    batch_size=batch_size)
            print("training generate adversary train datas")

            for adversary_train_input, adversary_train_label in tqdm(adversary_train_dataloader):
                attention_mask = adversary_train_input['attention_mask'].to(device)
                input_ids = adversary_train_input['input_ids'].squeeze(1).to(device)
                adversary_train_label = adversary_train_label.to(device)
                output = model(input_ids, attention_mask)
                loss = criterion(output, adversary_train_label.float().unsqueeze(1)) * 2
                total_loss_adversary_train += loss.item()
                acc = ((output >= 0.5).int() == adversary_train_label.unsqueeze(1)).sum().item()
                total_acc_train += acc
                loss.backward()
                optimizer.step()
                optimizer.zero_grad()

        with torch.no_grad():
            total_acc_val = 0
            total_loss_val = 0

            model.eval()

            for val_input, val_label in tqdm(val_dataloader):
                attention_mask = val_input['attention_mask'].to(device)
                input_ids = val_input['input_ids'].squeeze(1).to(device)

                val_label = val_label.to(device)

                output = model(input_ids, attention_mask)

                loss = criterion(output, val_label.float().unsqueeze(1))

                total_loss_val += loss.item()

                acc = ((output >= 0.5).int() == val_label.unsqueeze(1)).sum().item()
                total_acc_val += acc

            print(f'Epochs: {epoch + 1} '
                  f'| Train Loss: {total_loss_train / len(train_dataloader): .3f} '
                  f'| Train Accuracy: {total_acc_train / (len(train_dataloader.dataset)): .3f} '
                  f'| Val Loss: {total_loss_val / len(val_dataloader): .3f} '
                  f'| Val Accuracy: {total_acc_val / len(val_dataloader.dataset): .3f}')

            if best_val_loss > total_loss_val + total_loss_adversary_train:
                best_val_loss = total_loss_val + total_loss_adversary_train
                torch.save(model, save_name)
                print("Saved model")
            else:
                pass


def train_generator(
        classifier_model_name, classifier_model_path,
        generator_model_name, generator_model_path,
        try_generate_min_nums, generate_max_retry_times,
        min_available_score,
        ai_texts: list[str], max_text_nums,
        output_dir='./tmp/', tmp_data_str_num='1'
):
    # 首先加载对抗分类器
    begin_time = time.time()
    classifier_model, classifier_tokenizer = load_predict_model(classifier_model_name, classifier_model_path)
    print("load classifier: " + str(time.time() - begin_time))
    # 然后加载对抗文本打分器
    begin_time = time.time()
    generate_text_scorer = GenerateTextScorer(need_euclidean=False, need_cosine=False, need_edit_distance=False,
                                              need_rouge=False)
    print("load generate text scorer: " + str(time.time() - begin_time))
    # 最后加载对抗生成器
    begin_time = time.time()
    generator = MyGenerator(generator_model_name, generator_model_path)
    print("load generator scorer: " + str(time.time() - begin_time))

    # 将输入的ai文本先判别一遍，找出正确的判别文本进行重写
    row_rewrite_texts = []
    predict_texts_results = get_text_predictions(classifier_model, classifier_tokenizer, ai_texts)
    for i in range(0, len(ai_texts)):
        predict_texts_result = predict_texts_results[i]
        if not predict_texts_result:
            row_rewrite_texts.append(ai_texts[i])

    rewrite_results = []
    print("begin generate train datas")
    begin_time = time.time()
    for ai_text in tqdm(row_rewrite_texts):
        cur_rewrite_texts = []
        # 总生成样本数量足够就停止
        if len(rewrite_results) > max_text_nums:
            break
        # 对于每一个文本 尝试多次重写 直到判别失败 或者超过最大重试次数
        # 尝试找出两条成功重写文本
        for _ in range(0, generate_max_retry_times):
            cur_rewrite_text = generator.adversary_chat(ai_text)
            # 判别失败
            if get_text_predictions(classifier_model, classifier_tokenizer, [cur_rewrite_text])[0]:
                cur_rewrite_texts.append(cur_rewrite_text)
                if min_available_score is None:
                    break
            if len(cur_rewrite_texts) >= try_generate_min_nums:
                break

        # 找出最合适的那条训练样本
        cur_rewrite_result = None
        # 检查目前单条样本生成的对抗样本数量
        if len(cur_rewrite_texts) == 0:
            continue
        # 只有一条那没得选
        elif len(cur_rewrite_texts) == 1:
            cur_rewrite_result = cur_rewrite_texts[0]
        # 大于一条需要排序一下，得分高的优先
        else:
            scores = [generate_text_scorer.weighted_score(generate_text_scorer.row_score(ai_text, x)) for x in
                      cur_rewrite_texts]
            max_score = np.max(scores)
            # 不满足最低得分 则放弃
            if max_score < min_available_score:
                continue
            for i in range(0, len(cur_rewrite_texts)):
                if scores[i] == max_score:
                    cur_rewrite_result = cur_rewrite_texts[i]
                    break
        cur_prompt = '<s>[INST] ' + generator.prompt_template + ai_text + ' [/INST]'
        cur_chosen = cur_prompt + cur_rewrite_result + '</s>'
        cur_rejected = cur_prompt + ai_text + '</s>'
        rewrite_results.append(
            {
                'prompt': cur_prompt,
                'chosen': cur_chosen,
                'rejected': cur_rejected
            }
        )
    print("end generate train datas: " + str(time.time() - begin_time))

    # 临时保存结果
    tmp_out_f_path = output_dir + '.adv.' + str(max_text_nums) + '.' + tmp_data_str_num + '.train'
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f"Directory '{output_dir}' created.")
    else:
        print(f"Directory '{output_dir}' already exists.")
    with open(tmp_out_f_path, 'w', encoding='utf-8') as tmp_out_f:
        tmp_out_f.write(json.dumps(rewrite_results))

    # 清理内存 准备正式进行训练
    del classifier_model, classifier_tokenizer, generate_text_scorer, generator
    gc.collect()  # 执行垃圾回收
    torch.cuda.empty_cache()  # 清空CUDA缓存，释放GPU内存

    # 准备训练
    print("begin dpo train")
    train_args = load_trainer_args(output_dir)
    train_generator_model, ref_generator_train_model, train_generator_tokenizer = load_generator_train_model()
    train_dataset = datasets.load_dataset('json', data_files={'train': tmp_out_f_path})['train']

    dpo_trainer = DPOTrainer(train_generator_model, ref_model=ref_generator_train_model, args=train_args,
                             train_dataset=train_dataset,
                             # data_collator=functools.partial(collate_fn, tokenizer=tokenizer),
                             tokenizer=train_generator_tokenizer,
                             max_length=512,
                             max_prompt_length=512,
                             peft_config=peft_config
                             )

    dpo_trainer.train()
    dpo_trainer.save_model(train_args.output_dir)
    dpo_output_dir = os.path.join(train_args.output_dir, "final_checkpoint")
    dpo_trainer.model.save_pretrained(dpo_output_dir)


if __name__ == '__main__':
    ### generate data part
    # predict_model, predict_tokenizer = load_predict_model(model_path='../roberta_test/hc3_row.pt')
    # print(get_text_predictions(
    #     model=predict_model
    #     , tokenizer=predict_tokenizer,
    #     texts=['A proposed financial plan to support Mental Health Services for Healthcare Providers of Critical ($)Staff Mental Health Service Manager 400,000 Personal 840,000 Group 650,000 Technical Support Staff 180,000 Equipment Personal Laptops 12,000 Personal Mobile Phones 5,000 Services Telephone Service 2,400 Internet Service 300 Supplies Supplies 500 Medication 1,500 Training Employee Training 2,000 Managerial Training 2,000 Offices 3,000 Laptops and Mobile Phones 1,500 Other Unforeseen 5,000 The impact of a biblical on the proposed financial planThe biblical on presumes that leaders are granted control over other people and resources by God (Carradus, Zozimo, & Discua Cruz, 2020). Thus, they should strive to ensure the well-being of every person in order to honor the Lord. In this regard, the current financial plan was designed in a manner that would benefit all the involved, including healthcare providers of critical patients and consulting staff.The potential holes and unknowns in the project’s financial the financial plan seemingly addresses most of the expenses that may occur during the project there are still some unknowns that should be As such, O’Connell (2020) argues that managers should always think about the worst-case scenario and be prepared to respond In the case of the current project, first of all, it is hard to predict the real demand for mental health services among doctors. This, in turn, negatively affects the ability to predict the required number of mental health workers. Secondly, potential crises and resulting inflation rates may cause a price increase, which, in turn, would additional money to purchase equipment and Moreover, the costs for the services, training, and may also that can fill the potential holes in the project’s financial planAs for the former hole in the financial plan, it can be assumed that there will be an average demand for mental services among healthcare This assumption would help to partly mitigate the risks of the unknown need for offered services. It is explained by the fact that when the expected demand is low but the actual necessity is high, then the patients are largely On the contrary, when are high, but the actual demand is low, it leads to financial losses. As for inflation, it is necessary to assume a certain amount of money for unforeseen Therefore, if the prices rise, the will still be able to pay for the planned expenses.'],
    # ))
    # predict_jsonl(model=predict_model, tokenizer=predict_tokenizer, jsonl_file='./data/finance.mix.human.jsonl')
    # predict_jsonl(model=predict_model, tokenizer=predict_tokenizer, jsonl_file='./data/open_qa.mix.human.jsonl')
    # predict_jsonl(model=predict_model, tokenizer=predict_tokenizer, jsonl_file='./data/medicine.mix.human.jsonl')

    ### train part
    # prepare_train_data()

    # with open('./data/finance.mix.human.jsonl.all', 'r', encoding='utf-8') as f_1:
    #     jsons1 = json.load(f_1)
    # with open('./data/medicine.mix.human.jsonl.all', 'r', encoding='utf-8') as f_2:
    #     jsons2 = json.load(f_2)
    # with open('./data/medicine.mix.human.jsonl.all', 'r', encoding='utf-8') as f_3:
    #     jsons3 = json.load(f_3)
    # with open('./data/hc3_all_1.mix.human.jsonl.all', 'w', encoding='utf-8') as f_4:
    #     f_4.write(json.dumps(jsons1 + jsons2 + jsons3))
    #
    # train_args = load_trainer_args(output_dir='hc3_all_1')
    #
    # dataset_path = './data/hc3_all_1.mix.human.jsonl.all'
    # convert_dataset(dataset_path)
    # train_dataset = datasets.load_dataset('json', data_files={'train': dataset_path + '.conv'})['train']
    #
    # model, ref_model, tokenizer = load_model()
    # tokenizer.pad_token = tokenizer.eos_token

    # tarin_dataset = Dataset.from_generator(
    #     lambda: load_dataset(
    #         '../../data_collector/test_data/hc3_english_mix_multi/wiki_csai.mix.jsonl',
    #         tokenizer,
    #         prompt_key='question',
    #         accept_key='human',
    #         reject_key='ai'
    #     )
    # )

    # note: use gradient checkpointing to save memory at the expense of slower backward pass.
    # model.gradient_checkpointing_enable()
    # model.enable_input_require_grads()

    # ref_model = ref_model.eval().requires_grad_(False)
    # print(train_args)
    # trainer = DPOTrainer(model, ref_model=None, args=train_args, train_dataset=train_dataset,
    #                      # data_collator=functools.partial(collate_fn, tokenizer=tokenizer),
    #                      tokenizer=tokenizer,
    #                      max_length=512,
    #                      max_prompt_length=512,
    #                      peft_config=peft_config
    #                      )
    #
    # trainer.train()
    # trainer.save_model(train_args.output_dir)
    # output_dir = os.path.join(train_args.output_dir, "final_checkpoint")
    # trainer.model.save_pretrained(output_dir)

    # with open('../roberta_test/data/hc3_row.train', 'r', encoding='utf-8') as train_f:
    #     ai_texts = [x['content'] for x in json.load(train_f) if x['label'] == 1]
    # train_generator(
    #     'roberta-base', 'hc3_row.pt',
    #     "mistralai/Mistral-7B-Instruct-v0.2", './hc3_all_1/final_checkpoint',
    #     3, 10,
    #     None,
    #     ai_texts,
    #     1000,
    #     './dpo_no_blue/'
    #     '1'
    # )

    # with open('../moe_test/data/nature/mix/7.jsonl.rewrite.jsonl.train', 'r', encoding='utf-8') as train_f:
    #     ai_texts = [x['content'] for x in json.load(train_f) if x['label'] == 1]
    # train_generator(
    #     'roberta-base', '../roberta_test/moe_adt3.pt',
    #     "mistralai/Mistral-7B-Instruct-v0.2", "mistralai/Mistral-7B-Instruct-v0.2",
    #     3, 10,
    #     None,
    #     ai_texts,
    #     1000,
    #     './moe3/'
    #     '1'
    # )

    # with open('../moe_test/data/7_m4_chatGPT.json.train', 'r', encoding='utf-8') as train_f:
    #     ai_texts = [x['content'] for x in json.load(train_f) if x['label'] == 1]
    # train_generator(
    #     'roberta-base', '../roberta_test/moe_adt4.pt',
    #     "mistralai/Mistral-7B-Instruct-v0.2", "mistralai/Mistral-7B-Instruct-v0.2",
    #     3, 10,
    #     None,
    #     ai_texts,
    #     1000,
    #     './moe4/'
    #     '1'
    # )

    # train_file = '../roberta_test/data/hc3_row.train'
    # train_df, val_df = load_train_and_val_df(train_file)
    # train_classifier(
    #     'roberta-base', 'roberta-base',
    #     train_df, val_df,
    #     learning_rate=1e-5,
    #     epochs=5,
    #     batch_size=16,
    #     save_name='ppo_1.pt',
    #     adversary_generator=MyGenerator('mistralai/Mistral-7B-Instruct-v0.2', './ppo_1/final_checkpoint'),
    #     adversary_data_rate=2
    # )

    # train_file = '../moe_test/data/nature/mix/7.jsonl.rewrite.jsonl.train'
    # train_df, val_df = load_train_and_val_df(train_file)
    # train_classifier(
    #     'roberta-base', 'roberta-base',
    #     train_df, val_df,
    #     learning_rate=1e-5,
    #     epochs=5,
    #     batch_size=16,
    #     save_name='moe_3.pt',
    #     adversary_generator=MyGenerator('mistralai/Mistral-7B-Instruct-v0.2', './moe3/1/final_checkpoint'),
    #     adversary_data_rate=2
    # )

    train_file = '../moe_test/data/7_m4_chatGPT.json.train'
    train_df, val_df = load_train_and_val_df(train_file)
    train_classifier(
        'roberta-base', 'roberta-base',
        train_df, val_df,
        learning_rate=1e-5,
        epochs=5,
        batch_size=16,
        save_name='moe_4.pt',
        adversary_generator=MyGenerator('mistralai/Mistral-7B-Instruct-v0.2', './moe4/1/final_checkpoint'),
        adversary_data_rate=2
    )

    pass

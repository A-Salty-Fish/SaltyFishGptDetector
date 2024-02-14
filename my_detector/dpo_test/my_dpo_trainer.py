import functools
import json
import time
from typing import Dict

import datasets
import torch
from peft import LoraConfig
from torch import nn
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments, BitsAndBytesConfig, AutoModel
from datasets import Dataset
from trl import DPOTrainer
from torch.nn.utils.rnn import pad_sequence

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

peft_config = LoraConfig(
    target_modules=["q_proj", "k_proj"],
    init_lora_weights=False
)

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
)


def load_trainer_args():
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
    train_args = TrainingArguments(output_dir='./tmp.pt')
    train_args.gradient_accumulation_steps = 1
    train_args.num_train_epochs = 2
    train_args.save_steps = 200
    train_args.save_total_limit = 2
    train_args.learning_rate = 5e-4
    train_args.seed = 42
    train_args.ddp_find_unused_parameters = False
    train_args.remove_unused_columns = False
    train_args.logging_steps = 100
    train_args.per_device_train_batch_size = 1
    train_args.per_device_eval_batch_size = 1
    return train_args


def load_model(model_name="mistralai/Mistral-7B-Instruct-v0.2"):
    all_begin_time = time.time()

    begin_time = time.time()
    model = AutoModelForCausalLM.from_pretrained(model_name,
                                                 quantization_config=bnb_config,
                                                 trust_remote_code=True)
    print("load model success: " + str(time.time() - begin_time))

    begin_time = time.time()
    ref_model = AutoModelForCausalLM.from_pretrained(model_name,
                                                     quantization_config = bnb_config,
                                                     trust_remote_code=True)
    print("load ref_model success: " + str(time.time() - begin_time))

    begin_time = time.time()
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    print("load tokenizer success: " + str(time.time() - begin_time))

    print("load all success: " + str(time.time() - all_begin_time))
    return model, ref_model, tokenizer


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


#
# def collate_fn(batch, tokenizer):
#     # first, pad everything to the same length
#     tokenizer.pad_token_id = tokenizer.eod_id
#     padded_batch = {}
#     for k in batch[0].keys():
#         if k.endswith('_input_ids') or k.endswith('_attention_mask') or k.endswith('_labels'):
#             if 'prompt' in k:  # adapted from https://stackoverflow.com/questions/73256206
#                 to_pad = [torch.LongTensor(ex[k][::-1]) for ex in batch]
#             else:
#                 to_pad = [torch.LongTensor(ex[k]) for ex in batch]
#             if k.endswith('_input_ids'):
#                 padding_value = tokenizer.pad_token_id
#             elif k.endswith('_labels'):
#                 padding_value = -100
#             elif k.endswith('_attention_mask'):
#                 padding_value = 0
#             else:
#                 raise ValueError(f"Unexpected key in batch '{k}'")
#
#             padded_batch[k] = pad_sequence(to_pad, batch_first=True, padding_value=padding_value)
#             if 'prompt' in k:  # for the prompt, flip back so padding is on left side
#                 padded_batch[k] = padded_batch[k].flip(dims=[1])
#         else:
#             padded_batch[k] = [ex[k] for ex in batch]
#
#     return padded_batch


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
                      max_length=256,
                      truncation=True,
                      return_tensors="pt")
        )

    with torch.no_grad():
        model.eval()
        for data_input in tqdm(data_inputs):
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
    print(len(texts))
    print(len(predict_results))
    print(len([x for x in predict_results if x]))

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


if __name__ == '__main__':

    predict_model, predict_tokenizer = load_predict_model(model_path='../roberta_test/hc3_row.pt')
    # print(get_text_predictions(
    #     model=predict_model
    #     , tokenizer=predict_tokenizer,
    #     texts=['A proposed financial plan to support Mental Health Services for Healthcare Providers of Critical ($)Staff Mental Health Service Manager 400,000 Personal 840,000 Group 650,000 Technical Support Staff 180,000 Equipment Personal Laptops 12,000 Personal Mobile Phones 5,000 Services Telephone Service 2,400 Internet Service 300 Supplies Supplies 500 Medication 1,500 Training Employee Training 2,000 Managerial Training 2,000 Offices 3,000 Laptops and Mobile Phones 1,500 Other Unforeseen 5,000 The impact of a biblical on the proposed financial planThe biblical on presumes that leaders are granted control over other people and resources by God (Carradus, Zozimo, & Discua Cruz, 2020). Thus, they should strive to ensure the well-being of every person in order to honor the Lord. In this regard, the current financial plan was designed in a manner that would benefit all the involved, including healthcare providers of critical patients and consulting staff.The potential holes and unknowns in the project’s financial the financial plan seemingly addresses most of the expenses that may occur during the project there are still some unknowns that should be As such, O’Connell (2020) argues that managers should always think about the worst-case scenario and be prepared to respond In the case of the current project, first of all, it is hard to predict the real demand for mental health services among doctors. This, in turn, negatively affects the ability to predict the required number of mental health workers. Secondly, potential crises and resulting inflation rates may cause a price increase, which, in turn, would additional money to purchase equipment and Moreover, the costs for the services, training, and may also that can fill the potential holes in the project’s financial planAs for the former hole in the financial plan, it can be assumed that there will be an average demand for mental services among healthcare This assumption would help to partly mitigate the risks of the unknown need for offered services. It is explained by the fact that when the expected demand is low but the actual necessity is high, then the patients are largely On the contrary, when are high, but the actual demand is low, it leads to financial losses. As for inflation, it is necessary to assume a certain amount of money for unforeseen Therefore, if the prices rise, the will still be able to pay for the planned expenses.'],
    # ))
    predict_jsonl(model=predict_model, tokenizer=predict_tokenizer, jsonl_file='./data/finance.mix.human.jsonl')

    # # prepare_train_data()
    #
    # train_args = load_trainer_args()
    #
    # train_dataset = datasets.load_dataset('json', data_files={'train': './data/wiki_csai.mix.train'})['train']
    #
    # model, ref_model, tokenizer = load_model()
    # tokenizer.pad_token = tokenizer.eos_token
    #
    # # tarin_dataset = Dataset.from_generator(
    # #     lambda: load_dataset(
    # #         '../../data_collector/test_data/hc3_english_mix_multi/wiki_csai.mix.jsonl',
    # #         tokenizer,
    # #         prompt_key='question',
    # #         accept_key='human',
    # #         reject_key='ai'
    # #     )
    # # )
    #
    # # note: use gradient checkpointing to save memory at the expense of slower backward pass.
    # # model.gradient_checkpointing_enable()
    # # model.enable_input_require_grads()
    #
    # # ref_model = ref_model.eval().requires_grad_(False)
    # print(train_args)
    # trainer = DPOTrainer(model, ref_model, args=train_args, train_dataset=train_dataset,
    #                      # data_collator=functools.partial(collate_fn, tokenizer=tokenizer),
    #                      tokenizer=tokenizer,
    #                      max_length=512,
    #                      max_prompt_length=512,
    #                      peft_config=peft_config
    #                      )
    #
    # trainer.train()
    # trainer.save_model(train_args.output_dir)

    pass

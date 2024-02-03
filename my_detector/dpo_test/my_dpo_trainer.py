import functools
import json
import time
from typing import Dict

import datasets
import torch
from peft import LoraConfig
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments, BitsAndBytesConfig
from datasets import Dataset
from trl import DPOTrainer
from torch.nn.utils.rnn import pad_sequence

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
    train_args.per_device_train_batch_size = 2
    train_args.per_device_eval_batch_size = 2
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
    # prepare_train_data()

    train_args = load_trainer_args()

    train_dataset = datasets.load_dataset('json', data_files={'train': './data/wiki_csai.mix.train'})['train']

    model, ref_model, tokenizer = load_model("microsoft/phi-2")
    tokenizer.pad_token = tokenizer.eos_token

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
    print(train_args)
    trainer = DPOTrainer(model, ref_model, args=train_args, train_dataset=train_dataset,
                         # data_collator=functools.partial(collate_fn, tokenizer=tokenizer),
                         tokenizer=tokenizer,
                         max_length=512,
                         max_prompt_length=512,
                         peft_config=peft_config
                         )

    trainer.train()
    trainer.save_model(train_args.output_dir)

    pass

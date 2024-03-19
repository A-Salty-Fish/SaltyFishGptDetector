# Copyright (c) Guangsheng Bao.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
import csv
import random

import numpy as np
import torch
import os
import glob
import argparse
import json
from model import load_tokenizer, load_model
from fast_detect_gpt import get_sampling_discrepancy_analytic
from tqdm import tqdm
import time

# estimate the probability according to the distribution of our test results on ChatGPT and GPT-4
class ProbEstimator:
    def __init__(self, args):
        self.real_crits = []
        self.fake_crits = []
        for result_file in glob.glob(os.path.join(args.ref_path, '*.json')):
            with open(result_file, 'r') as fin:
                res = json.load(fin)
                self.real_crits.extend(res['predictions']['real'])
                self.fake_crits.extend(res['predictions']['samples'])
        print(f'ProbEstimator: total {len(self.real_crits) * 2} samples.')


    def crit_to_prob(self, crit):
        offset = np.sort(np.abs(np.array(self.real_crits + self.fake_crits) - crit))[100]
        cnt_real = np.sum((np.array(self.real_crits) > crit - offset) & (np.array(self.real_crits) < crit + offset))
        cnt_fake = np.sum((np.array(self.fake_crits) > crit - offset) & (np.array(self.fake_crits) < crit + offset))
        return cnt_fake / (cnt_real + cnt_fake)


def is_human(args, scoring_model, scoring_tokenizer,  reference_model , reference_tokenizer, prob_estimator, criterion_fn, text, bar=0.5):
    try:
        tokenized = scoring_tokenizer(text, return_tensors="pt", padding=True, return_token_type_ids=False).to(args.device)
        labels = tokenized.input_ids[:, 1:]
        with torch.no_grad():
            logits_score = scoring_model(**tokenized).logits[:, :-1]
            if args.reference_model_name == args.scoring_model_name:
                logits_ref = logits_score
            else:
                tokenized = reference_tokenizer(text, return_tensors="pt", padding=True, return_token_type_ids=False).to(
                    args.device)
                assert torch.all(tokenized.input_ids[:, 1:] == labels), "Tokenizer is mismatch."
                logits_ref = reference_model(**tokenized).logits[:, :-1]
            crit = criterion_fn(logits_ref, logits_score, labels)
        # estimate the probability of machine generated text
        prob = prob_estimator.crit_to_prob(crit)
        if prob <= bar:
            return 0
        else:
            return 1
    except Exception as e:
        return 0


def test_files(files, args, scoring_model, scoring_tokenizer, reference_model, reference_tokenizer,
                                      prob_estimator, criterion_fn, output_file_name):
    test_results = []
    for file in files:
        with open(file, 'r', encoding='utf-8') as in_f:
            json_arr = json.load(in_f)
        print(file)
        start_time = time.time()

        human_true = 0
        human_total = 0
        human_true_rate = 0.0
        ai_true = 0
        ai_total = 0
        ai_true_rate = 0.0
        precision = 0.0
        recall = 0.0
        f1 = 0

        ai_objs = [x for x in json_arr if x['label'] == 1][0:1000]
        human_objs = [x for x in json_arr if x['label'] == 0][0:1000]

        all_objs = ai_objs + human_objs
        # all_data = data_set['human'] + data_set['ai']
        i = 0
        for data in tqdm(all_objs):
            i += 1
            content = data['content']
            # 截断过长的数据
            words = content.split(' ')
            if len(words) > 512:
                content = " ".join(words[0: 512])
            label = data['label']
            try:
                pred_label = is_human(args, scoring_model, scoring_tokenizer, reference_model, reference_tokenizer,
                                      prob_estimator, criterion_fn, content)
                if label == 0:
                    if pred_label:
                        human_true += 1
                    human_total += 1

                elif label == 1:
                    if not pred_label:
                        ai_true += 1
                    ai_total += 1
                # percent = round(1.0 * (i) / len(all_data) * 100, 2)
                # print('test process : %s [%d/%d]' % (str(percent) + '%', i, len(all_data)), end='\r')
            except Exception as e:
                print(e)
                # print('error content:' + content)
        print("test process end", end='\n')

        if human_total != 0:
            human_true_rate = human_true / human_total
        if ai_total != 0:
            ai_true_rate = ai_true / ai_total

        if ai_total != 0 and human_total != 0:
            if (ai_true + (human_total - human_true)) != 0:
                precision = ai_true / (ai_true + (human_total - human_true))
            recall = ai_true / ai_total
            if (precision + recall) != 0:
                f1 = 2 * precision * recall / (precision + recall)

        end_time = time.time()
        print("time to test {} seconds.".format(end_time - start_time))

        test_result = {
            "human_true": human_true,
            "human_total": human_total,
            "human_true_rate": human_true_rate,
            "ai_true": ai_true,
            "ai_total": ai_total,
            "ai_true_rate": ai_true_rate,
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "run_seconds": end_time - start_time,
            "file": file,
            'method': 'fast-detect-gpt'
        }
        test_results.append(test_result)
    print('end')
    print(test_results)
    results = test_results
    with open(output_file_name + '.csv', 'w', encoding='utf-8') as output_file:
        fieldnames = results[0].keys()
        writer = csv.DictWriter(output_file, fieldnames=fieldnames)
        # 写入标题行
        writer.writeheader()
        # 写入数据行
        writer.writerows(results)


# run interactive local inference
def run(args):
    # load model
    scoring_tokenizer = load_tokenizer(args.scoring_model_name, args.dataset, args.cache_dir)
    scoring_model = load_model(args.scoring_model_name, args.device, args.cache_dir)
    scoring_model.eval()
    reference_model = None
    reference_tokenizer = None
    if args.reference_model_name != args.scoring_model_name:
        reference_tokenizer = load_tokenizer(args.reference_model_name, args.dataset, args.cache_dir)
        reference_model = load_model(args.reference_model_name, args.device, args.cache_dir)
        reference_model.eval()
    # evaluate criterion
    name = "sampling_discrepancy_analytic"
    criterion_fn = get_sampling_discrepancy_analytic
    prob_estimator = ProbEstimator(args)
    # input text
    print('Local demo for Fast-DetectGPT, where the longer text has more reliable result.')
    print('')

    test_file_base_dir = '../my_detector/roberta_test/data/'
    files = [
        'cheat_generation.test',
        'cheat_polish.test',
        'ghostbuster_claude.test',
        'hc3_plus_qa_row.test',
        # 'open_qa.academic.test',
        # 'open_qa.continue.test',
        # 'open_qa.difficult.test',
        # 'open_qa.easy.test',
        # 'open_qa.test',
        # 'open_qa.rewrite.test',
        'reddit_chatGPT.test',
        'reddit_cohere.test',
        'reddit_davinci.test',
        'reddit_dolly.test',
        'reddit_flant5.test',
        'wikipedia_chatgpt.test',
        'wikipedia_cohere.test',
        'wikipedia_davinci.test',
        'wikipedia_dolly.test',
    ]

    test_files([test_file_base_dir + x for x in files],
               args, scoring_model, scoring_tokenizer, reference_model, reference_tokenizer,
               prob_estimator, criterion_fn,
               output_file_name='result_2'
                       )



    # while True:
    #     print("Please enter your text: (Press Enter twice to start processing)")
    #     lines = []
    #     while True:
    #         line = input()
    #         if len(line) == 0:
    #             break
    #         lines.append(line)
    #     text = "\n".join(lines)
    #     if len(text) == 0:
    #         break
    #     # evaluate text
    #     tokenized = scoring_tokenizer(text, return_tensors="pt", padding=True, return_token_type_ids=False).to(args.device)
    #     labels = tokenized.input_ids[:, 1:]
    #     with torch.no_grad():
    #         logits_score = scoring_model(**tokenized).logits[:, :-1]
    #         if args.reference_model_name == args.scoring_model_name:
    #             logits_ref = logits_score
    #         else:
    #             tokenized = reference_tokenizer(text, return_tensors="pt", padding=True, return_token_type_ids=False).to(args.device)
    #             assert torch.all(tokenized.input_ids[:, 1:] == labels), "Tokenizer is mismatch."
    #             logits_ref = reference_model(**tokenized).logits[:, :-1]
    #         crit = criterion_fn(logits_ref, logits_score, labels)
    #     # estimate the probability of machine generated text
    #     prob = prob_estimator.crit_to_prob(crit)
    #     print(f'Fast-DetectGPT criterion is {crit:.4f}, suggesting that the text has a probability of {prob * 100:.0f}% to be fake.')
    #     print()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--reference_model_name', type=str, default="gpt-neo-2.7B")  # use gpt-j-6B for more accurate detection
    parser.add_argument('--scoring_model_name', type=str, default="gpt-neo-2.7B")
    parser.add_argument('--dataset', type=str, default="xsum")
    parser.add_argument('--ref_path', type=str, default="./local_infer_ref")
    parser.add_argument('--device', type=str, default="cuda")
    parser.add_argument('--cache_dir', type=str, default="../cache")
    args = parser.parse_args()

    run(args)




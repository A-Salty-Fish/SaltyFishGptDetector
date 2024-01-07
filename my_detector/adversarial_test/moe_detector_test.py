import json
import os
import time

from transformers import pipeline, AutoTokenizer, AutoModelForSequenceClassification

import torch

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def load_text_labels_config(base_path='./config/', config_name='generator.json'):
    with open(base_path + config_name, 'r', encoding='utf-8') as text_labels_file:
        return json.load(text_labels_file)['candidate_labels']

def load_moe_detector_config(base_path='./tmp/moe/', file_name = 'moe_detector.json'):
    with open(base_path + file_name, 'r' , encoding='utf-8') as config_file:
        return json.load(config_file)

def load_test_base_model_config(base_path='./config/', config_name='base_model.json'):
    with open(base_path + config_name, 'r', encoding='utf-8') as text_labels_file:
        return json.load(text_labels_file)['test']

def load_utc_base_model_config(base_path='./config/', config_name='base_model.json'):
    with open(base_path + config_name, 'r', encoding='utf-8') as text_labels_file:
        return json.load(text_labels_file)['utc']


def init_utc_pipe(utc_base_model_config):
    start_time = time.time()

    type = utc_base_model_config['type']
    model_name = utc_base_model_config['model_name']

    classifier = pipeline(type, model=model_name)
    end_time = time.time()
    print("load utc model success: " + str(end_time - start_time))
    return classifier


def utc_classify(classifier, labels, text):
    result = []
    output = classifier(text, labels, multi_label=False)
    for i in range(0, len(output['labels'])):
        result.append([output['labels'][i], output['scores'][i]])
    if len(result) == 0:
        result.append(['None', 1.00])
    return result


def init_single_model_and_tokenizer(model_path, base_test_model_config):
    start_time = time.time()

    tokenizer=AutoTokenizer.from_pretrained(base_test_model_config['tokenizer_name'], model_max_length=base_test_model_config['max_length'])
    max_step = -1
    output_dir = model_path
    for file in os.listdir(output_dir):
        max_step = max(max_step, int(file.split('-')[1]))

    model_path = output_dir + "/checkpoint-" + str(max_step)
    print('load model:' + model_path)
    model = AutoModelForSequenceClassification.from_pretrained(model_path, num_labels=base_test_model_config['num_labels'])

    print(DEVICE)
    model = model.to(DEVICE)

    end_time = time.time()
    print("load model successful: " + str(end_time - start_time))
    return model, tokenizer

def init_moe(text_labels, moe_base_path, base_test_model_config):
    start_time = time.time()
    moe_map = {}
    for text_label in text_labels:
        model, tokenizer = init_single_model_and_tokenizer(model_path=moe_base_path, base_test_model_config=base_test_model_config)
        moe_map[text_label] = model, tokenizer
    end_time = time.time()
    print("load moe models successful: " + str(end_time - start_time))
    return moe_map


def classify_by_moe(moe_map, utc_pipe, labels, text, top_k = 1, bar = 0.5000):
    utc_classify_result = utc_classify(utc_pipe, labels, text)
    moe_results = []
    for i in range(0, top_k):
        text_label = utc_classify_result[i][0]
        label_score = utc_classify_result[i][1]
        model, tokenizer = moe_map[text_label]
        inputs = tokenizer([text], padding=True, truncation=True, return_tensors="pt").to(DEVICE)
        outputs = model(**inputs)
        pred_labels = outputs.logits.cpu().argmax(-1).numpy()
        pred_label = pred_labels[0]
        moe_results.append([text_label, label_score, pred_label])

    score_sum = 0
    pred_sum = 0
    for moe_result in moe_results:
        score_sum += moe_result[1]
        pred_sum += moe_result[1] * moe_results[2]

    if pred_sum / score_sum <= bar:
        return 0
    else:
        return 1



def eval_moe(text_labels, moe_detector_config, utc_base_model_config, base_test_model_config):
    start_time = time.time()

    moe_map = init_moe(text_labels, moe_detector_config['output_dir'], base_test_model_config)
    utc_pipe = init_utc_pipe(utc_base_model_config)
    test_files = moe_detector_config['test_files']

    all_result = []

    for test_file in test_files:
        print("begin test "+ test_file)
        i = 0
        human_correct = 0
        human_total = 0
        gpt_correct = 0
        gpt_total = 0
        with open(test_files[test_file], 'r', encoding='utf-8') as test_f:
            json_array = json.load(test_f)
            for json_obj in json_array:
                i += 1
                print('test [%s] process : %s [%d/%d]' % (test_file, str(i * 100 / len(json_array)) + '%', i, len(json_array)), end='\r')
                raw_input = json_obj['content']
                raw_label = json_obj['label']
                pred_label = classify_by_moe(moe_map, utc_pipe, text_labels, raw_input, moe_detector_config['top_k'], moe_detector_config['bar'])
                if raw_label == 0:
                    human_total += 1
                    if pred_label == 0:
                        human_correct += 1
                else:
                    gpt_total += 1
                    if pred_label == 1:
                        gpt_correct += 1

        all_result.append({
            "test_file": test_file,
            "human_total": human_total,
            "ai_total": gpt_total,
            "total_acc": (1.0 * (human_correct + gpt_correct) / (human_total + gpt_total)),
            "human_acc": (1.0 * human_correct / human_total),
            "ai_acc": (1.0 * (gpt_correct) / (gpt_total))
        })
        end_time = time.time()
        print("test " + test_file + " successful: " + str(end_time - start_time))

    print("test all" + " successful: " + str(end_time - start_time))
    return all_result



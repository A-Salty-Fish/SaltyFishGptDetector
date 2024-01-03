import json
import os
import random
import time

from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline, AutoModelForSequenceClassification

import torch

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def load_test_base_model_config(base_path='./config/', config_name='base_model.json'):
    with open(base_path + config_name, 'r', encoding='utf-8') as text_labels_file:
        return json.load(text_labels_file)['test']

def load_text_labels_config(base_path='./config/', config_name='generator.json'):
    with open(base_path + config_name, 'r', encoding='utf-8') as text_labels_file:
        return json.load(text_labels_file)['candidate_labels']


def load_prompt_templates_config(base_path='./config/', config_name='generator.json'):
    with open(base_path + config_name, 'r', encoding='utf-8') as text_labels_file:
        return json.load(text_labels_file)['prompt_templates']

def load_init_eval_config(base_path='./config/', config_name='generator.json'):
    with open(base_path + config_name, 'r', encoding='utf-8') as text_labels_file:
        return json.load(text_labels_file)['init_eval']


def load_chat_base_model_config(base_path='./config/', config_name='base_model.json'):
    with open(base_path + config_name, 'r', encoding='utf-8') as text_labels_file:
        return json.load(text_labels_file)['chat']


def load_utc_base_model_config(base_path='./config/', config_name='base_model.json'):
    with open(base_path + config_name, 'r', encoding='utf-8') as text_labels_file:
        return json.load(text_labels_file)['utc']


def init_generator_model_and_tokenizer(chat_base_model_config):
    start_time = time.time()
    tokenizer = AutoTokenizer.from_pretrained(chat_base_model_config['tokenizer_name'])
    model = AutoModelForCausalLM.from_pretrained(chat_base_model_config['model_name'])

    end_time = time.time()
    print("load generator model successful: " + str(end_time - start_time))
    return model, tokenizer


def chat(model, tokenizer, context):
    device = 'cuda'
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


def init_detector_model_and_tokenizer(model_base_path, base_test_model_config):
    start_time = time.time()
    tokenizer=AutoTokenizer.from_pretrained(base_test_model_config['tokenizer_name'], model_max_length=base_test_model_config['max_length'])

    max_step = -1

    output_dir = model_base_path

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


def eval(model_base_path, test_file, text_labels, test_num_per_label=100, random_test=True):
    test_datas = []
    with open(test_file, 'r', encoding='utf-8') as test_input:
        for line in test_input:
            json_obj = json.loads(line)
            test_datas.append({
                'label': json_obj['label'],
                'content': json_obj['content'],
                "text_label": json_obj['class_label']
            })
    # to data map
    label_data_map = {}
    for text_label in text_labels:
        label_data_map[text_label] = []
    for test_data in test_datas:
        if test_data['text_label'] in text_labels:
            label_data_map[test_data['text_label']].append(test_data)
    # random list
    if random_test:
        for text_label in text_labels:
            random.shuffle(label_data_map[text_label])
    # limit size
    for text_label in text_labels:
        if len(label_data_map[text_label]) > test_num_per_label:
            label_data_map[text_label] = label_data_map[text_label][0: test_num_per_label]


    # load model
    start_time = time.time()
    test_base_model_config = load_test_base_model_config()
    model, tokenizer = init_detector_model_and_tokenizer(model_base_path, test_base_model_config)
    end_time = time.time()
    print("load test model successful: " + str(end_time - start_time))

    # test each label
    test_result = {}
    for text_label in text_labels:
        test_label_data = label_data_map[text_label]

        start_time = time.time()
        human_correct = 0
        human_total = 0
        ai_correct = 0
        ai_total = 0

        for i in range(0, len(test_label_data)):
            print('test [%s] process : %s [%d/%d]' % (text_label, str(i * 100 / len(test_label_data)) + '%', i, len(test_label_data)), end='\r')
            raw_input = test_label_data[i]['content']
            raw_label = test_label_data[i]['label']
            inputs = tokenizer([raw_input], padding=True, truncation=True, return_tensors="pt").to(DEVICE)
            outputs = model(**inputs)
            pred_labels = outputs.logits.cpu().argmax(-1).numpy()
            pred_label = pred_labels[0]

            if raw_label == 0:
                human_total += 1
                if pred_label == 0:
                    human_correct += 1
            else:
                ai_total += 1
                if pred_label == 1:
                    ai_correct += 1
        if human_total == 0:
            human_acc = 0
        else:
            human_acc = (1.0 * human_correct / human_total)
        if ai_total == 0:
            ai_acc = 0
        else:
            ai_acc = (1.0 * (ai_correct) / (ai_total))
        label_test_result = {
            "model_base_path": model_base_path,
            "test_file": test_file,
            "total": ai_total + human_total,
            "ai_total": ai_total,
            "human_total": human_total,
            "total_acc": (1.0 * (human_correct + ai_correct) / (human_total + ai_total)),
            "human_acc":  human_acc,
            "ai_acc": ai_acc
        }
        print(label_test_result)
        test_result[text_label] = label_test_result
        end_time = time.time()
        print("test label " + text_label + " successful: " + str(end_time - start_time))
    return test_result



if __name__ == '__main__':

    # chat_base_model_config = load_chat_base_model_config()
    # model, tokenizer = init_generator_model_and_tokenizer(chat_base_model_config)
    #
    # utc_base_model_config = load_utc_base_model_config()
    # text_labels = load_text_labels_config()
    # classifier = init_utc_pipe(utc_base_model_config)
    #
    # chat_res = chat(model, tokenizer, "hello, can you tell me something about smart phone. about 100 words")
    # print(chat_res)
    #
    # print(utc_classify(classifier, text_labels, chat_res))

    init_eval_config = load_init_eval_config()
    print(eval(init_eval_config['init_model_path'], init_eval_config['test_file'], load_text_labels_config(), init_eval_config['test_num_per_label'] , init_eval_config['random_test']))


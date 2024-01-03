import json
import os
import time

from transformers import AutoModelForSequenceClassification, AutoTokenizer
import torch
import numpy
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def load_detector_config(base_path='./config/', config_name='detector.json'):
    with open(base_path + config_name, 'r', encoding='utf-8') as text_labels_file:
        return json.load(text_labels_file)

def load_test_base_model_config(base_path='./config/', config_name='base_model.json'):
    with open(base_path + config_name, 'r', encoding='utf-8') as text_labels_file:
        return json.load(text_labels_file)['test']


def init_init_model_and_tokenizer(detector_config, base_test_model_config):
    start_time = time.time()


    tokenizer=AutoTokenizer.from_pretrained(base_test_model_config['tokenizer_name'], model_max_length=base_test_model_config['max_length'])

    max_step = -1

    output_dir = detector_config['init']['output_dir']

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

def init_cur_model_and_tokenizer(detector_config, base_test_model_config):
    start_time = time.time()


    tokenizer=AutoTokenizer.from_pretrained(base_test_model_config['tokenizer_name'], model_max_length=base_test_model_config['max_length'])

    max_step = -1

    output_dir = detector_config['cur']['output_dir']

    for file in os.listdir(output_dir):
        if file.find('checkpoint') == -1:
            continue
        max_step = max(max_step, int(file.split('-')[1]))

    model_path = output_dir + "/checkpoint-" + str(max_step)
    print('load model:' + model_path)
    model = AutoModelForSequenceClassification.from_pretrained(model_path, num_labels=base_test_model_config['num_labels'])

    print(DEVICE)
    model = model.to(DEVICE)

    end_time = time.time()
    print("load model successful: " + str(end_time - start_time))
    return model, tokenizer


def test_accurate(model, tokenizer, train_name, test_name, test_file):
    start_time = time.time()

    human_correct = 0
    human_total = 0
    gpt_correct = 0
    gpt_total = 0
    i = 0

    with open(test_file, 'r', encoding='utf-8') as test_input:
        json_array = json.load(test_input)
        for json_obj in json_array:
            i += 1
            print('test process : %s [%d/%d]' % (str(i*100 / len(json_array)) + '%', i, len(json_array)), end='\r')
            raw_input = json_obj['content']
            raw_label = json_obj['label']
            inputs = tokenizer([raw_input], padding=True, truncation=True, return_tensors="pt").to(DEVICE)
            outputs = model(**inputs)
            pred_labels = outputs.logits.cpu().argmax(-1).numpy()
            pred_label = pred_labels[0]

            if raw_label == 0:
                human_total += 1
                if pred_label == 0:
                    human_correct += 1
            else:
                gpt_total += 1
                if pred_label == 1:
                    gpt_correct += 1

        end_time = time.time()
        print("test " + test_name + " successful: " + str(end_time - start_time))

        return {
            "train_name": train_name,
            "test_name": test_name,
            "total_acc": (1.0 * (human_correct + gpt_correct) / (human_total + gpt_total)),
            "human_acc": (1.0 * human_correct / human_total),
            "ai_acc": (1.0 * (gpt_correct) / (gpt_total))
        }

if __name__ == '__main__':
    # detector_config = load_detector_config()
    # base_test_model_config = load_test_base_model_config()
    # model, tokenizer = init_init_model_and_tokenizer(detector_config, base_test_model_config)
    # result = []
    # for name in detector_config['test_files']:
    #     result.append(test_accurate(model, tokenizer, 'medicine', name, detector_config['test_files'][name]))
    # print(result)

    # [{'train_name': 'medicine', 'test_name': 'finance', 'total_acc': 0.6019746991669238,
    #   'human_acc': 0.22153656278926256, 'ai_acc': 0.982412835544585},
    #  {'train_name': 'medicine', 'test_name': 'medicine', 'total_acc': 0.9930619796484736,
    #   'human_acc': 0.9888991674375578, 'ai_acc': 0.9972247918593895},
    #  {'train_name': 'medicine', 'test_name': 'wiki_csai', 'total_acc': 0.5284327323162274,
    #   'human_acc': 0.0638002773925104, 'ai_acc': 0.9930651872399445}]

    detector_config = load_detector_config('./tmp/train_1/', 'detector.json')
    base_test_model_config = load_test_base_model_config()
    model, tokenizer = init_cur_model_and_tokenizer(detector_config, base_test_model_config)
    result = []
    for name in detector_config['test_files']:
        result.append(test_accurate(model, tokenizer, 'medicine', name, detector_config['test_files'][name]))
    print(result)
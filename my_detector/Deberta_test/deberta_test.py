import json
import os
import time

from transformers import AutoModelForSequenceClassification, AutoTokenizer
import torch
import numpy

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def init_model_and_tokenizer(name):
    start_time = time.time()
    tokenizer=AutoTokenizer.from_pretrained('microsoft/deberta-v3-small', model_max_length=256)

    max_step = -1
    for file in os.listdir('./' + name):
        max_step = max(max_step, int(file.split('-')[1]))

    model_path = './' + name + "/checkpoint-" + str(max_step)
    print('load model:' + model_path)
    model = AutoModelForSequenceClassification.from_pretrained(model_path, num_labels=2)

    print(DEVICE)
    model = model.to(DEVICE)

    end_time = time.time()
    print("load model successful: " + str(end_time - start_time))
    return model, tokenizer

def test_accurate(model, tokenizer, name):
    human_correct = 0
    human_total = 0
    gpt_correct = 0
    gpt_total = 0
    i = 0
    with open('./data/' + name + '.jsonl.test', 'r', encoding='utf-8') as test_input:
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

        return {
            "name": name,
            "total_acc": (1.0 * (human_correct + gpt_correct) / (human_total + gpt_total)),
            "human_acc": (1.0 * human_correct / human_total),
            "ai_acc": (1.0 * (gpt_correct) / (gpt_total))
        }


def test_all():
    model, tokenizer = init_model_and_tokenizer('hc3_all')
    human_correct = 0
    human_total = 0
    gpt_correct = 0
    gpt_total = 0
    i = 0
    with open('./data/' + 'finance' + '.jsonl.test', 'r', encoding='utf-8') as test_input:
        json_array = json.load(test_input)
        for json_obj in json_array:
            i += 2
            print('test process : %s [%d/%d]' % (str(i / len(json_array)) + '%', i, len(json_array)), end='\r')
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

        print({
            "name": 'finance',
            "total_acc": (1.0 * (human_correct + gpt_correct) / (human_total + gpt_total)),
            "human_acc": (1.0 * human_correct / human_total),
            "ai_acc": (1.0 * (gpt_correct) / (gpt_total))
        })
    human_correct = 0
    human_total = 0
    gpt_correct = 0
    gpt_total = 0
    i = 0
    with open('./data/' + 'medicine' + '.jsonl.test', 'r', encoding='utf-8') as test_input:
        json_array = json.load(test_input)
        for json_obj in json_array:
            i += 2
            print('test process : %s [%d/%d]' % (str(i / len(json_array)) + '%', i, len(json_array)), end='\r')
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

        print({
            "name": 'medicine',
            "total_acc": (1.0 * (human_correct + gpt_correct) / (human_total + gpt_total)),
            "human_acc": (1.0 * human_correct / human_total),
            "ai_acc": (1.0 * (gpt_correct) / (gpt_total))
        })
    human_correct = 0
    human_total = 0
    gpt_correct = 0
    gpt_total = 0
    i = 0
    with open('./data/' + 'wiki_csai' + '.jsonl.test', 'r', encoding='utf-8') as test_input:
        json_array = json.load(test_input)
        for json_obj in json_array:
            i += 2
            print('test process : %s [%d/%d]' % (str(i / len(json_array)) + '%', i, len(json_array)), end='\r')
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

        print({
            "name": 'wiki_csai',
            "total_acc": (1.0 * (human_correct + gpt_correct) / (human_total + gpt_total)),
            "human_acc": (1.0 * human_correct / human_total),
            "ai_acc": (1.0 * (gpt_correct) / (gpt_total))
        })

def test(name):
    model, tokenizer = init_model_and_tokenizer(name)
    return test_accurate(model, tokenizer, name)


def test_cheat_all(name, datasets):
    model, tokenizer = init_model_and_tokenizer(name)
    for dataset in datasets:
        human_correct = 0
        human_total = 0
        gpt_correct = 0
        gpt_total = 0
        i = 0
        with open('./data/' + dataset + '.jsonl.test', 'r', encoding='utf-8') as test_input:
            json_array = json.load(test_input)
            for json_obj in json_array:
                i += 2
                print('test process : %s [%d/%d]' % (str(i / len(json_array)) + '%', i, len(json_array)), end='\r')
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

            print({
                "name": dataset,
                "total_acc": (1.0 * (human_correct + gpt_correct) / (human_total + gpt_total)),
                "human_acc": (1.0 * human_correct / human_total),
                "ai_acc": (1.0 * (gpt_correct) / (gpt_total))
            })

if __name__ == '__main__':
    # test_all()
    # print(test('medicine'))
    # print(test('finance'))
    # print(test('wiki_csai'))
    # print(test('hc3_all'))

    # print(test('ieee-chatgpt-fusion'))
    # print(test('ieee-chatgpt-generation'))
    # print(test('ieee-chatgpt-polish'))
    # print(test('cheat_all'))
    print(test_cheat_all('cheat_all', ['ieee-chatgpt-fusion', 'ieee-chatgpt-generation', 'ieee-chatgpt-polish']))


#
# {'name': 'medicine', 'total_acc': 0.9958376690946931, 'human_acc': 0.9937565036420395, 'ai_acc': 0.9979188345473465}
# {'name': 'finance', 'total_acc': 0.9908018049288442, 'human_acc': 0.984380423464075, 'ai_acc': 0.9972231863936133}
# {'name': 'wiki_csai', 'total_acc': 0.9695787831513261, 'human_acc': 0.9469578783151326, 'ai_acc': 0.9921996879875195}
# {'name': 'hc3_all', 'total_acc': 0.9792549631942895, 'human_acc': 0.9616328351550301, 'ai_acc': 0.996877091233549}

#hc3 all
# {'name': 'finance', 'total_acc': 0.9849010760152724, 'human_acc': 0.9732731690385283, 'ai_acc': 0.9965289829920166}
# {'name': 'medicine', 'total_acc': 0.9963579604578564, 'human_acc': 0.9927159209157128, 'ai_acc': 1.0}
# {'name': 'wiki_csai', 'total_acc': 0.9282371294851794, 'human_acc': 0.8627145085803433, 'ai_acc': 0.9937597503900156}

# {'name': 'ieee-chatgpt-fusion', 'total_acc': 0.67875, 'human_acc': 0.4634375, 'ai_acc': 0.8940625}
# {'name': 'ieee-chatgpt-generation', 'total_acc': 0.9728125, 'human_acc': 0.9459375, 'ai_acc': 0.9996875}
# {'name': 'ieee-chatgpt-polish', 'total_acc': 0.89515625, 'human_acc': 0.8065625, 'ai_acc': 0.98375}
# {'name': 'cheat_all', 'total_acc': 0.8398416460196058, 'human_acc': 0.7142626306151029, 'ai_acc': 0.9654206614241085}

# cheat_all
# {'name': 'ieee-chatgpt-fusion', 'total_acc': 0.65515625, 'human_acc': 0.45625, 'ai_acc': 0.8540625}
# {'name': 'ieee-chatgpt-generation', 'total_acc': 0.72703125, 'human_acc': 0.45625, 'ai_acc': 0.9978125}
# {'name': 'ieee-chatgpt-polish', 'total_acc': 0.72265625, 'human_acc': 0.45625, 'ai_acc': 0.9890625}
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

        return {
            "name": name,
            "total_acc": (1.0 * (human_correct + gpt_correct) / (human_total + gpt_total)),
            "human_acc": (1.0 * human_correct / human_total),
            "ai_acc": (1.0 * (gpt_correct) / (gpt_total))
        }

def test(name):
    model, tokenizer = init_model_and_tokenizer(name)
    return test_accurate(model, tokenizer, name)


if __name__ == '__main__':
    print(test('medicine'))
    print(test('finance'))
    print(test('wiki_csai'))
    print(test('hc3_all'))
#
# load model:./finance/checkpoint-1600
# cuda
# load model successful: 2.0808396339416504
# {'name': 'finance', 'total_acc': 0.9908018049288442, 'human_acc': 0.984380423464075, 'ai_acc': 0.9972231863936133}
# load model:./wiki_csai/checkpoint-200
# cuda
# load model successful: 2.097062826156616
# {'name': 'wiki_csai', 'total_acc': 0.9695787831513261, 'human_acc': 0.9469578783151326, 'ai_acc': 0.9921996879875195}
# load model:./hc3_all/checkpoint-4200
# cuda
# load model successful: 2.371985673904419
# {'name': 'hc3_all', 'total_acc': 0.9792549631942895, 'human_acc': 0.9616328351550301, 'ai_acc': 0.996877091233549}

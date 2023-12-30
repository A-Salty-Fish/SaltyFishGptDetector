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



def test_accurate(model, tokenizer, train_name , test_name, file_type='.jsonl'):
    human_correct = 0
    human_total = 0
    gpt_correct = 0
    gpt_total = 0
    i = 0
    with open('./data/' + train_name + "_" + test_name + file_type + '.acc', 'a', encoding='utf-8') as test_output:
        accurate_array = []
        with open('./data/' + test_name + file_type + '.test', 'r', encoding='utf-8') as test_input:
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
                        accurate_array.append(json_obj)
                else:
                    gpt_total += 1
                    if pred_label == 1:
                        gpt_correct += 1
                        accurate_array.append(json_obj)
            for acc in accurate_array:
                test_output.write(json.dumps(acc, ensure_ascii=False) + '\n')
            return {
                "test_dataset": test_name,
                "total_acc": (1.0 * (human_correct + gpt_correct) / (human_total + gpt_total)),
                "human_acc": (1.0 * human_correct / human_total),
                "ai_acc": (1.0 * (gpt_correct) / (gpt_total))
            }

def test_with_dataset(train_name, test_name, file_type = '.jsonl'):
    model, tokenizer = init_model_and_tokenizer(train_name)
    test_result = test_accurate(model, tokenizer, train_name , test_name, file_type)
    test_result['train_name'] = train_name
    return test_result


# 测试mask生成后的准确率
def test_with_fill_mask(model, tokenizer, train_name , test_name, file_type='.jsonl'):
    human_correct = 0
    human_total = 0
    gpt_correct = 0
    gpt_total = 0
    i = 0
    with open('./data/' + train_name + "_" + test_name + file_type + '.acc', 'a', encoding='utf-8') as test_output:
        accurate_array = []
        with open('./data/' + test_name + file_type + '.test', 'r', encoding='utf-8') as test_input:
            json_array = json.load(test_input)
            for json_obj in json_array:
                i += 1
                print('test process : %s [%d/%d]' % (str(i * 100 / len(json_array)) + '%', i, len(json_array)),
                      end='\r')
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
                        accurate_array.append(json_obj)
                else:
                    gpt_total += 1
                    if pred_label == 1:
                        gpt_correct += 1
                        accurate_array.append(json_obj)
            for acc in accurate_array:
                test_output.write(json.dumps(acc, ensure_ascii=False) + '\n')
            return {
                "test_dataset": test_name,
                "total_acc": (1.0 * (human_correct + gpt_correct) / (human_total + gpt_total)),
                "human_acc": (1.0 * human_correct / human_total),
                "ai_acc": (1.0 * (gpt_correct) / (gpt_total))
            }


def test_with_masked_data(model, tokenizer, train_name, test_name, file_type='.jsonl'):
    human_correct = 0
    human_total = 0
    gpt_correct = 0
    gpt_total = 0
    i = 0
    with open('./' + train_name + "_" + test_name + file_type, 'a', encoding='utf-8') as test_output:
        error_array = []
        for file_name in ['masked_result_1.jsonl', 'masked_result_2.jsonl', 'masked_result_3.jsonl']:
            with open('./' + file_name, 'r', encoding='utf-8') as test_input:
                for line in test_input:
                    try:
                        json_obj = json.loads(line)
                        i += 1
                        print('test process : %s' % (str(i)),
                              end='\r')
                        raw_input = json_obj['result_text']
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
                                error_array.append(json_obj)

                        else:
                            gpt_total += 1
                            if pred_label == 1:
                                gpt_correct += 1
                            else:
                                error_array.append(json_obj)

                    except Exception as e:
                        print(e)
                        continue
            for error_obj in error_array:
                test_output.write(json.dumps(error_obj) + '\n')
            return {
                "train": train_name,
                "test_dataset": test_name,
                "total_acc": (1.0 * (human_correct + gpt_correct) / (human_total + gpt_total)),
                "human_acc": (1.0 * human_correct / human_total),
                "ai_acc": (1.0 * (gpt_correct) / (gpt_total))
            }

def test_split_accurate(model, tokenizer, train_name , test_name, file_type='.jsonl'):
    human_correct = 0
    human_total = 0
    gpt_correct = 0
    gpt_total = 0
    i = 0
    with open('./data/' + train_name + "_" + test_name + file_type + '.split.acc', 'a', encoding='utf-8') as test_output:
        accurate_array = []
        with open('./data/' + test_name + file_type + '.test', 'r', encoding='utf-8') as test_input:
            json_array = json.load(test_input)
            for json_obj in json_array:
                i += 1
                print('test process : %s [%d/%d]' % (str(i*100 / len(json_array)) + '%', i, len(json_array)), end='\r')
                raw_input = json_obj['content']
                raw_label = json_obj['label']

                words = raw_input.split(' ')
                for begin_word in range(0, int(len(words) / 10) + 1):

                    split_input = " ".join(words[begin_word: begin_word + 10])
                    inputs = tokenizer([split_input], padding=True, truncation=True, return_tensors="pt").to(DEVICE)
                    outputs = model(**inputs)
                    pred_labels = outputs.logits.cpu().argmax(-1).numpy()
                    pred_label = pred_labels[0]
                    split_json_obj = {
                        'label': raw_label,
                        "content": split_input
                    }
                    if raw_label == 0:
                        human_total += 1
                        if pred_label == 0:
                            human_correct += 1
                            accurate_array.append(split_json_obj)
                    else:
                        gpt_total += 1
                        if pred_label == 1:
                            gpt_correct += 1
                            accurate_array.append(split_json_obj)
            for acc in accurate_array:
                test_output.write(json.dumps(acc, ensure_ascii=False) + '\n')
            return {
                "test_dataset": test_name,
                "total_acc": (1.0 * (human_correct + gpt_correct) / (human_total + gpt_total)),
                "human_acc": (1.0 * human_correct / human_total),
                "ai_acc": (1.0 * (gpt_correct) / (gpt_total))
            }


if __name__ == '__main__':
    # print(test_with_dataset('finance', 'finance'))
    # print(test_with_dataset('finance', 'medicine'))
    # print(test_with_dataset('finance', 'wiki_csai'))
    # print(test_with_dataset('medicine', 'medicine'))
    # print(test_with_dataset('medicine', 'finance'))
    # print(test_with_dataset('medicine', 'wiki_csai'))
    # print(test_with_dataset('wiki_csai', 'wiki_csai'))
    # print(test_with_dataset('wiki_csai', 'finance'))
    # print(test_with_dataset('wiki_csai', 'medicine'))

    # {'test_dataset': 'finance', 'total_acc': 0.9939833384757791, 'human_acc': 0.9907435976550447, 'ai_acc': 0.9972230792965134, 'train_name': 'finance'}
    # {'test_dataset': 'medicine', 'total_acc': 0.7562442183163737, 'human_acc': 0.5291396854764108, 'ai_acc': 0.9833487511563367, 'train_name': 'finance'}
    # {'test_dataset': 'wiki_csai', 'total_acc': 0.6789181692094314, 'human_acc': 0.3606102635228849, 'ai_acc': 0.9972260748959778, 'train_name': 'finance'}
    # {'test_dataset': 'medicine', 'total_acc': 0.9912118408880666, 'human_acc': 0.9953746530989824, 'ai_acc': 0.9870490286771508, 'train_name': 'medicine'}
    # {'test_dataset': 'finance', 'total_acc': 0.7062634989200864, 'human_acc': 0.45788336933045354, 'ai_acc': 0.9546436285097192, 'train_name': 'medicine'}
    # {'test_dataset': 'wiki_csai', 'total_acc': 0.6095700416088765, 'human_acc': 0.231622746185853, 'ai_acc': 0.9875173370319001, 'train_name': 'medicine'}
    # {'test_dataset': 'wiki_csai', 'total_acc': 0.9403606102635229, 'human_acc': 0.9001386962552012, 'ai_acc': 0.9805825242718447, 'train_name': 'wiki_csai'}
    # {'test_dataset': 'finance', 'total_acc': 0.8128663992594878, 'human_acc': 0.6467139771675409, 'ai_acc': 0.9790188213514347, 'train_name': 'wiki_csai'}
    # {'test_dataset': 'medicine', 'total_acc': 0.8580018501387604, 'human_acc': 0.7298797409805735, 'ai_acc': 0.9861239592969473, 'train_name': 'wiki_csai'}

    model, tokenizer = init_model_and_tokenizer('medicine')
    # test_with_masked_data(model, tokenizer,  'medicine', 'masked_1')
    test_split_accurate(model, tokenizer, 'medicine', 'medicine')
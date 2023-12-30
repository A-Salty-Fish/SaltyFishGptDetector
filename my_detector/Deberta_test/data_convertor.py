import json
import os
import random


def convert_hc3_dataset_to_train_and_test(row_path, row_name ,target_path, total_num, train_rate):
    i = 0
    train_list = []
    test_list = []
    with open(row_path + row_name, 'r', encoding='utf-8') as in_file:
        for line in in_file:
            i+=1
            json_obj = json.loads(line)
            human_obj = {}
            gpt_obj = {}
            human_obj['label'] = 0
            gpt_obj['label'] = 1
            human_obj['content'] = json_obj['human_answers'][0]
            gpt_obj['content'] = json_obj['chatgpt_answers'][0]
            if i > total_num:
                break
            if i < train_rate * total_num:
                train_list.append(human_obj)
                train_list.append(gpt_obj)
            else:
                test_list.append(human_obj)
                test_list.append(gpt_obj)
    with open(target_path + row_name + ".test", 'w', encoding='utf-8') as out_test_file:
        out_test_file.write(json.dumps(test_list))
    with open(target_path + row_name + ".train", 'w', encoding='utf-8') as out_train_file:
        out_train_file.write(json.dumps(train_list))

def merge_hc3_dataset(target_path):
    test_list = []
    train_list = []
    for file in  os.listdir(target_path):
        with open(target_path + file, 'r', encoding='utf-8') as f:
            arr = json.load(f)
            if file.find('test') != -1:
                for obj in arr:
                    test_list.append(obj)
            else:
                for obj in arr:
                    train_list.append(obj)
    with open(target_path + "hc3_all.jsonl.test", 'w', encoding='utf-8') as out_test_file:
        out_test_file.write(json.dumps(test_list))
    with open(target_path + "hc3_all.jsonl.train", 'w', encoding='utf-8') as out_train_file:
        out_train_file.write(json.dumps(train_list))



def convert_CHEAT_dataset_to_train_and_test(row_path, row_name, target_path, total_num, train_rate):
    i = 0
    train_list = []
    test_list = []
    with open(row_path + row_name, 'r', encoding='utf-8') as in_file_1:
        for line in in_file_1:
            i+=1
            if i > total_num:
                break
            json_obj = json.loads(line)
            gpt_obj = {}
            gpt_obj['label'] = 1
            gpt_obj['content'] = json_obj['abstract']
            if i <= total_num * train_rate:
                train_list.append(gpt_obj)
            else:
                test_list.append(gpt_obj)
    i = 0
    with open(row_path + 'ieee-init.jsonl', 'r', encoding='utf-8') as in_file_2:
        for line in in_file_2:
            i += 1
            if i > total_num:
                break
            json_obj = json.loads(line)
            human_obj = {}
            human_obj['label'] = 0
            human_obj['content'] = json_obj['abstract']
            if i <= total_num * train_rate:
                train_list.append(human_obj)
            else:
                test_list.append(human_obj)
    random.shuffle(test_list)
    random.shuffle(train_list)
    with open(target_path + row_name + '.test', 'w', encoding='utf-8') as out_test_file:
        out_test_file.write(json.dumps(test_list))
    with open(target_path + row_name + '.train', 'w', encoding='utf-8') as out_train_file:
        out_train_file.write(json.dumps(train_list))


def merge_CHEAT_dataset(target_path):
    test_list = []
    train_list = []
    for file in os.listdir(target_path):
        if file.find('ieee-chatgpt') == -1:
            continue
        with open(target_path + file, 'r', encoding='utf-8') as f:
            arr = json.load(f)
            if file.find('test') != -1:
                for obj in arr:
                    test_list.append(obj)
            else:
                for obj in arr:
                    train_list.append(obj)
    print(len(train_list))
    print(len(test_list))
    with open(target_path + "cheat_all.jsonl.test", 'w', encoding='utf-8') as out_test_file:
        out_test_file.write(json.dumps(test_list))
    with open(target_path + "cheat_all.jsonl.train", 'w', encoding='utf-8') as out_train_file:
        out_train_file.write(json.dumps(train_list))


def convert_ghostbuster_dataset_to_train_and_test(row_path, row_name, target_path, total_num, train_rate):
    i = 0
    train_list = []
    test_list = []
    with open(row_path + row_name, 'r', encoding='utf-8') as in_file_1:
        for line in in_file_1:
            i += 1
            if i > total_num:
                break
            gpt_obj = {}
            gpt_obj['label'] = 1
            gpt_obj['content'] = line
            if i <= total_num * train_rate:
                train_list.append(gpt_obj)
            else:
                test_list.append(gpt_obj)
    i = 0
    with open(row_path + 'essay_human.txt', 'r', encoding='utf-8') as in_file_2:
        for line in in_file_2:
            i += 1
            if i > total_num:
                break
            human_obj = {}
            human_obj['label'] = 0
            human_obj['content'] = line
            if i <= total_num * train_rate:
                train_list.append(human_obj)
            else:
                test_list.append(human_obj)
    with open(target_path + row_name + '.test', 'w', encoding='utf-8') as out_test_file:
        out_test_file.write(json.dumps(test_list))
    with open(target_path + row_name + '.train', 'w', encoding='utf-8') as out_train_file:
        out_train_file.write(json.dumps(train_list))



def convert_m4_dataset_to_train_and_test(row_path, row_name, target_path, total_num, train_rate):
    i = 0
    train_list = []
    test_list = []
    with open(row_path + row_name, 'r', encoding='utf-8') as in_file_1:
        for line in in_file_1:
            i+=1
            if i > total_num:
                break
            try:
                json_obj = json.loads(line)
                if isinstance(json_obj['human_text'], list):
                    human_text = json_obj['human_text'][0].replace('\n', '')
                else:
                    human_text = json_obj['human_text'].replace('\n', '')
                if isinstance(json_obj['machine_text'], list):
                    machine_text = json_obj['machine_text'][0].replace('\n', '')
                else:
                    machine_text = json_obj['machine_text'].replace('\n', '')
                human_obj = {
                    'content': human_text,
                    'label': 0
                }
                machine_obj = {
                    'content': machine_text,
                    'label': 1
                }
                if i <= total_num * train_rate:
                    train_list.append(human_obj)
                    train_list.append(machine_obj)
                else:
                    test_list.append(human_obj)
                    test_list.append(machine_obj)
            except Exception as e:
                print(e)

    with open(target_path + row_name + '.test', 'w', encoding='utf-8') as out_test_file:
        out_test_file.write(json.dumps(test_list))
    with open(target_path + row_name + '.train', 'w', encoding='utf-8') as out_train_file:
        out_train_file.write(json.dumps(train_list))

def merge_m4(target_path):
    test_list = []
    train_list = []
    for file in os.listdir(target_path):
        if file.find('reddit') == -1 or file.find('reddit_all') != -1:
            continue
        with open(target_path + file, 'r', encoding='utf-8') as f:
            arr = json.load(f)
            print(file + ":" + str(len(arr)))
            if file.find('test') != -1:
                for obj in arr:
                    test_list.append(obj)
            else:
                for obj in arr:
                    train_list.append(obj)
    with open(target_path + "reddit_all.jsonl.test", 'w', encoding='utf-8') as out_test_file:
        out_test_file.write(json.dumps(test_list))
    with open(target_path + "reddit_all.jsonl.train", 'w', encoding='utf-8') as out_train_file:
        out_train_file.write(json.dumps(train_list))

def merge_ghostbuster(target_path):
    test_list = []
    train_list = []
    for file in os.listdir(target_path):
        if file.find('essay') == -1 and file.find('.txt') == -1:
            continue
        with open(target_path + file, 'r', encoding='utf-8') as f:
            arr = json.load(f)
            if file.find('test') != -1:
                for obj in arr:
                    test_list.append(obj)
            else:
                for obj in arr:
                    train_list.append(obj)
    with open(target_path + "ghostbuster_all.txt.test", 'w', encoding='utf-8') as out_test_file:
        out_test_file.write(json.dumps(test_list))
    with open(target_path + "ghostbuster_all.txt.train", 'w', encoding='utf-8') as out_train_file:
        out_train_file.write(json.dumps(train_list))





if __name__ == '__main__':
    pass
    finance_total = 3600
    medicine_total = 1200
    wiki_total = 800
    convert_hc3_dataset_to_train_and_test('../../data_collector/test_data/hc3_english/', 'finance.jsonl', '../token_test/data/', finance_total, 0.1)
    convert_hc3_dataset_to_train_and_test('../../data_collector/test_data/hc3_english/', 'medicine.jsonl', '../token_test/data/', medicine_total, 0.1)
    convert_hc3_dataset_to_train_and_test('../../data_collector/test_data/hc3_english/', 'wiki_csai.jsonl', '../token_test/data/', wiki_total, 0.1)
    merge_hc3_dataset('../token_test/data/')
    # total = 4000
    # convert_CHEAT_dataset_to_train_and_test('../../data_collector/test_data/CHEAT/', 'ieee-chatgpt-fusion.jsonl', './data/', total, 0.2)
    # convert_CHEAT_dataset_to_train_and_test('../../data_collector/test_data/CHEAT/', 'ieee-chatgpt-generation.jsonl', './data/', total, 0.2)
    # convert_CHEAT_dataset_to_train_and_test('../../data_collector/test_data/CHEAT/', 'ieee-chatgpt-polish.jsonl', './data/', total, 0.2)
    # merge_CHEAT_dataset('./data/')

    # total = 1000
    # convert_ghostbuster_dataset_to_train_and_test('../../data_collector/test_data/ghostbuster/', 'essay_claude.txt', './data/' ,total, 0.2)
    # convert_ghostbuster_dataset_to_train_and_test('../../data_collector/test_data/ghostbuster/', 'essay_gpt.txt', './data/' ,total, 0.2)
    # merge_ghostbuster('./data/')

    # total = 3000
    # convert_m4_dataset_to_train_and_test('../../data_collector/test_data/m4/', 'reddit_chatGPT.jsonl', './data/', total ,0.2)
    # convert_m4_dataset_to_train_and_test('../../data_collector/test_data/m4/', 'reddit_flant5.jsonl', './data/', total ,0.2)
    # convert_m4_dataset_to_train_and_test('../../data_collector/test_data/m4/', 'reddit_cohere.jsonl', './data/', total ,0.2)
    # convert_m4_dataset_to_train_and_test('../../data_collector/test_data/m4/', 'reddit_davinci.jsonl', './data/', total ,0.2)
    # convert_m4_dataset_to_train_and_test('../../data_collector/test_data/m4/', 'reddit_dolly.jsonl', './data/', total ,0.2)
    # merge_m4('./data/')
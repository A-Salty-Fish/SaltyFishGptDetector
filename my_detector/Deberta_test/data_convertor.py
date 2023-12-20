import json
import os


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


if __name__ == '__main__':

    finance_total = 3600
    medicine_total = 1200
    wiki_total = 800
    # convert_hc3_dataset_to_train_and_test('../../data_collector/test_data/hc3_english/', 'finance.jsonl', './data/', finance_total, 0.2)
    # convert_hc3_dataset_to_train_and_test('../../data_collector/test_data/hc3_english/', 'medicine.jsonl', './data/', medicine_total, 0.2)
    # convert_hc3_dataset_to_train_and_test('../../data_collector/test_data/hc3_english/', 'wiki_csai.jsonl', './data/', wiki_total, 0.2)
    merge_hc3_dataset('./data/')

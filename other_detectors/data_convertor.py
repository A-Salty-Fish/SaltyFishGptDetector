# 用来将不同格式的数据集统一格式
import json
import os


# 统一后的格式为

# {
#     "content": "xxxx",
#     "label": 0
# }

# 其中content为文本，label为标识，0为人类，1为AI

def convert_CHEAT_dataset(CHEAT_path):
    result = []
    with open(CHEAT_path + "/" + "ieee-chatgpt-fusion.jsonl", 'r', encoding='utf-8') as fusion_file:
        pass
    with open(CHEAT_path + "/" + "ieee-chatgpt-generation.jsonl", 'r', encoding='utf-8') as generation_file:
        for line in generation_file:
            json_obj = json.loads(line)
            result.append({
                'content': json_obj['abstract'],
                'label': 1
            })
    with open(CHEAT_path + "/" + "ieee-chatgpt-polish.jsonl", 'r', encoding='utf-8') as polish_file:
        pass
    with open(CHEAT_path + "/" + "ieee-init.jsonl", 'r', encoding='utf-8') as init_file:
        for line in init_file:
            json_obj = json.loads(line)
            result.append({
                'content': json_obj['abstract'],
                'label': 0
            })
    return result


def convert_ghostbuster_dataset(ghostbuster_path):
    result = []
    for file in os.listdir(ghostbuster_path):
        if file.find('human') != -1:
            with open(ghostbuster_path + '/' + file, 'r', encoding='utf-8') as f:
                for line in f:
                    if len(line.replace('\n', '').strip()) == 0:
                        continue
                    result.append({
                        'content': line,
                        'label': 0
                    })
        else:
            with open(ghostbuster_path + '/' + file, 'r', encoding='utf-8') as f:
                for line in f:
                    if len(line.replace('\n', '').strip()) == 0:
                        continue
                    result.append({
                        'content': line,
                        'label': 1
                    })
    return result


def convert_hc3_english(hc3_english_path):
    result = []
    for file in os.listdir(hc3_english_path):
        if file.find('.jsonl') == -1:
            continue
        with open(hc3_english_path + '/' + file, 'r', encoding='utf-8') as f:
            for line in f:
                json_obj = json.loads(line)
                human_answer = json_obj['human_answers'][0].replace('\n', '')
                chatgpt_answer = json_obj['chatgpt_answers'][0].replace('\n', '')
                result.append({
                    'content': chatgpt_answer,
                    'label': 1
                })
                result.append({
                    'content': human_answer,
                    'label': 0
                })
    return result


def convert_hc3_plus_english(hc3_plus_english_path):
    result = []
    with open(hc3_plus_english_path + '/' + 'test_hc3_QA.jsonl', 'r', encoding='utf-8') as qa_test_file:
        for line in qa_test_file:
            json_obj = json.loads(line)
            result.append({
                'content': json_obj['text'],
                'label': json_obj['label']
            })
    with open(hc3_plus_english_path + '/' + 'test_hc3_si.jsonl', 'r', encoding='utf-8') as si_test_file:
        for line in si_test_file:
            json_obj = json.loads(line)
            result.append({
                'content': json_obj['text'],
                'label': json_obj['label']
            })
    return result


def convert_m4(m4_path):
    result = []
    for file in os.listdir(m4_path):
        if file.find('.jsonl') == -1:
            continue
        if file.find('bloomz') != -1:
            continue
        with open(m4_path + '/' + file, 'r', encoding='utf-8') as f:
            for line in f:
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
                    result.append({
                        'content': machine_text,
                        'label': 1
                    })
                    result.append({
                        'content': human_text,
                        'label': 0
                    })
                except Exception as e:
                    print(e)
                    print(file + ":" + line)
    return result

if __name__ == '__main__':
    # print(convert_CHEAT_dataset("../data_collector/test_data/CHEAT"))
    # print(len(convert_ghostbuster_dataset('../data_collector/test_data/ghostbuster')))
    # print(len(convert_hc3_english('..\\data_collector\\test_data\\hc3_english')))
    # print(len(convert_hc3_plus_english('..\\data_collector\\test_data\\hc3_plus_english')))
    # print(len(convert_m4('..\\data_collector\\test_data\\m4')))
    pass
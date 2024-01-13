import json
import time

from transformers import pipeline


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


def load_text_labels_config(base_path='./config/', config_name='generator.json'):
    with open(base_path + config_name, 'r', encoding='utf-8') as text_labels_file:
        return json.load(text_labels_file)['candidate_labels']


# 将hcc3 wiki qa 和 token化的人类文本混合入数据集中
def merge_hc3_and_wiki_qa_and_tokenized_human():
    hc3_file_names = ['finance', 'medicine', 'open_qa', 'wiki_csai']
    i = 0
    for hc3_file_name in hc3_file_names:
        with open('./data/' + hc3_file_name + '.mix.jsonl', 'r', encoding='utf-8') as f:
            for line in f:
                i += 1
    print(i)
    with open('./data/hc3_mix_label_result', 'r', encoding='utf-8') as hc3_mix_f, open(
            './data/hc3_mix_token_fill_1_result', 'r', encoding='utf-8') as hc3_token_human_f:
        hc3_mix_arr = json.load(hc3_mix_f)
        hc3_token_human_arr = json.load(hc3_token_human_f)
        print(len(hc3_token_human_arr))


def utc_classify(classifier, labels, text):
    result = []
    output = classifier(text, labels, multi_label=False)
    for i in range(0, len(output['labels'])):
        result.append([output['labels'][i], output['scores'][i]])
    if len(result) == 0:
        result.append(['None', 1.00])
    return result


def output_utc_datas(classifier, labels, datas, output_file, top_k=3):
    label_contents = {}
    for label in labels:
        label_contents[label] = []
    i = 0
    for data in datas:
        try:
            i += 1
            print('process : %s' % (str(i)), end='\r')
            utc_result = utc_classify(classifier, labels, data['content'])
            for ii in range(0, top_k):
                label_contents[utc_result[ii][0]].append({
                    'label': data['label'],
                    'content': data['content']
                })
        except Exception as e:
            print(e)
            print(utc_result)
    with open(output_file, 'w', encoding='utf-8') as out_f:
        out_f.write(json.dumps(label_contents))


def load_wiki_qa_mix_data():
    results = []
    with open('./data/wiki_qa_mix.jsonl', 'r', encoding='utf-8') as f:
        for line in f:
            json_obj = json.loads(line)
            results.append({
                'label': 0,
                'content': json_obj['human']
            })
            results.append({
                'label': 1,
                'content': json_obj['ai']
            })
    return results


if __name__ == '__main__':
    # merge_hc3_and_wiki_qa_and_tokenized_human()

    output_utc_datas(
        init_utc_pipe(load_utc_base_model_config()),
        load_text_labels_config(),
        load_wiki_qa_mix_data(),
        './label_wiki_qa'
    )
    pass
import json
import time

from transformers import pipeline


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


def load_text_labels_config(base_path='./config/', config_name='generator.json'):
    with open(base_path + config_name, 'r', encoding='utf-8') as text_labels_file:
        return json.load(text_labels_file)['candidate_labels']


# 将hcc3 wiki qa 和 token化的人类文本混合入数据集中
def merge_hc3_and_wiki_qa_and_tokenized_human():
    hc3_file_names = ['finance', 'medicine', 'open_qa', 'wiki_csai']
    i = 0
    for hc3_file_name in hc3_file_names:
        with open('./data/' + hc3_file_name + '.mix.jsonl', 'r', encoding='utf-8') as f:
            for line in f:
                i += 1
    print(i)
    with open('./data/hc3_mix_label_result', 'r', encoding='utf-8') as hc3_mix_f, open(
            './data/hc3_mix_token_fill_1_result', 'r', encoding='utf-8') as hc3_token_human_f:
        hc3_mix_arr = json.load(hc3_mix_f)
        hc3_token_human_arr = json.load(hc3_token_human_f)
        print(len(hc3_token_human_arr))


def utc_classify(classifier, labels, text):
    result = []
    output = classifier(text, labels, multi_label=False)
    for i in range(0, len(output['labels'])):
        result.append([output['labels'][i], output['scores'][i]])
    if len(result) == 0:
        result.append(['None', 1.00])
    return result


def output_utc_datas(classifier, labels, datas, output_file, top_k=3):
    label_contents = {}
    for label in labels:
        label_contents[label] = []
    i = 0
    for data in datas:
        try:
            i += 1
            print('process : %s' % (str(i)), end='\r')
            utc_result = utc_classify(classifier, labels, data['content'])
            for ii in range(0, top_k):
                label_contents[utc_result[ii][0]].append({
                    'label': data['label'],
                    'content': data['content']
                })
        except Exception as e:
            print(e)
            print(utc_result)
    with open(output_file, 'w', encoding='utf-8') as out_f:
        out_f.write(json.dumps(label_contents))


def load_wiki_qa_mix_data():
    results = []
    with open('./data/wiki_qa_mix.jsonl', 'r', encoding='utf-8') as f:
        for line in f:
            json_obj = json.loads(line)
            results.append({
                'label': 0,
                'content': json_obj['human']
            })
            results.append({
                'label': 1,
                'content': json_obj['ai']
            })
    return results


if __name__ == '__main__':
    # merge_hc3_and_wiki_qa_and_tokenized_human()

    output_utc_datas(
        init_utc_pipe(load_utc_base_model_config()),
        load_text_labels_config(),
        load_wiki_qa_mix_data(),
        './label_wiki_qa'
    )
    pass

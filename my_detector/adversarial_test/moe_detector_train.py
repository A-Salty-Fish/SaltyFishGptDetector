import json
import time

from transformers import pipeline

def load_moe_detector_config(base_path='./tmp/moe/', file_name = 'moe_detector.json'):
    with open(base_path + file_name, 'r' , encoding='utf-8') as config_file:
        return json.load(config_file)


def load_text_labels_config(base_path='./config/', config_name='generator.json'):
    with open(base_path + config_name, 'r', encoding='utf-8') as text_labels_file:
        return json.load(text_labels_file)['candidate_labels']

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

def train_single():

    pass


def prepare_data_utc(labels, classifier, top_k=3):
    label_contents = {}
    for label in labels:
        label_contents[label] = []
    i = 0
    with open('./data/cp_wiki_qa_mix.jsonl', 'r', encoding='utf-8') as f:
        for line in f:
            i += 1
            print('process : %s' % (str(i)), end='\r')
            json_obj = json.loads(line)
            ai_content = json_obj['ai'].replace('\n', '')
            try:
                utc_result = utc_classify(classifier, labels, ai_content)
                for ii in range(0, top_k):
                    label_contents[utc_result[ii][0]].append(json_obj)
            except Exception as e:
                print(e)
                print(ai_content)
                print(utc_result)

    with open('./label_result', 'w', encoding='utf-8') as f_out:
        f_out.write(json.dumps(label_contents, ensure_ascii=False))


if __name__ == '__main__':
    prepare_data_utc(labels=load_text_labels_config(), classifier=init_utc_pipe(load_utc_base_model_config()))
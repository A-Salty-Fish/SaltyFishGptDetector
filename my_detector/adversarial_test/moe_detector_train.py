import json
import time

from datasets import load_dataset
from sklearn.metrics import f1_score, accuracy_score
from transformers import pipeline, AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments
import torch

def load_moe_detector_config(base_path='./tmp/moe/', file_name = 'moe_detector.json'):
    with open(base_path + file_name, 'r' , encoding='utf-8') as config_file:
        return json.load(config_file)


def load_text_labels_config(base_path='./config/', config_name='generator.json'):
    with open(base_path + config_name, 'r', encoding='utf-8') as text_labels_file:
        return json.load(text_labels_file)['candidate_labels']

def load_utc_base_model_config(base_path='./config/', config_name='base_model.json'):
    with open(base_path + config_name, 'r', encoding='utf-8') as text_labels_file:
        return json.load(text_labels_file)['utc']

def load_train_base_model_config(base_path='./config/', config_name='base_model.json'):
    with open(base_path + config_name, 'r', encoding='utf-8') as text_labels_file:
        return json.load(text_labels_file)['train']


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

# 加载模型和配置
def init_base_model_and_tokenizer(base_train_model_config):
    start_time = time.time()
    model_name = base_train_model_config['model_name']
    tokenizer_name = base_train_model_config['tokenizer_name']
    max_length = base_train_model_config['max_length']

    tokenizer = AutoTokenizer.from_pretrained(model_name, model_max_length=max_length)
    model = AutoModelForSequenceClassification.from_pretrained(tokenizer_name)
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(DEVICE)
    model = model.to(DEVICE)

    end_time = time.time()
    print("load model successful: " + str(end_time - start_time))
    return model, tokenizer


def load_local_train_dataset(train_file):
    start_time = time.time()

    local_dataset = load_dataset('json', data_files={
        'train': train_file,
        # 'test': test_file,
    })
    end_time = time.time()
    print("load dataset successful: " + str(end_time - start_time))
    return local_dataset


def tokenize_data(tokenizer, local_dataset):
    start_time = time.time()

    def tokenize(batch):
        return tokenizer(batch["content"], padding=True, truncation=True)

    tokenized_data = local_dataset.map(tokenize, batched=True, batch_size=None)
    tokenized_data.set_format("torch", columns=["input_ids", "attention_mask", "label"])

    end_time = time.time()
    print("tokenize dataset successful: " + str(end_time - start_time))
    return tokenized_data


def compute_metrics(pred):
    labels = pred.label_ids
    preds = pred.predictions.argmax(-1)
    f1 = f1_score(labels, preds, average="weighted")
    acc = accuracy_score(labels, preds)
    return {"accuracy": acc, "f1": f1}


def convert_train_config_to_train_args(train_config, label):
    train_config = train_config['cur']
    output_dir = train_config['output_dir'] + label
    num_train_epochs = train_config['num_train_epochs']
    save_total_limit = train_config['save_total_limit']
    eval_steps = train_config['eval_steps']
    save_steps = train_config['save_steps']
    evaluation_strategy = train_config['evaluation_strategy']
    training_args = TrainingArguments(output_dir=output_dir, num_train_epochs=num_train_epochs,
                                      load_best_model_at_end=True,
                                      save_total_limit=save_total_limit, eval_steps=eval_steps,
                                      evaluation_strategy=evaluation_strategy,
                                      save_steps=save_steps)
    return training_args

def train_single(label,
                 base_model,
                 base_tokenizer,
                 train_path,
                 detector_config
                 ):
    print("begin train: " + label)
    start_time = time.time()

    model, tokenizer = base_model, base_tokenizer

    local_dataset = load_local_train_dataset(train_path + label + '.train')

    tokenized_data = tokenize_data(tokenizer, local_dataset)

    training_args = convert_train_config_to_train_args(train_config=detector_config, label=label)

    trainer = Trainer(model=model, args=training_args, compute_metrics=compute_metrics,
                      train_dataset=tokenized_data["train"],
                      eval_dataset=tokenized_data["train"]
                      )

    trainer.train()
    end_time = time.time()
    print("train " + label + " successful " + " : " + str(end_time - start_time))
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


def prepare_hc3_data_utc(labels, classifier, top_k=3):
    label_contents = {}
    for label in labels:
        label_contents[label] = []
    i = 0
    hc3_file_names = ['finance', 'medicine', 'open_qa', 'wiki_csai']
    for hc3_file_name in hc3_file_names:
        with open('./data/' + hc3_file_name +'.mix.jsonl', 'r', encoding='utf-8') as f:
            for line in f:
                i += 1
                print('process : %s' % (str(i)), end='\r')
                json_obj = json.loads(line)
                # todo ai only
                ai_content = json_obj['ai'].replace('\n', '')
                try:
                    utc_result = utc_classify(classifier, labels, ai_content)
                    for ii in range(0, top_k):
                        label_contents[utc_result[ii][0]].append(json_obj)
                except Exception as e:
                    print(e)
                    print(ai_content)
                    print(utc_result)

    with open('./hc3_mix_label_result', 'w', encoding='utf-8') as f_out:
        f_out.write(json.dumps(label_contents, ensure_ascii=False))


# 测试用，用于生成测试数据
def prepare_test_utc_datas(labels, target_path, train_rate = 0.2):
    all_train_datas = []
    all_test_datas = []
    with open('./data/label_result', 'r', encoding='utf-8') as f:
        json_arr = json.load(f)
        for label in labels:
            tmp_label_datas = json_arr[label]
            label_datas = []
            for tmp_label_data in tmp_label_datas:
                label_datas.append({
                    "content": tmp_label_data['human'].replace('\n', ''),
                    "label": 0
                })
                label_datas.append({
                    "content": tmp_label_data['ai'].replace('\n', ''),
                    "label": 1
                })
            train_datas = label_datas[0: int(len(label_datas) * train_rate)]
            test_datas = label_datas[int(len(label_datas) * train_rate):]
            all_train_datas += train_datas
            all_test_datas += test_datas
            with open(target_path + label + '.train', 'w', encoding='utf-8') as train_f:
                train_f.write(json.dumps(train_datas, ensure_ascii=False))
            with open(target_path + label + '.test', 'w', encoding='utf-8') as test_f:
                test_f.write(json.dumps(test_datas, ensure_ascii=False))
        with open(target_path + 'all' + '.train', 'w', encoding='utf-8') as train_f:
            train_f.write(json.dumps(all_train_datas, ensure_ascii=False))
        with open(target_path + 'all' + '.test', 'w', encoding='utf-8') as test_f:
            test_f.write(json.dumps(all_test_datas, ensure_ascii=False))

if __name__ == '__main__':
    # prepare_data_utc(labels=load_text_labels_config(), classifier=init_utc_pipe(load_utc_base_model_config()))
    prepare_hc3_data_utc(labels=load_text_labels_config(), classifier=init_utc_pipe(load_utc_base_model_config()))
    # for label in load_text_labels_config():
    #     base_model, base_tokenizer = init_base_model_and_tokenizer(load_train_base_model_config())
    #     train_single(
    #         label,
    #         base_model,
    #         base_tokenizer,
    #         './tmp/train_1/',
    #         load_moe_detector_config()
    #     )
    # base_model, base_tokenizer = init_base_model_and_tokenizer(load_train_base_model_config())
    # train_single(
    #     'all',
    #     base_model,
    #     base_tokenizer,
    #     './tmp/train_1/',
    #     load_moe_detector_config()
    # )

    pass
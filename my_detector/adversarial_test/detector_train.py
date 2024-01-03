import json
import time

from datasets import load_dataset
from sklearn.metrics import f1_score, accuracy_score
from transformers import TrainingArguments, AutoTokenizer, AutoModelForSequenceClassification, Trainer
import torch



# 初始化初始训练配置
def load_detector_init_train_config(base_path='./config/', config_name='detector.json'):
    with open(base_path + config_name, 'r', encoding='utf-8') as text_labels_file:
        return json.load(text_labels_file)['init']


def load_detector_cur_train_config(base_path='./config/', config_name='detector.json'):
    with open(base_path + config_name, 'r', encoding='utf-8') as text_labels_file:
        return json.load(text_labels_file)['cur']


def load_train_base_model_config(base_path='./config/', config_name='base_model.json'):
    with open(base_path + config_name, 'r', encoding='utf-8') as text_labels_file:
        return json.load(text_labels_file)['train']

def convert_train_config_to_train_args(train_config):
    output_dir = train_config['output_dir']
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


def load_local_dataset(train_file, test_file):
    start_time = time.time()

    local_dataset = load_dataset('json', data_files={
        'train': train_file,
        'test': test_file,
    })
    end_time = time.time()
    print("load dataset successful: " + str(end_time - start_time))
    return local_dataset


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


def init_train(base_train_model_config, detector_init_train_config):
    print("begin init train")
    start_time = time.time()

    model, tokenizer = init_base_model_and_tokenizer(base_train_model_config)

    train_file = detector_init_train_config['train_file']
    test_file = detector_init_train_config['test_file']
    local_dataset = load_local_dataset(train_file, test_file)

    tokenized_data = tokenize_data(tokenizer, local_dataset)

    training_args = convert_train_config_to_train_args(train_config=detector_init_train_config)

    trainer = Trainer(model=model, args=training_args, compute_metrics=compute_metrics,
                      train_dataset=tokenized_data["train"],
                      eval_dataset=tokenized_data["train"]
                      )

    end_time = time.time()
    print("train init successful " + " : " + str(end_time - start_time))
    trainer.train()



def adversarial_train(base_train_model_config, detector_train_config):
    print("begin train")
    start_time = time.time()

    model, tokenizer = init_base_model_and_tokenizer(base_train_model_config)

    train_file = detector_train_config['train_file']
    test_file = detector_train_config['test_file']
    local_dataset = load_local_dataset(train_file, test_file)

    tokenized_data = tokenize_data(tokenizer, local_dataset)

    training_args = convert_train_config_to_train_args(train_config=detector_train_config)

    trainer = Trainer(model=model, args=training_args, compute_metrics=compute_metrics,
                      train_dataset=tokenized_data["train"],
                      eval_dataset=tokenized_data["train"]
                      )

    end_time = time.time()
    print("train successful " + " : " + str(end_time - start_time))
    trainer.train()



if __name__ == '__main__':
    # print(load_detector_init_train_config())
    # print(convert_train_config_to_train_args(load_detector_init_train_config()))

    # base_model_config = load_train_base_model_config()
    # detector_init_train_config = load_detector_init_train_config()
    # init_train(base_model_config, detector_init_train_config)

    base_model_config = load_train_base_model_config()
    detector_train_config = load_detector_cur_train_config('./tmp/train_1/', 'detector.json')
    adversarial_train(base_model_config, detector_train_config)


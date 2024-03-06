import json
import time

from datasets import load_dataset
# fine tune the radar vicuna

from transformers import AutoModelForSequenceClassification, TrainingArguments, Trainer, AutoTokenizer
from sklearn.metrics import accuracy_score, f1_score

def init_model_and_tokenizer():

    model = AutoModelForSequenceClassification.from_pretrained(
        "TrustSafeAI/RADAR-Vicuna-7B", num_labels=2,
    )

    tokenizer = AutoTokenizer.from_pretrained("TrustSafeAI/RADAR-Vicuna-7B", trust_remote_code=True, max_length=512)

    return model, tokenizer

def compute_metrics(pred):
    labels = pred.label_ids
    preds = pred.predictions.argmax(-1)
    f1 = f1_score(labels, preds, average="weighted")
    acc = accuracy_score(labels, preds)
    return {"accuracy": acc, "f1": f1}

def load_local_dataset(train_file_path):
    start_time = time.time()

    local_dataset = load_dataset('json', data_files={
        'train': train_file_path,
    })
    end_time = time.time()
    print("load dataset successful: " + str(end_time - start_time))
    return local_dataset

def tokenize_data(tokenizer, local_dataset):
    start_time = time.time()

    def tokenize(batch):
        return tokenizer(batch["content"], padding=True, truncation=True, max_length=512)

    tokenized_data = local_dataset.map(tokenize, batched=True, batch_size=None)
    tokenized_data.set_format("torch", columns=["input_ids", "attention_mask", "label"])

    end_time = time.time()
    print("tokenize dataset successful: " + str(end_time - start_time))
    return tokenized_data

def train(model, tokenizer, train_file_path , output_dif):
    training_args = TrainingArguments(
        output_dir=output_dif,
        learning_rate=1e-5,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=16,
        num_train_epochs=5,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        save_total_limit=2,
        load_best_model_at_end=True,
        push_to_hub=False,
    )

    local_dataset = load_local_dataset(train_file_path)
    tokenized_data = tokenize_data(tokenizer, local_dataset)

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_data["train"],
        eval_dataset=tokenized_data["train"],
        compute_metrics=compute_metrics,
    )

    trainer.train()


def prepare_train_data():
    with open('../my_detector/roberta_test/data/hc3_row.train', 'r', encoding='utf-8') as hc3_file, \
        open('../my_detector/roberta_test/data/hc3_mix_multi_prompt.train', 'r', encoding='utf-8') as hc3_adv_file:
        all_json_objs = json.load(hc3_adv_file) + json.load(hc3_file)
        with open('./radar_vicuna_ft.train', 'w', encoding='utf-8') as out_f:
            out_f.write(json.dumps(all_json_objs))


if __name__ == '__main__':

    model, tokenizer = init_model_and_tokenizer()
    prepare_train_data()
    train(model, tokenizer, './radar_vicuna_ft.train', 'radar_vicuna_ft')

    pass
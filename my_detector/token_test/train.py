from datasets import load_dataset
from datasets import DatasetDict
from sklearn.metrics import accuracy_score, f1_score
from transformers import Trainer, TrainingArguments
from transformers import AutoModelForSequenceClassification, AutoTokenizer
import torch
import time

def init_model_and_tokenizer():
    start_time = time.time()
    # tokenizer = AutoTokenizer.from_pretrained("../Erlangshen-DeBERTa-v2-710M-Chinese", model_max_length=256)
    # model = AutoModelForSequenceClassification.from_pretrained("../Erlangshen-DeBERTa-v2-710M-Chinese", num_labels=2)
    tokenizer = AutoTokenizer.from_pretrained('microsoft/deberta-v3-small', model_max_length=512)
    model = AutoModelForSequenceClassification.from_pretrained('microsoft/deberta-v3-small')

    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(DEVICE)
    model = model.to(DEVICE)

    end_time = time.time()
    print("load model successful: " + str(end_time - start_time))
    return model, tokenizer


def load_local_dataset(name):
    start_time = time.time()

    local_dataset = load_dataset('json', data_files={
        'train': './data/' + name + '.train',
        'test': './data/' + name + '.test',
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



def train(name, eval_steps=10, num_train_epochs=10, file_type='.jsonl'):
    print("begin train: " + name)
    start_time = time.time()
    model, tokenizer = init_model_and_tokenizer()
    local_dataset = load_local_dataset(name + file_type)
    tokenized_data = tokenize_data(tokenizer, local_dataset)
    training_args = TrainingArguments(output_dir=name, num_train_epochs=num_train_epochs, load_best_model_at_end=True,
                                      save_total_limit=2, eval_steps=eval_steps, evaluation_strategy='steps',
                                      save_steps=eval_steps)
    trainer = Trainer(model=model, args=training_args, compute_metrics=compute_metrics,
                      train_dataset=tokenized_data["train"],
                      eval_dataset=tokenized_data["train"]
                      )
    end_time = time.time()
    print("train successful: " + name + " : " + str(end_time - start_time))
    trainer.train()


if __name__ == '__main__':
    train('medicine', 20, 12)
    train('finance', 40, 20)
    train('wiki_csai', 20, 8)
    train('hc3_all', 40, 30)

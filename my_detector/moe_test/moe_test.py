import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from torch import nn
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from torch.optim import Adam
from transformers import AutoTokenizer, AutoModel

def get_test_dataloader_and_labels(tokenizer, test_data_path="../Deberta_test/data/hc3_all.jsonl.test", batch_size=16, max_nums=None):
    test_df = pd.read_json(test_data_path)
    test_dataset = MyTestDataset(test_df, tokenizer, max_nums=max_nums)
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size)
    return test_dataloader, test_dataset.labels


class MyTestDataset(Dataset):
    def __init__(self, dataframe, tokenizer, max_nums=None):
        texts = [x['content'] for i, x in dataframe.iterrows()][0: max_nums]

        self.labels = [x['label'] for i, x in dataframe.iterrows()][0: max_nums]

        self.texts = [tokenizer(text, padding='max_length',
                                max_length=512,
                                truncation=True,
                                return_tensors="pt")
                      for text in texts]

        print("end tokenize datas")

    def __getitem__(self, idx):
        text = self.texts[idx]
        label = self.labels[idx]

        return text, label

    def __len__(self):
        return min(len(self.texts), len(self.labels))


class MyClassifier(nn.Module):
    def __init__(self, base_model):
        super(MyClassifier, self).__init__()

        self.bert = base_model
        self.fc1 = nn.Linear(768, 32)
        self.fc2 = nn.Linear(32, 1)

        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

    def forward(self, input_ids, attention_mask):
        bert_out = self.bert(input_ids=input_ids,
                             attention_mask=attention_mask)[0][:, 0]
        x = self.fc1(bert_out)
        x = self.relu(x)

        x = self.fc2(x)
        x = self.sigmoid(x)

        return x


def get_attention_mask_ids_prediction(model, attention_mask, input_ids, bar=0.5):
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    model = model.to(device)
    results_predictions = []
    with torch.no_grad():
        model.eval()
        output = model(input_ids, attention_mask)
        output = (output > bar).int()
        results_predictions.append(output)

    return torch.cat(results_predictions).cpu().detach().numpy()[0]


def get_text_predictions(model, loader, bar=0.5):
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")

    model = model.to(device)

    results_predictions = []
    with torch.no_grad():
        model.eval()
        for data_input, _ in tqdm(loader):
            attention_mask = data_input['attention_mask'].to(device)
            input_ids = data_input['input_ids'].squeeze(1).to(device)

            output = model(input_ids, attention_mask)

            output = (output > bar).int()
            results_predictions.append(output)

    return torch.cat(results_predictions).cpu().detach().numpy()



def gate_prediction(model, model_1, model_2, loader, gate_bar=0.5, classify_bar=0.5):
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")

    model = model.to(device)

    model_1_result = get_text_predictions(model_1, loader, classify_bar)
    model_2_result = get_text_predictions(model_2, loader, classify_bar)

    results_predictions = []
    with torch.no_grad():
        model.eval()
        for data_input, _ in tqdm(loader):
            attention_mask = data_input['attention_mask'].to(device)
            input_ids = data_input['input_ids'].squeeze(1).to(device)

            output = model(input_ids, attention_mask)

            # output = (output > bar).int()
            for i in range(0, len(output)):
                if output[i][0] >= gate_bar:
                    results_predictions.append(model_2_result[i])
                elif output[i][0] < gate_bar:
                    results_predictions.append(model_1_result[i])

    return results_predictions



# 加载两个模型
def init_test_model_and_tokenizer(base_model_name="roberta-base", test_model_path='best_model.pt'):
    tokenizer = AutoTokenizer.from_pretrained(base_model_name)
    if test_model_path == 'roberta-base':
        model = MyClassifier(AutoModel.from_pretrained(base_model_name))
    else:
        model = (torch.load(test_model_path))
        model.eval()

    return model, tokenizer


def get_acc(predictions, test_labels):
    human_total = 0
    ai_total = 0
    human_acc = 0
    ai_acc = 0
    for i in range(0, len(test_labels)):
        if test_labels[i] == 0:
            human_total += 1
            if predictions[i] == 0:
                human_acc += 1
        elif test_labels[i] == 1:
            ai_total += 1
            if predictions[i] == 1:
                ai_acc += 1
    return {
        'ai_acc': ai_acc,
        'ai_total': ai_total,
        'ai_acc_r': ai_acc / ai_total,
        'human_acc': human_acc,
        'human_total': human_total,
        'human_acc_r': human_acc / human_total,
        'total_acc_r': (ai_acc + human_acc) / (ai_total + human_total)
    }



if __name__ == '__main__':
    model_name = 'roberta-base'

    model1_path = '../dpo_test/hc3_adt.pt'
    model1, tokenizer1 = init_test_model_and_tokenizer(base_model_name=model_name, test_model_path=model1_path)
    model2_path = '../dpo_test/dpo_1_2.pt'
    model2, tokenizer2 = init_test_model_and_tokenizer(base_model_name=model_name, test_model_path=model2_path)
    test_path = "best_model.pt"
    test_file = '../roberta_test/data/hc3_mix_multi_prompt.train'
    test_model, test_tokenizer = init_test_model_and_tokenizer(test_model_path=test_path)
    test_dataloader, test_labels = get_test_dataloader_and_labels(test_tokenizer, test_file, 16, 1000)
    text_predictions = gate_prediction(test_model, model1, model2, test_dataloader)
    acc_result = get_acc(text_predictions, test_labels)
    print(acc_result)
    pass
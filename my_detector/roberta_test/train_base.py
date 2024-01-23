import pandas as pd
from sklearn.model_selection import train_test_split

train_df = pd.read_json("../Deberta_test/data/hc3_all.jsonl.train")
# test_df = pd.read_json("../Deberta_test/data/hc3_all.jsonl.train")
test_df = pd.read_csv("../Deberta_test/data/hc3_all.jsonl.test")

train_df, val_df = train_test_split(train_df, test_size=0.2, random_state=42)


from torch.utils.data import Dataset
import numpy as np


class MyDataset(Dataset):
    def __init__(self, dataframe, tokenizer):

        texts = [x['content'] for i, x in dataframe.iterrows()]
        print("begin tokenize datas")
        self.texts = [tokenizer(text, padding='max_length',
                                max_length=256,
                                truncation=True,
                                return_tensors="pt")
                      for text in texts]

        print("end tokenize datas")
        self.labels = [x['label'] for i, x in dataframe.iterrows()]

    def __getitem__(self, idx):
        text = self.texts[idx]
        label = self.labels[idx]

        return text, label

    def __len__(self):
        return min(len(self.texts), len(self.labels))


from torch import nn


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


import torch
from torch.optim import Adam
from tqdm import tqdm


def train(model, train_dataloader, val_dataloader, learning_rate, epochs):
    best_val_loss = float('inf')
    early_stopping_threshold_count = 0

    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")

    criterion = nn.BCELoss()
    optimizer = Adam(model.parameters(), lr=learning_rate)

    model = model.to(device)
    criterion = criterion.to(device)

    for epoch in range(epochs):
        total_acc_train = 0
        total_loss_train = 0

        model.train()

        for train_input, train_label in tqdm(train_dataloader):
            attention_mask = train_input['attention_mask'].to(device)
            input_ids = train_input['input_ids'].squeeze(1).to(device)

            train_label = train_label.to(device)

            output = model(input_ids, attention_mask)

            loss = criterion(output, train_label.float().unsqueeze(1))

            total_loss_train += loss.item()

            acc = ((output >= 0.5).int() == train_label.unsqueeze(1)).sum().item()
            total_acc_train += acc

            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

        with torch.no_grad():
            total_acc_val = 0
            total_loss_val = 0

            model.eval()

            for val_input, val_label in tqdm(val_dataloader):
                attention_mask = val_input['attention_mask'].to(device)
                input_ids = val_input['input_ids'].squeeze(1).to(device)

                val_label = val_label.to(device)

                output = model(input_ids, attention_mask)

                loss = criterion(output, val_label.float().unsqueeze(1))

                total_loss_val += loss.item()

                acc = ((output >= 0.5).int() == val_label.unsqueeze(1)).sum().item()
                total_acc_val += acc

            print(f'Epochs: {epoch + 1} '
                  f'| Train Loss: {total_loss_train / len(train_dataloader): .3f} '
                  f'| Train Accuracy: {total_acc_train / (len(train_dataloader.dataset)): .3f} '
                  f'| Val Loss: {total_loss_val / len(val_dataloader): .3f} '
                  f'| Val Accuracy: {total_acc_val / len(val_dataloader.dataset): .3f}')

            if best_val_loss > total_loss_val:
                best_val_loss = total_loss_val
                torch.save(model, f"best_model.pt")
                print("Saved model")
                early_stopping_threshold_count = 0
            else:
                early_stopping_threshold_count += 1

            if early_stopping_threshold_count >= 1:
                print("Early stopping")
                break


from transformers import AutoTokenizer, AutoModel
from torch.utils.data import DataLoader

torch.manual_seed(0)
np.random.seed(0)

BERT_MODEL = "roberta-base"
tokenizer = AutoTokenizer.from_pretrained(BERT_MODEL)
base_model = AutoModel.from_pretrained(BERT_MODEL)

train_dataloader = DataLoader(MyDataset(train_df, tokenizer), batch_size=16, shuffle=True)
val_dataloader = DataLoader(MyDataset(val_df, tokenizer), batch_size=16)

model = MyClassifier(base_model)

learning_rate = 1e-5
epochs = 5
train(model, train_dataloader, val_dataloader, learning_rate, epochs)


def get_text_predictions(model, loader):
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

            output = (output > 0.5).int()
            results_predictions.append(output)

    return torch.cat(results_predictions).cpu().detach().numpy()


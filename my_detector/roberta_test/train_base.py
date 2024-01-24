import pandas as pd
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset
import numpy as np

from torch import nn
import torch
from torch.optim import Adam
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModel
from torch.utils.data import DataLoader


def load_train_and_val_df(train_data_path = "../Deberta_test/data/hc3_all.jsonl.train", val_size=0.2, random_state=0):
    train_file = pd.read_json(train_data_path)
    train_df, val_df = train_test_split(train_file, test_size=val_size, random_state=random_state)
    return train_df, val_df

def get_train_and_val_dataloader(train_df, val_df, tokenizer, batch_size=16, shuffle=False):
    train_dataloader = DataLoader(MyDataset(train_df, tokenizer), batch_size=batch_size, shuffle=shuffle)
    val_dataloader = DataLoader(MyDataset(val_df, tokenizer), batch_size=batch_size, shuffle=shuffle)

    return train_dataloader, val_dataloader

# train_df = pd.read_json("../Deberta_test/data/hc3_all.jsonl.train")
# test_df = pd.read_json("../Deberta_test/data/hc3_all.jsonl.train")
# test_df = pd.read_csv("../Deberta_test/data/hc3_all.jsonl.test")

# train_df, val_df = train_test_split(train_df, test_size=0.2, random_state=42)


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

        self.domains = []
        self.prompts = []
        for i, x in dataframe.iterrows():
            try:
                if x['domain'] is None:
                    self.domains.append('default')
                else:
                    self.domains.append(x['domain'])
            except Exception as e:
                self.domains.append('default')
            try:
                if x['prompt'] is None:
                    self.prompts.append('default')
                else:
                    self.prompts.append(x['prompt'])
            except Exception as e:
                self.prompts.append('default')
        # if dataframe.iterrows()[0]['domain'] is None:
        #     self.domains = ['default' for i, x in dataframe.iterrows() ]
        # else:
        #     self.domains = [x['domain'] for i, x in dataframe.iterrows()]
        #
        # if dataframe.iterrows()[0]['prompt'] is None:
        #     self.prompts = ['default' for i, x in dataframe.iterrows() ]
        # else:
        #     self.prompts = [x['prompt'] for i, x in dataframe.iterrows()]

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



def train(model, train_dataloader, val_dataloader, learning_rate, epochs, save_name = "best_model.pt"):
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
                torch.save(model, save_name)
                print("Saved model")
                early_stopping_threshold_count = 0
            else:
                early_stopping_threshold_count += 1

            if early_stopping_threshold_count >= 1:
                print("Early stopping")
                break


def init_model_and_tokenizer(base_model_name='roberta-base'):
    torch.manual_seed(0)
    np.random.seed(0)

    # BERT_MODEL = "roberta-base"
    tokenizer = AutoTokenizer.from_pretrained(base_model_name)
    base_model = AutoModel.from_pretrained(base_model_name)
    model = MyClassifier(base_model)

    return model, tokenizer

if __name__ == '__main__':

    model, tokenizer = init_model_and_tokenizer()
    train_df, val_df = load_train_and_val_df('./data/hc3_mix_multi_prompt.train')
    train_dataloader, val_dataloader = get_train_and_val_dataloader(train_df, val_df, tokenizer)

    learning_rate = 1e-5
    epochs = 5
    train(model, train_dataloader, val_dataloader, learning_rate, epochs, save_name='hc3_mix_multi_prompt.pt')





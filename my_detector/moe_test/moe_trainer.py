import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from torch import nn
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from torch.optim import Adam
from transformers import AutoTokenizer, AutoModel

# 训练集
class MyTrainDataset(Dataset):
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

def load_train_and_val_df(train_data_path="../Deberta_test/data/hc3_all.jsonl.train", val_size=0.2, random_state=0):
    train_file = pd.read_json(train_data_path)
    train_df, val_df = train_test_split(train_file, test_size=val_size, random_state=random_state)
    return train_df, val_df

def get_train_and_val_dataloader(train_df, val_df, tokenizer, batch_size=16, shuffle=False):
    train_dataloader = DataLoader(MyTrainDataset(train_df, tokenizer), batch_size=batch_size, shuffle=shuffle)
    val_dataloader = DataLoader(MyTrainDataset(val_df, tokenizer), batch_size=batch_size, shuffle=shuffle)

    return train_dataloader, val_dataloader

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

# 加载两个模型
def init_test_model_and_tokenizer(base_model_name="roberta-base", test_model_path='best_model.pt'):
    tokenizer = AutoTokenizer.from_pretrained(base_model_name)
    if test_model_path == 'roberta-base':
        model = MyClassifier(AutoModel.from_pretrained(base_model_name))
    else:
        model = (torch.load(test_model_path))
        model.eval()

    return model, tokenizer

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


def train(base_model,
          model_1,
          model_2,
          train_dataloader, val_dataloader,
          learning_rate=1e-5, epochs=5,
          save_name="best_model.pt"):
    best_val_loss = float('inf')
    early_stopping_threshold_count = 0

    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")

    criterion = nn.BCELoss()
    optimizer = Adam(base_model.parameters(), lr=learning_rate)

    base_model = base_model.to(device)
    criterion = criterion.to(device)

    for epoch in range(epochs):
        total_acc_train = 0
        total_loss_train = 0

        base_model.train()

        for train_input, train_label in tqdm(train_dataloader):
            attention_mask = train_input['attention_mask'].to(device)
            input_ids = train_input['input_ids'].squeeze(1).to(device)

            train_label = train_label.to(device)

            output = base_model(input_ids, attention_mask)

            output_labels = []
            for output1 in output:
                if output1 == 0:
                    output_labels.append(get_attention_mask_ids_prediction(model_1, attention_mask, input_ids))
                elif output1 == 1:
                    output_labels.append(get_attention_mask_ids_prediction(model_2, attention_mask, input_ids))

            loss = criterion(output_labels, train_label.float().unsqueeze(1))

            total_loss_train += loss.item()

            acc = (output_labels == train_label.unsqueeze(1)).sum().item()
            total_acc_train += acc

            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

        total_loss_adversary_train = 0

        with torch.no_grad():
            total_acc_val = 0
            total_loss_val = 0

            base_model.eval()

            for val_input, val_label in tqdm(val_dataloader):
                attention_mask = val_input['attention_mask'].to(device)
                input_ids = val_input['input_ids'].squeeze(1).to(device)

                val_label = val_label.to(device)

                output = base_model(input_ids, attention_mask)

                output_labels = []
                for output1 in output:
                    if output1 == 0:
                        output_labels.append(get_attention_mask_ids_prediction(model_1, attention_mask, input_ids))
                    elif output1 == 1:
                        output_labels.append(get_attention_mask_ids_prediction(model_2, attention_mask, input_ids))

                loss = criterion(output_labels, val_label.float().unsqueeze(1))

                total_loss_val += loss.item()

                acc = (output_labels == val_label.unsqueeze(1)).sum().item()
                total_acc_val += acc

            print(f'Epochs: {epoch + 1} '
                  f'| Train Loss: {total_loss_train / len(train_dataloader): .3f} '
                  f'| Train Accuracy: {total_acc_train / (len(train_dataloader.dataset)): .3f} '
                  f'| Val Loss: {total_loss_val / len(val_dataloader): .3f} '
                  f'| Val Accuracy: {total_acc_val / len(val_dataloader.dataset): .3f}')

            if best_val_loss > total_loss_val + total_loss_adversary_train:
                best_val_loss = total_loss_val + total_loss_adversary_train
                torch.save(base_model, save_name)
                print("Saved model")
                early_stopping_threshold_count = 0
            else:
                early_stopping_threshold_count += 1


if __name__ == '__main__':
    model_name = 'roberta-base'

    model1_path = ''
    model1, tokenizer1 = init_test_model_and_tokenizer(base_model_name=model_name, test_model_path=model1_path)
    model2_path = ''
    model2, tokenizer2 = init_test_model_and_tokenizer(base_model_name=model_name, test_model_path=model2_path)

    base_model, base_tokenizer = init_test_model_and_tokenizer(model_name, model_name)

    train_file = './data/hc3_mix_multi_prompt.train'
    train_df, val_df = load_train_and_val_df(train_file)
    train_dataloader, val_dataloader = get_train_and_val_dataloader(train_df, val_df, base_tokenizer, 16)
    train(
        base_model,
        model1,
        model2,
        train_dataloader,
        val_dataloader,
    )

import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from torch import nn
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from torch.optim import Adam
from transformers import AutoTokenizer, AutoModel

device = 'cuda'

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


def load_train_and_val_df(train_data_paths=["../Deberta_test/data/hc3_all.jsonl.train"], val_size=0.2, random_state=0):
    # 存储多个训练集和验证集的列表
    train_dfs = []
    val_dfs = []
    print(train_data_paths)
    # 假设有多个文件，依次读取并拆分
    for file in train_data_paths:
        print(file)
        data = pd.read_json(file)
        train_df, val_df = train_test_split(data, test_size=val_size, random_state=random_state)
        train_dfs.append(train_df)
        val_dfs.append(val_df)

    # 合并多个训练集和验证集
    final_train_df = pd.concat(train_dfs, axis=0)
    final_val_df = pd.concat(val_dfs, axis=0)

    # final_train_df.drop_duplicates(subset=['content'], keep='first', inplace=True)
    # final_train_df.reset_index(drop=True, inplace=True)
    # final_val_df.drop_duplicates(subset=['content'], keep='first', inplace=True)
    # final_val_df.reset_index(drop=True, inplace=True)

    return final_train_df, final_val_df
    # train_file = pd.read_json(train_data_path)
    # train_df, val_df = train_test_split(train_file, test_size=val_size, random_state=random_state)
    # return train_df, val_df


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


def get_gate_attention_mask_ids_prediction(model, attention_mask, input_ids, bar=0.5):
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    model = model.to(device)
    results_predictions = []
    with torch.no_grad():
        model.eval()
        output = model(input_ids, attention_mask)
        output = (output > bar).int()
        results_predictions.append(output)

    result = torch.cat(results_predictions).cpu().detach().numpy()
    return result


def get_model_attention_mask_ids_prediction(model, attention_mask, input_ids, bar=0.5):
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    model = model.to(device)
    results_predictions = []
    with torch.no_grad():
        model.eval()
        output = model(input_ids, attention_mask)
        output = (output > bar).int()
        results_predictions.append(output)
    result = torch.cat(results_predictions).cpu().detach().numpy()
    # print(result)
    return result


def gate_prediction(gate_outputs, model_1, model_2, attention_mask, input_ids, bar=0.5):
    output_labels = []
    model_1_results = get_model_attention_mask_ids_prediction(model_1, attention_mask, input_ids)
    model_2_results = get_model_attention_mask_ids_prediction(model_2, attention_mask, input_ids)

    for i in range(0, len(gate_outputs)):
        if gate_outputs[i] < bar:
            output_labels.append(model_1_results[i])
        elif gate_outputs[i] >= bar:
            output_labels.append(model_2_results[i])
    # for output1 in gate_outputs:
    #     # print(output1)
    #     if output1[0] < bar:
    #         output_labels.append(get_gate_attention_mask_ids_prediction(model_1, attention_mask, input_ids))
    #     elif output1[0] >= bar:
    #         output_labels.append(get_gate_attention_mask_ids_prediction(model_2, attention_mask, input_ids))
    result = torch.from_numpy(np.array(output_labels)).to(device)
    # print(result)
    return result

def convert_train_to_actual_label(train_label,  model_1, model_2, attention_mask, input_ids):
    model_1_results = get_model_attention_mask_ids_prediction(model_1, attention_mask, input_ids)
    model_2_results = get_model_attention_mask_ids_prediction(model_2, attention_mask, input_ids)
    results = []

    for i in range(0, len(train_label)):
        cur_label = train_label[i]
        if model_1_results[i][0] != cur_label and model_2_results[i][0] != cur_label:
            results.append(0.5)
        elif model_1_results[i][0] == cur_label and model_2_results[i][0] == cur_label:
            results.append(0.5)
        elif model_1_results[i][0] == cur_label:
            results.append(0.0)
        elif model_2_results[i][0] == cur_label:
            results.append(1.0)

        # if model_1_results[i][0] != cur_label and model_2_results[i][0] != cur_label:
        #     results.append(1.0)
        # elif model_1_results[i][0] == cur_label and model_2_results[i][0] == cur_label:
        #     results.append(0.0)
        # elif model_1_results[i][0] == cur_label:
        #     results.append(0.0)
        # elif model_2_results[i][0] == cur_label:
        #     results.append(1.0)

        # if model_1_results[i][0] == 1 and cur_label == 1:
        #     results.append(0.0)
        # elif model_1_results[i][0] == 0 and cur_label == 0:
        #     results.append(0.0)
        # else:
        #     if model_2_results[i][0] == 1 and cur_label == 1:
        #         results.append(1.0)
        #     else:
        #         results.append(0.0)

    return torch.from_numpy(np.array(results)).to(device)


def train(base_model,
          model_1,
          model_2,
          train_dataloader, val_dataloader,
          learning_rate=1e-5, epochs=5,
          save_name="best_model.pt",
          bar=0.5):
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

            # print(train_label)

            actual_label = convert_train_to_actual_label(train_label, model_1, model_2, attention_mask, input_ids)
            # print(output)
            # print(actual_label)

            # gate_outputs = gate_prediction(output, model_1, model_2, attention_mask, input_ids)
            #
            # for i in range(0, len(gate_outputs)):
            #     output[i] = gate_outputs[i]

            # print(gate_outputs)
            # print(train_label.float().unsqueeze(1))

            loss = criterion(output, actual_label.float().unsqueeze(1))

            total_loss_train += loss.item()

            # acc = (output == actual_label.unsqueeze(1)).sum().item()
            acc = 0
            for i in range(0, len(output)):
                if (output[i][0] > bar) and (actual_label[i] > bar):
                    acc+=1
                elif (output[i][0] <= bar) and (actual_label[i] <= bar):
                    acc+=1

            total_acc_train += acc

            loss.backward()
            optimizer.step()
            optimizer.zero_grad()


        with torch.no_grad():
            total_acc_val = 0
            total_loss_val = 0

            base_model.eval()

            for val_input, val_label in tqdm(val_dataloader):
                attention_mask = val_input['attention_mask'].to(device)
                input_ids = val_input['input_ids'].squeeze(1).to(device)

                val_label = val_label.to(device)

                output = base_model(input_ids, attention_mask)

                actual_label = convert_train_to_actual_label(val_label, model_1, model_2, attention_mask, input_ids)
                # gate_outputs = gate_prediction(output, model_1, model_2, attention_mask, input_ids)

                loss = criterion(output, actual_label.float().unsqueeze(1))

                total_loss_val += loss.item()

                # acc = (output == actual_label.unsqueeze(1)).sum().item()
                acc = 0
                for i in range(0, len(output)):
                    if (output[i][0] >= bar) and (actual_label[i] >= bar):
                        acc += 1
                    elif (output[i][0] < bar) and (actual_label[i] < bar):
                        acc += 1

                total_acc_val += acc

            print(f'Epochs: {epoch + 1} '
                  f'| Train Loss: {total_loss_train / len(train_dataloader): .3f} '
                  f'| Train Accuracy: {total_acc_train / (len(train_dataloader.dataset)): .3f} '
                  f'| Val Loss: {total_loss_val / len(val_dataloader): .3f} '
                  f'| Val Accuracy: {total_acc_val / len(val_dataloader.dataset): .3f}')

            if best_val_loss > total_loss_val:
                best_val_loss = total_loss_val
                torch.save(base_model, save_name)
                print("Saved model")
                early_stopping_threshold_count = 0
            else:
                early_stopping_threshold_count += 1


if __name__ == '__main__':
    model_name = 'roberta-base'

    model1_path = '../roberta_test/moe_adt3.pt'
    model1, tokenizer1 = init_test_model_and_tokenizer(base_model_name=model_name, test_model_path=model1_path)
    model2_path = '../dpo_test/moe_3.pt'
    model2, tokenizer2 = init_test_model_and_tokenizer(base_model_name=model_name, test_model_path=model2_path)

    base_model, base_tokenizer = init_test_model_and_tokenizer(model_name, model_name)

    # train_file = '../roberta_test/data/hc3_mix_multi_prompt.train'
    train_files = [
        './data/nature/qwen/7.jsonl.qwen.rewrite.jsonl.train',
        # './data/nature/qwen/8.jsonl.qwen.rewrite.jsonl.train',
        # './data/nature/qwen/9.jsonl.qwen.rewrite.jsonl.train',
        # './data/nature/qwen/10.jsonl.qwen.rewrite.jsonl.train',
        './data/adversary/qwen/7.jsonl.qwen.rewrite.jsonl.qwen.paraphase.jsonl.train',
        # './data/adversary/qwen/8.jsonl.qwen.rewrite.jsonl.qwen.paraphase.jsonl.train',
        # './data/adversary/qwen/9.jsonl.qwen.rewrite.jsonl.qwen.paraphase.jsonl.train',
        # './data/adversary/qwen/10.jsonl.qwen.rewrite.jsonl.qwen.paraphase.jsonl.train',
    ]
    train_df, val_df = load_train_and_val_df(train_data_paths=train_files, random_state=1)
    train_dataloader, val_dataloader = get_train_and_val_dataloader(train_df, val_df, base_tokenizer, 8, True)
    train(
        base_model,
        model1,
        model2,
        train_dataloader,
        val_dataloader,
        save_name='moe_gate_8.pt'
    )

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

from natural_adversal import AdversaryGenerator, MyAdversaryDataset


def load_train_and_val_df(train_data_path="../Deberta_test/data/hc3_all.jsonl.train", val_size=0.2, random_state=0):
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
    def __init__(self, dataframe, tokenizer, max_nums=None):

        texts = [x['content'] for i, x in dataframe.iterrows()]

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

        if max_nums is None:
            print("begin tokenize datas")
            self.texts = [tokenizer(text, padding='max_length',
                                    max_length=256,
                                    truncation=True,
                                    return_tensors="pt")
                          for text in texts]

            print("end tokenize datas")
        else:
            tmp_objs = []
            for i in range(0, len(self.labels)):
                tmp_objs.append(
                    {
                        'label': self.labels[i],
                        'text': texts[i],
                        'domain': self.domains[i],
                        'prompt': self.prompts[i],
                    }
                )
            print("tmp objs len:" + str(len(tmp_objs)))
            tmp_objs_map = {}
            for tmp_obj in tmp_objs:
                if tmp_objs_map.get(tmp_obj['prompt']) is None:
                    tmp_objs_map[tmp_obj['prompt']] = [tmp_obj]
                else:
                    tmp_objs_map[tmp_obj['prompt']].append(tmp_obj)

            result_objs = []
            for key in tmp_objs_map:
                tmp_human_objs = [x for x in tmp_objs_map[key] if x['label'] == 0][0: max_nums]
                tmp_ai_objs = [x for x in tmp_objs_map[key] if x['label'] == 1][0: max_nums]
                result_objs += tmp_ai_objs+tmp_human_objs
                print(f"{key}:{len(tmp_ai_objs)}:{len(tmp_human_objs)}")

            self.labels = [x['label'] for x in result_objs]
            tmp_texts = [x['text'] for x in result_objs]
            self.domains = [x['domain'] for x in result_objs]
            self.prompts = [x['prompt'] for x in result_objs]

            print("begin tokenize datas")
            self.texts = [tokenizer(text, padding='max_length',
                                    max_length=256,
                                    truncation=True,
                                    return_tensors="pt")
                          for text in tmp_texts]

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


def train(model, tokenizer, train_dataloader, val_dataloader, learning_rate, epochs, batch_size=16,
          save_name="best_model.pt", adversary_generator=None, adv_loss_alpha = 1.0):
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

        total_loss_adversary_train = 0
        if adversary_generator is not None:
            total_loss_adversary = 0
            print("begin adversary module")
            adversary_loss_map = adversary_generator.val_cur_train_model_by_local_datas(cur_train_model=model,
                                                                                        cur_train_tokenizer=tokenizer)
            print(adversary_loss_map)
            for k in adversary_loss_map:
                total_loss_adversary += adversary_loss_map[k]
            # total_loss_adversary /= len(adversary_loss_map)

            adversary_data_num_map = adversary_generator.calculate_adversary_data_num_by_loss(adversary_loss_map)
            print(adversary_data_num_map)
            print("begin generate adversary train datas....")
            adversary_train_data = (
                adversary_generator.generate_adversary_train_data_by_val_result(adversary_data_num_map))

            adversary_train_dataloader = DataLoader(MyAdversaryDataset(adversary_train_data, tokenizer),
                                                    batch_size=batch_size)
            print("training generate adversary train datas")


            for adversary_train_input, adversary_train_label in tqdm(adversary_train_dataloader):
                attention_mask = adversary_train_input['attention_mask'].to(device)
                input_ids = adversary_train_input['input_ids'].squeeze(1).to(device)
                adversary_train_label = adversary_train_label.to(device)
                output = model(input_ids, attention_mask)
                loss = criterion(output, adversary_train_label.float().unsqueeze(1)) * adv_loss_alpha
                total_loss_adversary_train += loss.item()
                acc = ((output >= 0.5).int() == adversary_train_label.unsqueeze(1)).sum().item()
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

            if best_val_loss > total_loss_val + total_loss_adversary_train * adv_loss_alpha:
                best_val_loss = total_loss_val + total_loss_adversary_train * adv_loss_alpha
                torch.save(model, save_name)
                print("Saved model")
                early_stopping_threshold_count = 0
            else:
                early_stopping_threshold_count += 1

            # if early_stopping_threshold_count >= 1:
            #     print("Early stopping")
            #     break


def init_model_and_tokenizer(base_model_name='roberta-base'):
    torch.manual_seed(0)
    np.random.seed(0)

    # BERT_MODEL = "roberta-base"
    tokenizer = AutoTokenizer.from_pretrained(base_model_name)
    base_model = AutoModel.from_pretrained(base_model_name)
    model = MyClassifier(base_model)

    return model, tokenizer


def init_adversary_generator(train_df, random_generate=False,random_select_key=None):
    local_val_file_map = {
        'rewrite': './data/open_qa.rewrite.mix.jsonl.train',
        'continue': './data/open_qa.continue.mix.jsonl.train',
        'academic': './data/open_qa.academic.mix.jsonl.train',
        'difficult': './data/open_qa.difficult.mix.jsonl.train',
        'easy': './data/open_qa.easy.mix.jsonl.train',
        'qa': './data/open_qa.mix.jsonl.train',
    }
    prompts_map = {
        'rewrite': 'Please rewrite the following content, {without any useless content}:',
        'continue': 'Please continue to write the following content, {without any useless content}:',
        'easy': 'Please change the following content to make it easier to understand, {without any useless content}:',
        'academic': 'Please change the following content to be more academic and professional, {without any useless content}:',
        'difficult': 'Please change the following content to make it more difficult to understand, {without any useless content}:',
        'qa': 'The following is a response to a question, please re-answer the question based on this response, {without any useless content}:'
    }


    adversary_generator = AdversaryGenerator(
        local_val_files_map=local_val_file_map,
        prompts_map=prompts_map,
        train_df=train_df,
        random_generate=random_generate,
        random_select_key=random_select_key
    )

    return adversary_generator


def begin_train(train_data_file,
                adversary_generator,
                model_save_name,
                base_model_name='roberta-base',
                batch_size=16,
                learning_rate=1e-5,
                epochs=5,
                adv_loss_alpha=1.0
                ):
    model, tokenizer = init_model_and_tokenizer(base_model_name)
    train_df, val_df = load_train_and_val_df(train_data_file)
    train_dataloader, val_dataloader = get_train_and_val_dataloader(train_df, val_df, tokenizer, batch_size)
    train(model, tokenizer, train_dataloader, val_dataloader, learning_rate, epochs, batch_size,
          save_name=model_save_name, adversary_generator=adversary_generator, adv_loss_alpha=adv_loss_alpha)

def begin_train_hc3_row():
    train_file = './data/hc3_row.train'
    begin_train(
        train_file,
        None,
        'hc3_row.pt'
    )

def begin_train_hc3_adt():
    train_file = './data/hc3_row.train'
    train_df, val_df = load_train_and_val_df(train_file)
    adversary_generator = init_adversary_generator(train_df, random_generate=False, random_select_key=None)
    begin_train(
        train_file,
        adversary_generator,
        'hc3_adt.pt'
    )

def begin_train_hc3_random_adt():
    train_file = './data/hc3_row.train'
    train_df, val_df = load_train_and_val_df(train_file)
    adversary_generator = init_adversary_generator(train_df, random_generate=True, random_select_key=None)
    begin_train(
        train_file,
        adversary_generator,
        'hc3_random_adt.pt'
    )

def begin_train_hc3_random_select_adt():
    train_file = './data/hc3_row.train'
    train_df, val_df = load_train_and_val_df(train_file)
    adversary_generator = init_adversary_generator(train_df, random_generate=True, random_select_key='qa')
    begin_train(
        train_file,
        adversary_generator,
        'hc3_random_select_adt.pt'
    )

def begin_train_hc3_row_adt():
    train_file = './data/hc3_mix_multi_prompt.train'
    train_df, val_df = load_train_and_val_df(train_file)
    adversary_generator = None
    begin_train(
        train_file,
        adversary_generator,
        'hc3_row_adt.pt'
    )

def begin_train_hc3_adt_alpha(alpha):
    train_file = './data/hc3_row.train'
    train_df, val_df = load_train_and_val_df(train_file)
    adversary_generator = init_adversary_generator(train_df, random_generate=False, random_select_key=None)
    begin_train(
        train_file,
        adversary_generator,
        'hc3_adt.alpha.' + str(alpha) + '.pt',
        adv_loss_alpha=alpha
    )

if __name__ == '__main__':
    # batch_size = 16
    # train_file = './data/hc3_row.train'
    # model, tokenizer = init_model_and_tokenizer()
    # train_df, val_df = load_train_and_val_df(train_file)
    # train_dataloader, val_dataloader = get_train_and_val_dataloader(train_df, val_df, tokenizer, batch_size)
    # adversary_generator = init_adversary_generator(train_df, random_generate=False, random_select_key=None)
    # learning_rate = 1e-5
    # epochs = 5
    # begin_train(
    #     train_file,
    #     adversary_generator,
    #     'hc3_mix_at.pt'
    # )
    # train(model, tokenizer, train_dataloader, val_dataloader, learning_rate, epochs, batch_size,
    #       save_name='hc3_mix_ad.pt', adversary_generator=adversary_generator)
    # begin_train_hc3_row()
    # begin_train_hc3_adt()
    # begin_train_hc3_random_adt()
    # begin_train_hc3_random_select_adt()
    # begin_train_hc3_row_adt()
    begin_train_hc3_adt_alpha(0.0)
    begin_train_hc3_adt_alpha(10.0)
    pass

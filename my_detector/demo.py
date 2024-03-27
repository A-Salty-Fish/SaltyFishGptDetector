
import json
import os
import time

import pandas as pd
import torch
from torch import nn
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
)

device = "cuda"


def load_test_model(model_name="mistralai/Mistral-7B-Instruct-v0.2", perf_path='./tmp.pt/checkpoint-1600'):
    all_begin_time = time.time()

    model = AutoModelForCausalLM.from_pretrained(perf_path,
                                                 # quantization_config=bnb_config,
                                                 low_cpu_mem_usage=True,
                                                 torch_dtype=torch.float16,
                                                 load_in_4bit=True,
                                                 trust_remote_code=True)
    print("load model success: " + str(time.time() - all_begin_time))

    begin_time = time.time()
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    print("load tokenizer success: " + str(time.time() - begin_time))

    print("load all success: " + str(time.time() - all_begin_time))
    return model, tokenizer


def chat(model, tokenizer, context):
    # start_time = time.time()
    messages = [
        {"role": "user", "content": context}
        # {"role": "assistant", "content": "Well, I'm quite partial to a good squeeze of fresh lemon juice. It adds just the right amount of zesty flavour to whatever I'm cooking up in the kitchen!"},
        # {"role": "user", "content": "Do you have mayonnaise recipes?"}
    ]

    encodeds = tokenizer.apply_chat_template(messages, return_tensors="pt")

    model_inputs = encodeds.to(device)

    generated_ids = model.generate(model_inputs, max_new_tokens=1024, do_sample=True,
                                   pad_token_id=tokenizer.eos_token_id)
    decoded = tokenizer.batch_decode(generated_ids)
    # end_time = time.time()
    # print("generate response successful: " + str(end_time - start_time))
    # print(decoded[0])
    # print('-----------------------')
    # print(decoded)
    return decoded[0].split('[/INST]')[1].replace('</s>', '')
    # return decoded[0]


from torch.utils.data import Dataset, DataLoader


class MyTestDataset(Dataset):
    def __init__(self, texts, tokenizer, max_nums=None):

        if max_nums is None:
            self.texts = texts
        else:
            self.texts = texts[0: max_nums]
        self.texts = [tokenizer(text, padding='max_length',
                            max_length=512,
                            truncation=True,
                            return_tensors="pt")
                  for text in texts]
        print("end tokenize datas")

    def __getitem__(self, idx):
        text = self.texts[idx]
        return text, 0

    def __len__(self):
        return len(self.texts)


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
            print(output)
            # output = (output > bar).int()
            results_predictions.append(output.item())

    # return torch.cat(results_predictions).cpu().detach().numpy()
    return results_predictions


def get__classifier_test_dataloader_and_labels(tokenizer, test_data_path="../Deberta_test/data/hc3_all.jsonl.test",
                                               batch_size=16, max_nums=2000):
    test_df = pd.read_json(test_data_path)
    test_dataset = MyTestDataset(test_df, tokenizer, max_nums=max_nums)
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size)
    # print(test_dataset.labels)
    return test_dataloader, test_dataset.labels


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


def output_acc(file_name, acc_json):
    total_acc_r = acc_json['total_acc_r']
    human_total = acc_json['human_total']
    human_acc = acc_json['human_acc']
    human_acc_r = acc_json['human_acc_r']
    ai_total = acc_json['ai_total']
    ai_acc = acc_json['ai_acc']
    ai_acc_r = acc_json['ai_acc_r']
    acc_str = f'{file_name}\t{total_acc_r}\t{human_total}\t{human_acc}\t{human_acc_r}\t{ai_total}\t{ai_acc}\t{ai_acc_r}'
    return acc_str


def test_classifier_multi_files(classifier_name, classifier_path, test_files, max_nums=1000):
    classifier_tokenizer = AutoTokenizer.from_pretrained(classifier_name)
    classifier_model = (torch.load(classifier_path))
    classifier_model.eval()
    acc_str_result = ''
    for test_file in test_files:
        print(test_file)
        test_dataloader, test_labels = get__classifier_test_dataloader_and_labels(classifier_tokenizer, test_file,
                                                                                  max_nums=max_nums)
        text_predictions = get_text_predictions(classifier_model, test_dataloader)
        acc_result = get_acc(text_predictions, test_labels)
        acc_str = output_acc(test_file.split('/')[-1], acc_result)
        acc_str_result += acc_str + '\n'
    del classifier_model, classifier_tokenizer

    return acc_str_result


def get_text_result(model, tokenzier, texts):
    test_dataset = MyTestDataset(texts, tokenzier)
    test_dataloader = DataLoader(test_dataset, batch_size=len(texts))
    return get_text_predictions(model, test_dataloader)

import gradio as gr

if __name__ == '__main__':

    classifier_tokenizer = AutoTokenizer.from_pretrained('roberta-base')
    classifier_model = (torch.load('./dpo_test/dpo_1_2.pt'))

    def get_text_pred(text):
        res = get_text_result(classifier_model, classifier_tokenizer, [text])[0]
        percentage_str = "{:.3%}".format(res)
        return percentage_str

    iface = gr.Interface(
        fn=get_text_pred,
        inputs=gr.TextArea(lines=8, label="需要检测的文本"),
        outputs=gr.Textbox(label="检测结果为AI生成的百分比"),
        title="ARaDaT: Adversarial robust AI-generated text detection based on DPO adversarial training",
        description="demo:v0"
    )

    iface.launch(share=True)

    pass

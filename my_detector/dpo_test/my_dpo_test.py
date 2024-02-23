import json
import os
import time

import pandas as pd
import torch
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


def generate_datas(model, tokenizer, output_dir, file_pre):
    import os
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f"Directory '{output_dir}' created successfully.")
    else:
        print(f"Directory '{output_dir}' already exists.")

    base_dir = '../../my_detector/roberta_test/data/'
    files = [
        'hc3_row.train',
        'hc3_row.test',
        'cheat_generation.test',
        'cheat_polish.test',
        'ghostbuster_claude.test',
        'hc3_plus_qa_row.test',
        # 'open_qa.academic.test',
        # 'open_qa.continue.test',
        # 'open_qa.difficult.test',
        # 'open_qa.easy.test',
        # 'open_qa.test',
        # 'open_qa.rewrite.test',
        'reddit_chatGPT.test',
        'reddit_cohere.test',
        'reddit_davinci.test',
        'reddit_dolly.test',
        'reddit_flant5.test',
        'wikipedia_chatgpt.test',
        'wikipedia_cohere.test',
        'wikipedia_davinci.test',
        'wikipedia_dolly.test',
    ]
    for file in files:
        generate_paraphase_data(base_dir + file, file, model, tokenizer, output_dir, file_pre)


def generate_paraphase_data(file_path, file_name, model, tokenizer, output_dir= './qwen/', file_pre='', max_num=1000):
    prompt_template = "Please rewrite the following AI-generated text to make it more like human text, {without any useless content}: "
    print(file_path)
    try:
        with open(output_dir + file_name + file_pre + '.jsonl', 'r', encoding='utf-8') as out_f:
            existed_lines = 0
            for line in out_f:
                existed_lines += 1
            print(existed_lines)
    except Exception as e:
        existed_lines = 0
    with open(file_path, 'r',encoding='utf-8') as in_f:
        cur = 0
        with open(output_dir + file_name + file_pre + '.jsonl', 'a', encoding='utf-8') as out_f:

            file_objs = json.load(in_f)
            ai_objs = [x for x in file_objs if x['label'] == 1][0: max_num]
            results = []

            for ai_obj in tqdm(ai_objs):
                cur+=1
                if cur < existed_lines:
                    continue
                ai_content = ai_obj['content']
                ai_rewrite = chat(model, tokenizer, prompt_template + ai_obj['content'])
                new_ai_obj = {
                    'label': 1,
                    'prompt': prompt_template + ai_obj['content'],
                    'ai': ai_content,
                    'ai_rewrite': ai_rewrite
                }
                out_f.write(json.dumps(new_ai_obj) + '\n')

from torch.utils.data import Dataset, DataLoader
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

def get__classifier_test_dataloader_and_labels(tokenizer, test_data_path="../Deberta_test/data/hc3_all.jsonl.test", batch_size=16, max_nums=None):
    test_df = pd.read_json(test_data_path)
    test_dataset = MyTestDataset(test_df, tokenizer, max_nums=max_nums)
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size)
    return test_dataloader, test_dataset.labels

def get_acc(predictions, test_labels):
    human_total = 0
    ai_total = 0
    human_acc = 0
    ai_acc = 0
    for i in range(0, len(test_labels)):
        if test_labels[i] == 0:
            human_total+=1
            if predictions[i] == 0:
                human_acc+=1
        elif test_labels[i] == 1:
            ai_total+=1
            if predictions[i] == 1:
                ai_acc+=1
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
    print(f'{file_name}\t{total_acc_r}\t{human_total}\t{human_acc}\t{human_acc_r}\t{ai_total}\t{ai_acc}\t{ai_acc_r}')


def test_classifier_multi_files(classifier_name, classifier_path, test_files, max_nums=1000):
    classifier_tokenizer = AutoTokenizer.from_pretrained(classifier_name)
    classifier_model = (torch.load(classifier_path))
    classifier_model.eval()
    for test_file in test_files:
        print(test_file)
        test_dataloader, test_labels = get__classifier_test_dataloader_and_labels(classifier_tokenizer, test_file, max_nums=max_nums)
        text_predictions = get_text_predictions(classifier_model, test_dataloader)
        acc_result = get_acc(text_predictions, test_labels)
        output_acc(test_file.split('/')[-1], acc_result)

def test_classifier_base_dir(classifier_name, classifier_path, base_dir='./qwen/', max_nums=1000):
    test_files = [
        base_dir + file for file in os.listdir(base_dir) if file.endswith('.test')
    ]
    test_classifier_multi_files(classifier_name, classifier_path, test_files, max_nums)


if __name__ == '__main__':
    # model, tokenizer = load_test_model(perf_path='./dpo_1/1/final_checkpoint')
    # # model.to(device)
    # prompt = "Please rewrite the following AI-generated text to make it more like human text, {without any useless content}:  I cannot definitively answer whether your chest pain is related to the intake of Clindamycin and Oxycodone without conducting a thorough examination or reviewing your medical history. However, I can share some information that may help you better understand the potential risks associated with these medications.\n\nClindamycin is an antibiotic that is sometimes associated with gastrointestinal (GI) side effects, including abdominal pain, diarrhea, and nausea. In rare cases, Clindamycin can cause serious GI conditions such as Clostridium difficile-associated diarrhea (CDAD). While chest pain is not a common side effect, it is possible that your GI symptoms could be causing referred pain in your chest.\n\nOxycodone is an opioid pain medication that is sometimes associated with side effects such as respiratory depression, dizziness, and constipation. Rarely, opioids can cause heart-related side effects such as arrhythmias or chest pain.\n\nIt is important to note that chest pain can have many different causes, including heart conditions, lung conditions, and gastroesophageal reflux disease (GERD). Therefore, it is crucial that you contact your healthcare provider as soon as possible to report your symptoms. They may recommend further testing to evaluate the underlying cause of your chest pain.\n\nIn the meantime, you can take the following steps to help manage your symptoms:\n\n1. Continue taking your medications as prescribed, but do not increase the dosage without speaking to your healthcare provider.\n2. Avoid taking large or frequent doses of opioids, as this can increase the risk of side effects.\n3. Stay hydrated by drinking plenty of water or other clear fluids.\n4. Eat smaller, more frequent meals throughout the day instead of large meals.\n5. Avoid caffeine, alcohol, and other substances that can irritate your GI tract.\n6. Practice deep breathing exercises to help reduce anxiety and improve oxygenation to your body.\n\nI hope this information is helpful. I encourage you to contact your healthcare provider with any concerns or questions you may have."
    # print(chat(model, tokenizer, prompt))
    # generate_datas(model, tokenizer, './mix_1/', '.mix.1000')

    test_classifier_base_dir('roberta-base', './dpo_1.pt', './qwen/')

    pass

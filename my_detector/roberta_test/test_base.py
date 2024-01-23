import pandas as pd
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import AutoModel, AutoTokenizer

from train_base import MyDataset

from train_base import MyClassifier


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


def init_test_model_and_tokenizer(base_model_name="roberta-base", test_model_path='best_model.pt'):
    # BERT_MODEL = "roberta-base"
    tokenizer = AutoTokenizer.from_pretrained(base_model_name)
    base_model = AutoModel.from_pretrained(base_model_name)
    model = (torch.load(test_model_path))
    # model = MyClassifier(base_model)
    model.eval()

    return model, tokenizer


def get_test_dataloader_and_labels(tokenizer, test_data_path="../Deberta_test/data/hc3_all.jsonl.test", batch_size=16):
    test_df = pd.read_json(test_data_path)
    test_dataset = MyDataset(test_df, tokenizer)
    test_dataloader = DataLoader(MyDataset(test_df, tokenizer), batch_size=batch_size)
    return test_dataloader, test_dataset.labels, test_dataset.domains, test_dataset.prompts


def get_acc(predictions, test_labels, domains, prompts):
    domain_results = {

    }
    for domain in domains:
        if domain_results.get(domain) is None:
            print(domain)
            domain_results[domain] = {
                'human_total': 0,
                'ai_total': 0,
                'human_acc': 0,
                'ai_acc': 0
            }
    prompt_results = {

    }
    for prompt in prompts:
        if prompt_results.get(prompt) is None:
            print(prompt)
            prompt_results[prompt] = {
                'human_total': 0,
                'ai_total': 0,
                'human_acc': 0,
                'ai_acc': 0
            }
    human_total = 0
    ai_total = 0
    human_acc = 0
    ai_acc = 0
    for i in range(0, len(predictions)):
        prompt = prompts[i]
        domain = domains[i]
        if test_labels[i] == 0:
            human_total += 1
            domain_results[domain]['human_total'] += 1
            prompt_results[prompt]['human_total'] += 1
            if predictions[i] == 0:
                human_acc += 1
                domain_results[domain]['human_acc'] += 1
                prompt_results[prompt]['human_acc'] += 1
        else:
            ai_total += 1
            domain_results[domain]['ai_total'] += 1
            prompt_results[prompt]['ai_total'] += 1
            if predictions[i] == 1:
                ai_acc += 1
                domain_results[domain]['ai_acc'] += 1
                prompt_results[prompt]['ai_acc'] += 1

    for domain in domain_results:
        domain_result = domain_results[domain]
        domain_result['ai_acc_r'] = domain_result['ai_acc'] / domain_result['ai_total']
        domain_result['human_acc_r'] = domain_result['human_acc'] / domain_result['human_total']
        domain_result['total_acc_r'] = (domain_result['human_acc'] + domain_result['ai_acc']) / (
                    domain_result['human_total'] + domain_result['ai_total'])

    for prompt in prompt_results:
        prompt_result = prompt_results[prompt]
        prompt_result['ai_acc_r'] = prompt_result['ai_acc'] / prompt_result['ai_total']
        prompt_result['human_acc_r'] = prompt_result['human_acc'] / prompt_result['human_total']
        prompt_result['total_acc_r'] = (prompt_result['human_acc'] + prompt_result['ai_acc']) / (
                    prompt_result['human_total'] + prompt_result['ai_total'])

    return {
        'prompts': prompt_results,
        'domains': domain_results,
        'all': {
            'ai_acc': ai_acc,
            'ai_total': ai_total,
            'ai_acc_r': ai_acc / ai_total,
            'human_acc': human_acc,
            'human_total': human_total,
            'human_acc_r': human_acc / human_total,
            'total_acc_r': (ai_acc + human_acc) / (ai_total + human_total)
        }
    }


if __name__ == '__main__':
    model, tokenizer = init_test_model_and_tokenizer()
    test_dataloader, test_labels, test_domains, test_prompts = get_test_dataloader_and_labels(tokenizer, './data/hc3_mix_multi_prompt.test')
    text_predictions = get_text_predictions(model, test_dataloader)
    print(get_acc(text_predictions, test_labels, test_domains, test_prompts))

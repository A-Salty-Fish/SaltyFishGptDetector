import json

import pandas as pd
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import AutoModel, AutoTokenizer

from train_base import MyDataset

from train_base import MyClassifier


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


def output_acc_with_key(key, acc_json):
    total_acc_r = acc_json['total_acc_r']
    human_total = acc_json['human_total']
    human_acc = acc_json['human_acc']
    human_acc_r = acc_json['human_acc_r']
    ai_total = acc_json['ai_total']
    ai_acc = acc_json['ai_acc']
    ai_acc_r = acc_json['ai_acc_r']
    print(f'{key}\t{total_acc_r}\t{human_total}\t{human_acc}\t{human_acc_r}\t{ai_total}\t{ai_acc}\t{ai_acc_r}')


def test_multi_prompt(model_path, test_file):
    save_model = model_path
    test_file = test_file
    model, tokenizer = init_test_model_and_tokenizer(test_model_path=save_model)
    test_dataloader, test_labels, test_domains, test_prompts = get_test_dataloader_and_labels(tokenizer, test_file)
    text_predictions = get_text_predictions(model, test_dataloader)
    acc_result = get_acc(text_predictions, test_labels, test_domains, test_prompts)
    print(acc_result)
    for key in acc_result['prompts']:
        output_acc_with_key(key, acc_result['prompts'][key])


# def test_multi_prompt_map(model_path, test_file_map):
#     save_model = model_path
#     test_file = test_file
#     model, tokenizer = init_test_model_and_tokenizer(test_model_path=save_model)
#     test_dataloader, test_labels, test_domains, test_prompts = get_test_dataloader_and_labels(tokenizer, test_file)
#     text_predictions = get_text_predictions(model, test_dataloader)
#     acc_result = get_acc(text_predictions, test_labels, test_domains, test_prompts)
#     print(acc_result)
#     for key in acc_result['prompts']:
#         output_acc_with_key(key, acc_result['prompts'][key])


if __name__ == '__main__':
    # prompt_json = {
    #     'academic': {'human_total': 1600, 'ai_total': 1600, 'human_acc': 1590, 'ai_acc': 499, 'ai_acc_r': 0.311875,
    #                  'human_acc_r': 0.99375, 'total_acc_r': 0.6528125},
    #     'continue': {'human_total': 1600, 'ai_total': 1600, 'human_acc': 1588, 'ai_acc': 1323, 'ai_acc_r': 0.826875,
    #                  'human_acc_r': 0.9925, 'total_acc_r': 0.9096875},
    #     'difficult': {'human_total': 1600, 'ai_total': 1600, 'human_acc': 1589, 'ai_acc': 4, 'ai_acc_r': 0.0025,
    #                   'human_acc_r': 0.993125, 'total_acc_r': 0.4978125},
    #     'easy': {'human_total': 1600, 'ai_total': 1600, 'human_acc': 1590, 'ai_acc': 903, 'ai_acc_r': 0.564375,
    #              'human_acc_r': 0.99375, 'total_acc_r': 0.7790625},
    #     'rewrite': {'human_total': 1600, 'ai_total': 1600, 'human_acc': 1592, 'ai_acc': 800, 'ai_acc_r': 0.5,
    #                 'human_acc_r': 0.995, 'total_acc_r': 0.7475},
    #     'qa': {'human_total': 5768, 'ai_total': 5770, 'human_acc': 5742, 'ai_acc': 4769, 'ai_acc_r': 0.8265164644714038,
    #            'human_acc_r': 0.9954923717059639, 'total_acc_r': 0.9109897729242503}}
    # for key in prompt_json:
    #     output_acc_with_key(key, prompt_json[key])


    save_model = 'hc3_mix_at.pt'
    test_file = './data/hc3_plus_qa_row.test'
    # test_file = './data/hc3_plus_qa_row.test'
    # model, tokenizer = init_test_model_and_tokenizer(test_model_path=save_model)
    # test_dataloader, test_labels, test_domains, test_prompts = get_test_dataloader_and_labels(tokenizer, test_file)
    # text_predictions = get_text_predictions(model, test_dataloader)
    # acc_result = get_acc(text_predictions, test_labels, test_domains, test_prompts)
    # print(acc_result)
    # for key in acc_result['prompts']:
    #     output_acc_with_key(key, acc_result['prompts'][key])

    test_multi_prompt('hc3_row.pt', test_file)
    test_multi_prompt('hc3_adt.pt', test_file)
    test_multi_prompt('hc3_random_adt.pt', test_file)
    test_multi_prompt('hc3_random_select_adt.pt', test_file)


    # for bar in [0.05, 0.1, 0.15, 0.2, 0.25, 0.3]:
    #     print(bar)
    #     text_predictions = get_text_predictions(model, test_dataloader, bar)
    #     print(get_acc(text_predictions, test_labels, test_domains, test_prompts))

    # row -> row
    # {'prompts': {'default': {'human_total': 5768, 'ai_total': 5770, 'human_acc': 5766, 'ai_acc': 3755,
    #                          'ai_acc_r': 0.6507798960138648, 'human_acc_r': 0.9996532593619972,
    #                          'total_acc_r': 0.8251863407869648}}, 'domains': {
    #     'default': {'human_total': 5768, 'ai_total': 5770, 'human_acc': 5766, 'ai_acc': 3755,
    #                 'ai_acc_r': 0.6507798960138648, 'human_acc_r': 0.9996532593619972,
    #                 'total_acc_r': 0.8251863407869648}},
    #  'all': {'ai_acc': 3755, 'ai_total': 5770, 'ai_acc_r': 0.6507798960138648, 'human_acc': 5766, 'human_total': 5768,
    #          'human_acc_r': 0.9996532593619972, 'total_acc_r': 0.8251863407869648}}

    # row -> mix
    # {'prompts': {
    #     'academic': {'human_total': 1600, 'ai_total': 1600, 'human_acc': 1590, 'ai_acc': 499, 'ai_acc_r': 0.311875,
    #                  'human_acc_r': 0.99375, 'total_acc_r': 0.6528125},
    #     'continue': {'human_total': 1600, 'ai_total': 1600, 'human_acc': 1588, 'ai_acc': 1323, 'ai_acc_r': 0.826875,
    #                  'human_acc_r': 0.9925, 'total_acc_r': 0.9096875},
    #     'difficult': {'human_total': 1600, 'ai_total': 1600, 'human_acc': 1589, 'ai_acc': 4, 'ai_acc_r': 0.0025,
    #                   'human_acc_r': 0.993125, 'total_acc_r': 0.4978125},
    #     'easy': {'human_total': 1600, 'ai_total': 1600, 'human_acc': 1590, 'ai_acc': 903, 'ai_acc_r': 0.564375,
    #              'human_acc_r': 0.99375, 'total_acc_r': 0.7790625},
    #     'rewrite': {'human_total': 1600, 'ai_total': 1600, 'human_acc': 1592, 'ai_acc': 800, 'ai_acc_r': 0.5,
    #                 'human_acc_r': 0.995, 'total_acc_r': 0.7475},
    #     'qa': {'human_total': 5768, 'ai_total': 5770, 'human_acc': 5742, 'ai_acc': 4769, 'ai_acc_r': 0.8265164644714038,
    #            'human_acc_r': 0.9954923717059639, 'total_acc_r': 0.9109897729242503}}, 'domains': {
    #     'finance': {'human_total': 5146, 'ai_total': 5147, 'human_acc': 5143, 'ai_acc': 3878,
    #                 'ai_acc_r': 0.7534486108412668, 'human_acc_r': 0.9994170229304314,
    #                 'total_acc_r': 0.8764208685514427},
    #     'medicine': {'human_total': 2998, 'ai_total': 2999, 'human_acc': 2996, 'ai_acc': 1851,
    #                  'ai_acc_r': 0.6172057352450817, 'human_acc_r': 0.9993328885923949,
    #                  'total_acc_r': 0.808237452059363},
    #     'open_qa': {'human_total': 2950, 'ai_total': 2950, 'human_acc': 2909, 'ai_acc': 1626,
    #                 'ai_acc_r': 0.5511864406779661, 'human_acc_r': 0.9861016949152542,
    #                 'total_acc_r': 0.7686440677966102},
    #     'wiki_csai': {'human_total': 2674, 'ai_total': 2674, 'human_acc': 2643, 'ai_acc': 943,
    #                   'ai_acc_r': 0.3526551982049364, 'human_acc_r': 0.9884068810770381,
    #                   'total_acc_r': 0.6705310396409873}},
    #  'all': {'ai_acc': 8298, 'ai_total': 13770, 'ai_acc_r': 0.6026143790849673, 'human_acc': 13691,
    #          'human_total': 13768, 'human_acc_r': 0.9944073213248111, 'total_acc_r': 0.7984966228484276}}

    # mix -> mix
    # {'prompts': {'academic': {'human_total': 1600, 'ai_total': 1600, 'human_acc': 1598, 'ai_acc': 1600, 'ai_acc_r': 1.0,
    #                           'human_acc_r': 0.99875, 'total_acc_r': 0.999375},
    #              'continue': {'human_total': 1600, 'ai_total': 1600, 'human_acc': 1599, 'ai_acc': 1600, 'ai_acc_r': 1.0,
    #                           'human_acc_r': 0.999375, 'total_acc_r': 0.9996875},
    #              'difficult': {'human_total': 1600, 'ai_total': 1600, 'human_acc': 1598, 'ai_acc': 1600,
    #                            'ai_acc_r': 1.0, 'human_acc_r': 0.99875, 'total_acc_r': 0.999375},
    #              'easy': {'human_total': 1600, 'ai_total': 1600, 'human_acc': 1598, 'ai_acc': 1600, 'ai_acc_r': 1.0,
    #                       'human_acc_r': 0.99875, 'total_acc_r': 0.999375},
    #              'rewrite': {'human_total': 1600, 'ai_total': 1600, 'human_acc': 1598, 'ai_acc': 1600, 'ai_acc_r': 1.0,
    #                          'human_acc_r': 0.99875, 'total_acc_r': 0.999375},
    #              'qa': {'human_total': 5768, 'ai_total': 5770, 'human_acc': 5766, 'ai_acc': 5769,
    #                     'ai_acc_r': 0.9998266897746967, 'human_acc_r': 0.9996532593619972,
    #                     'total_acc_r': 0.999739989599584}}, 'domains': {
    #     'finance': {'human_total': 5146, 'ai_total': 5147, 'human_acc': 5146, 'ai_acc': 5146,
    #                 'ai_acc_r': 0.9998057120652808, 'human_acc_r': 1.0, 'total_acc_r': 0.9999028465947731},
    #     'medicine': {'human_total': 2998, 'ai_total': 2999, 'human_acc': 2998, 'ai_acc': 2999, 'ai_acc_r': 1.0,
    #                  'human_acc_r': 1.0, 'total_acc_r': 1.0},
    #     'open_qa': {'human_total': 2950, 'ai_total': 2950, 'human_acc': 2950, 'ai_acc': 2950, 'ai_acc_r': 1.0,
    #                 'human_acc_r': 1.0, 'total_acc_r': 1.0},
    #     'wiki_csai': {'human_total': 2674, 'ai_total': 2674, 'human_acc': 2663, 'ai_acc': 2674, 'ai_acc_r': 1.0,
    #                   'human_acc_r': 0.9958863126402393, 'total_acc_r': 0.9979431563201197}},
    #  'all': {'ai_acc': 13769, 'ai_total': 13770, 'ai_acc_r': 0.9999273783587509, 'human_acc': 13757,
    #          'human_total': 13768, 'human_acc_r': 0.9992010459035444, 'total_acc_r': 0.9995642385067907}}

    # mix -> hc3
    # {'prompts': {'default': {'human_total': 5768, 'ai_total': 5770, 'human_acc': 5766, 'ai_acc': 3755,
    #                          'ai_acc_r': 0.6507798960138648, 'human_acc_r': 0.9996532593619972,
    #                          'total_acc_r': 0.8251863407869648}}, 'domains': {
    #     'default': {'human_total': 5768, 'ai_total': 5770, 'human_acc': 5766, 'ai_acc': 3755,
    #                 'ai_acc_r': 0.6507798960138648, 'human_acc_r': 0.9996532593619972,
    #                 'total_acc_r': 0.8251863407869648}},
    #  'all': {'ai_acc': 3755, 'ai_total': 5770, 'ai_acc_r': 0.6507798960138648, 'human_acc': 5766, 'human_total': 5768,
    #          'human_acc_r': 0.9996532593619972, 'total_acc_r': 0.8251863407869648}}

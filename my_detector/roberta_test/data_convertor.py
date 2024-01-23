
import json

def convert_hc3_multi_jsonls_to_train_and_test_json(jsonl_files, target_json_file, train_rate=0.2):
    test_jsons = []
    train_jsons = []

    for jsonl_file in jsonl_files:
        jsonl_file_jsons = []
        file_name = jsonl_file.split('/')[-1]
        domain_name = file_name.split('.')[0]
        prompt_name = file_name.split('.')[1]
        if prompt_name == 'mix':
            prompt_name = 'qa'
        with open(jsonl_file, 'r', encoding='utf-8') as in_jsonl_file:
            for line in in_jsonl_file:
                json_obj = json.loads(line)
                human_obj = {
                    'label':0,
                    'content':json_obj['human'],
                    'domain': domain_name,
                    'prompt': prompt_name
                }
                ai_obj = {
                    'label': 1,
                    'content': json_obj['ai'],
                    'domain': domain_name,
                    'prompt': prompt_name
                }
                jsonl_file_jsons.append(human_obj)
                jsonl_file_jsons.append(ai_obj)
        train_jsons += jsonl_file_jsons[0: int(len(jsonl_file_jsons) * train_rate)]
        test_jsons += jsonl_file_jsons[int(len(jsonl_file_jsons) * train_rate):]

    with open(target_json_file + '.test', 'w', encoding='utf-8') as test_output:
        test_output.write(json.dumps(test_jsons))
    with open(target_json_file + '.train', 'w', encoding='utf-8') as train_output:
        train_output.write(json.dumps(train_jsons))

def convet_multi_domain_prompt():
    multi_domains = [
        'finance',
        'medicine',
        'open_qa',
        'wiki_csai'
    ]
    multi_prompts = [
        'academic',
        'continue',
        'difficult',
        'easy',
        'rewrite'
    ]
    direct_files = []
    for domain in multi_domains:
        for prompt in multi_prompts:
            direct_files.append(
                '../../data_collector/test_data/hc3_english_mix_multi/' + domain + '.' + prompt + '.mix.jsonl')
        direct_files.append('../../data_collector/test_data/hc3_english_mix_multi/' + domain + '.mix.jsonl')
    convert_hc3_multi_jsonls_to_train_and_test_json(direct_files, './data/hc3_mix_multi_prompt')


if __name__ == '__main__':
    convet_multi_domain_prompt()
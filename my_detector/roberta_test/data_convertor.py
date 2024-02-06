
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


def convert_hc3_jsonls_to_train_and_test_json(jsonl_files, target_json_file, train_rate=0.2):
    test_jsons = []
    train_jsons = []

    for jsonl_file in jsonl_files:
        jsonl_file_jsons = []
        file_name = jsonl_file.split('/')[-1]
        domain_name = 'default'
        prompt_name = 'default'
        if prompt_name == 'mix':
            prompt_name = 'qa'
        with open(jsonl_file, 'r', encoding='utf-8') as in_jsonl_file:
            for line in in_jsonl_file:
                json_obj = json.loads(line)
                human_obj = {
                    'label': 0,
                    'content': json_obj['human_answers'][0].replace('\n', ''),
                    'domain': domain_name,
                    'prompt': prompt_name
                }
                ai_obj = {
                    'label': 1,
                    'content': json_obj['chatgpt_answers'][0].replace('\n', ''),
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


def convert_hc3_plus():


    #处理hc3 plus数据集
    jsons = []
    with open('../../data_collector/test_data/hc3_plus_english/test_hc3_QA.jsonl', 'r', encoding='utf-8') as q_in_f:
        for line in q_in_f:
            row_obj = json.loads(line)
            jsons.append({
                'content': row_obj['text'],
                'label': row_obj['label']
            })
    with open('./data/hc3_plus_qa_row.test', 'w', encoding='utf-8') as q_out_f:
        q_out_f.write(json.dumps(jsons))
    jsons = []
    with open('../../data_collector/test_data/hc3_plus_english/test_hc3_si.jsonl', 'r', encoding='utf-8') as si_in_f:
        for line in si_in_f:
            row_obj = json.loads(line)
            jsons.append({
                'content': row_obj['text'],
                'label': row_obj['label']
            })
    with open('./data/hc3_plus_si_row.test', 'w', encoding='utf-8') as si_out_f:
        si_out_f.write(json.dumps(jsons))


def convert_cheat():
    # 处理cheat数据集
    jsons = []
    with open('../../data_collector/test_data/CHEAT/ieee-chatgpt-generation.jsonl', 'r', encoding='utf-8') as q_in_f:
        for line in q_in_f:
            row_obj = json.loads(line)
            jsons.append({
                'content': row_obj['abstract'],
                'label': 1
            })
    with open('../../data_collector/test_data/CHEAT/ieee-init.jsonl', 'r', encoding='utf-8') as q_in_f:
        for line in q_in_f:
            row_obj = json.loads(line)
            jsons.append({
                'content': row_obj['abstract'],
                'label': 0
            })
    with open('./data/cheat_generation.test', 'w', encoding='utf-8') as q_out_f:
        q_out_f.write(json.dumps(jsons))
    jsons = []
    with open('../../data_collector/test_data/CHEAT/ieee-chatgpt-polish.jsonl', 'r', encoding='utf-8') as si_in_f:
        for line in si_in_f:
            row_obj = json.loads(line)
            jsons.append({
                'content': row_obj['abstract'],
                'label': 1
            })
    with open('../../data_collector/test_data/CHEAT/ieee-init.jsonl', 'r', encoding='utf-8') as q_in_f:
        for line in q_in_f:
            row_obj = json.loads(line)
            jsons.append({
                'content': row_obj['abstract'],
                'label': 0
            })
    with open('./data/cheat_polish.test', 'w', encoding='utf-8') as si_out_f:
        si_out_f.write(json.dumps(jsons))


def convert_ghostbuster():
    # 处理ghostbuster数据集
    jsons = []
    with open('../../data_collector/test_data/ghostbuster/essay_claude.txt', 'r', encoding='utf-8') as q_in_f:
        for line in q_in_f:
            jsons.append({
                'label': 1,
                'content': line
            })
    with open('../../data_collector/test_data/ghostbuster/essay_human.txt', 'r', encoding='utf-8') as q_in_f:
        for line in q_in_f:
            jsons.append({
                'label': 0,
                'content': line
            })
    with open('./data/ghostbuster_claude.test', 'w', encoding='utf-8') as si_out_f:
        si_out_f.write(json.dumps(jsons))

if __name__ == '__main__':

    convert_ghostbuster()



    # convet_multi_domain_prompt()
    # convert_hc3_jsonls_to_train_and_test_json(
    #     [
    #         '../../data_collector/test_data/hc3_english/finance.jsonl',
    #         '../../data_collector/test_data/hc3_english/medicine.jsonl',
    #         '../../data_collector/test_data/hc3_english/open_qa.jsonl',
    #         '../../data_collector/test_data/hc3_english/wiki_csai.jsonl',
    #      ],
    #     './data/hc3_row'
    # )
    # local_file_map = {
    #     'rewrite': '../../data_collector/test_data/hc3_english_mix_multi/open_qa.rewrite.mix.jsonl',
    #     'continue': '../../data_collector/test_data/hc3_english_mix_multi/open_qa.continue.mix.jsonl',
    #     'academic': '../../data_collector/test_data/hc3_english_mix_multi/open_qa.academic.mix.jsonl',
    #     'difficult': '../../data_collector/test_data/hc3_english_mix_multi/open_qa.difficult.mix.jsonl',
    #     'easy': '../../data_collector/test_data/hc3_english_mix_multi/open_qa.easy.mix.jsonl',
    #     'qa': '../../data_collector/test_data/hc3_english_mix_multi/open_qa.mix.jsonl',
    # }
    # for f in local_file_map:
    #     jsons = []
    #     with open(local_file_map[f], 'r', encoding='utf-8') as in_f:
    #         for line in in_f:
    #             jsons.append(json.loads(line))
    #     with open(local_file_map[f]+'.train', 'w', encoding='utf-8') as train_f:
    #         for obj in jsons[0: int(len(jsons) * 0.2)]:
    #             train_f.write(json.dumps(obj) + '\n')
    #     with open(local_file_map[f]+'.test', 'w', encoding='utf-8') as test_f:
    #         for obj in jsons[int(len(jsons) * 0.2):]:
    #             test_f.write(json.dumps(obj) + '\n')

    pass
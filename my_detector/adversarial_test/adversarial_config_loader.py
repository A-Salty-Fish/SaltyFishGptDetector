import json


def load_text_labels_config(base_path='./config/', config_name='generator.json'):
    with open(base_path + config_name, 'r', encoding='utf-8') as text_labels_file:
        return json.load(text_labels_file)['candidate_labels']


def load_prompt_templates_config(base_path='./config/', config_name='generator.json'):
    with open(base_path + config_name, 'r', encoding='utf-8') as text_labels_file:
        return json.load(text_labels_file)['prompt_templates']


def load_train_base_model_config(base_path='./config/', config_name='base_model.json'):
    with open(base_path + config_name, 'r', encoding='utf-8') as text_labels_file:
        return json.load(text_labels_file)['train']


def load_test_base_model_config(base_path='./config/', config_name='base_model.json'):
    with open(base_path + config_name, 'r', encoding='utf-8') as text_labels_file:
        return json.load(text_labels_file)['test']


def load_chat_base_model_config(base_path='./config/', config_name='base_model.json'):
    with open(base_path + config_name, 'r', encoding='utf-8') as text_labels_file:
        return json.load(text_labels_file)['chat']


if __name__ == '__main__':
    print(load_text_labels_config())
    print(load_prompt_templates_config())
    print(load_train_base_model_config())
    print(load_test_base_model_config())
    print(load_chat_base_model_config())

import json
import time

from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline


def load_text_labels_config(base_path='./config/', config_name='generator.json'):
    with open(base_path + config_name, 'r', encoding='utf-8') as text_labels_file:
        return json.load(text_labels_file)['candidate_labels']


def load_prompt_templates_config(base_path='./config/', config_name='generator.json'):
    with open(base_path + config_name, 'r', encoding='utf-8') as text_labels_file:
        return json.load(text_labels_file)['prompt_templates']


def load_chat_base_model_config(base_path='./config/', config_name='base_model.json'):
    with open(base_path + config_name, 'r', encoding='utf-8') as text_labels_file:
        return json.load(text_labels_file)['chat']


def load_utc_base_model_config(base_path='./config/', config_name='base_model.json'):
    with open(base_path + config_name, 'r', encoding='utf-8') as text_labels_file:
        return json.load(text_labels_file)['utc']


def init_generator_model_and_tokenizer(chat_base_model_config):
    start_time = time.time()
    tokenizer = AutoTokenizer.from_pretrained(chat_base_model_config['tokenizer_name'])
    model = AutoModelForCausalLM.from_pretrained(chat_base_model_config['model_name'])

    end_time = time.time()
    print("load generator model successful: " + str(end_time - start_time))
    return model, tokenizer


def chat(model, tokenizer, context):
    device = 'cuda'
    messages = [
        {"role": "user", "content": context}
    ]
    encodeds = tokenizer.apply_chat_template(messages, return_tensors="pt")
    model_inputs = encodeds.to(device)
    model.to(device)
    generated_ids = model.generate(model_inputs, max_new_tokens=1000, do_sample=True,
                                   pad_token_id=tokenizer.eos_token_id)
    decoded = tokenizer.batch_decode(generated_ids)
    return decoded[0].split('[/INST]')[1].replace('</s>', '')


def init_utc_pipe(utc_base_model_config):
    start_time = time.time()

    type = utc_base_model_config['type']
    model_name = utc_base_model_config['model_name']

    classifier = pipeline(type, model=model_name)
    end_time = time.time()
    print("load utc model success: " + str(end_time - start_time))
    return classifier


def utc_classify(classifier, labels, text):
    result = []
    output = classifier(text, labels, multi_label=False)
    for i in range(0, len(output['labels'])):
        result.append([output['labels'][i], output['scores'][i]])
    if len(result) == 0:
        result.append(['None', 1.00])
    return result


if __name__ == '__main__':

    chat_base_model_config = load_chat_base_model_config()
    model, tokenizer = init_generator_model_and_tokenizer(chat_base_model_config)

    utc_base_model_config = load_utc_base_model_config()
    text_labels = load_text_labels_config()
    classifier = init_utc_pipe(utc_base_model_config)

    chat_res = chat(model, tokenizer, "hello, can you tell me something about smart phone. about 100 words")
    print(chat_res)

    print(utc_classify(classifier, text_labels, chat_res))
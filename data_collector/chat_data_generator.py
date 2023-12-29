import json
import time

from transformers import AutoModelForCausalLM, AutoTokenizer

device = "cuda"  # the device to load the model onto


def init_model_and_tokenizer():
    start_time = time.time()
    model = AutoModelForCausalLM.from_pretrained("mistralai/Mistral-7B-Instruct-v0.1")
    tokenizer = AutoTokenizer.from_pretrained("mistralai/Mistral-7B-Instruct-v0.1")

    end_time = time.time()
    print("load model successful: " + str(end_time - start_time))
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
    model.to(device)

    generated_ids = model.generate(model_inputs, max_new_tokens=1000, do_sample=True,
                                   pad_token_id=tokenizer.eos_token_id)
    decoded = tokenizer.batch_decode(generated_ids)
    # end_time = time.time()
    # print("generate response successful: " + str(end_time - start_time))
    return decoded[0].split('[/INST]')[1].replace('</s>', '')


def rewrite(model, tokenizer, content):
    prompt_template = "Please rewrite the following paragraphs to keep the original meaning intact and prohibit the output of irrelevant content:"
    prompt = prompt_template + content
    return chat(model, tokenizer, prompt)


def replace_words(model, tokenizer, content):
    prompt_template = "Please change some words the following paragraphs to keep the original meaning intact and prohibit the output of irrelevant content:"
    prompt = prompt_template + content
    return chat(model, tokenizer, prompt)


def continue_content(model, tokenizer, content):
    prompt_template = "Please continue writing about 100 words based on the following paragraphs and prohibit the output of irrelevant content:"
    prompt = prompt_template + content
    return chat(model, tokenizer, prompt)


def academic_content(model, tokenizer, content):
    prompt_template = "Make the following paragraphs more academic and prohibit the output of irrelevant content:"
    prompt = prompt_template + content
    return chat(model, tokenizer, prompt)


def summarize_content(model, tokenizer, content):
    prompt_template = "Summarize the following paragraphs and prohibit the output of irrelevant content:"
    prompt = prompt_template + content
    return chat(model, tokenizer, prompt)


def rewrite_objs(model, tokenizer):
    json_objs = []
    with open('./row_data/arxiv_paras/7.jsonl', 'r', encoding='utf-8') as f7, open('./row_data/arxiv_paras/8.jsonl',
                                                                                   'r', encoding='utf-8') as f8, open(
            './row_data/arxiv_paras/9.jsonl', 'r', encoding='utf-8') as f9:
        with open('./rewrite_1.jsonl', 'a', encoding='utf-8') as out_f:
            for line in f7:
                try:
                    json_obj = json.loads(line)
                    json_objs.append(json_obj)
                except Exception as e:
                    print(e)
            for line in f8:
                try:
                    json_obj = json.loads(line)
                    json_objs.append(json_obj)
                except Exception as e:
                    print(e)
            for line in f9:
                try:
                    json_obj = json.loads(line)
                    json_objs.append(json_obj)
                except Exception as e:
                    print(e)
            total_num = 4000
            for i in range(0, total_num):
                print('test process : %s [%d/%d]' % (str(i * 100 / total_num) + '%', i, total_num), end='\r')
                json_obj = json_objs[i]
                json_obj['rewrite'] = rewrite(model, tokenizer, json_obj['content'])
                out_f.write(json.dumps(json_obj, ensure_ascii=False) + '\n')


def replace_objs(model, tokenizer):
    json_objs = []
    with open('./row_data/arxiv_paras/7.jsonl', 'r', encoding='utf-8') as f7, open('./row_data/arxiv_paras/8.jsonl',
                                                                                   'r', encoding='utf-8') as f8, open(
            './row_data/arxiv_paras/9.jsonl', 'r', encoding='utf-8') as f9:
        with open('./replace_1.jsonl', 'a', encoding='utf-8') as out_f:
            for line in f7:
                try:
                    json_obj = json.loads(line)
                    json_objs.append(json_obj)
                except Exception as e:
                    print(e)
            for line in f8:
                try:
                    json_obj = json.loads(line)
                    json_objs.append(json_obj)
                except Exception as e:
                    print(e)
            for line in f9:
                try:
                    json_obj = json.loads(line)
                    json_objs.append(json_obj)
                except Exception as e:
                    print(e)
            total_num = 4000
            for i in range(0, total_num):
                print('test process : %s [%d/%d]' % (str(i * 100 / total_num) + '%', i, total_num), end='\r')
                json_obj = json_objs[i]
                json_obj['replace'] = replace_words(model, tokenizer, json_obj['content'])
                out_f.write(json.dumps(json_obj, ensure_ascii=False) + '\n')


def continue_objs(model, tokenizer):
    json_objs = []
    with open('./row_data/arxiv_paras/7.jsonl', 'r', encoding='utf-8') as f7, open('./row_data/arxiv_paras/8.jsonl',
                                                                                   'r', encoding='utf-8') as f8, open(
        './row_data/arxiv_paras/9.jsonl', 'r', encoding='utf-8') as f9:
        with open('./continue_1.jsonl', 'a', encoding='utf-8') as out_f:
            for line in f7:
                try:
                    json_obj = json.loads(line)
                    json_objs.append(json_obj)
                except Exception as e:
                    print(e)
            for line in f8:
                try:
                    json_obj = json.loads(line)
                    json_objs.append(json_obj)
                except Exception as e:
                    print(e)
            for line in f9:
                try:
                    json_obj = json.loads(line)
                    json_objs.append(json_obj)
                except Exception as e:
                    print(e)
            total_num = 4000
            for i in range(0, total_num):
                print('test process : %s [%d/%d]' % (str(i * 100 / total_num) + '%', i, total_num), end='\r')
                json_obj = json_objs[i]
                json_obj['continue'] = continue_content(model, tokenizer, json_obj['content'])
                out_f.write(json.dumps(json_obj, ensure_ascii=False) + '\n')


def generate_chat_objs(model, tokenizer, output_file_name, chat_funcion, result_key="chat_content",
                       inputs_file_names=['7.jsonl', '8.jsonl', '9.jsonl'], total_num=4000):
    with open('./' + output_file_name, 'a', encoding='utf-8') as out_f:
        json_objs = []
        for inputs_file_name in inputs_file_names:
            with open('./row_data/arxiv_paras/' + inputs_file_name, 'r', encoding='utf-8') as in_f:
                for line in in_f:
                    try:
                        json_obj = json.loads(line)
                        json_objs.append(json_obj)
                    except Exception as e:
                        print(e)
        for i in range(0, total_num):
            print('test process : %s [%d/%d]' % (str(i * 100 / total_num) + '%', i, total_num), end='\r')
            json_obj = json_objs[i]
            json_obj[result_key] = chat_funcion(model, tokenizer, json_obj['content'])
            out_f.write(json.dumps(json_obj, ensure_ascii=False) + '\n')


# 初始化人类pdf内容
def arxiv_init():
    json_objs = []
    with open('./row_data/arxiv_paras/7.jsonl', 'r', encoding='utf-8') as f7, open('./row_data/arxiv_paras/8.jsonl',
                                                                                   'r', encoding='utf-8') as f8, open(
            './row_data/arxiv_paras/9.jsonl', 'r', encoding='utf-8') as f9:
        with open('./init_1.jsonl', 'a', encoding='utf-8') as out_f:
            for line in f7:
                try:
                    json_obj = json.loads(line)
                    json_objs.append(json_obj)
                except Exception as e:
                    print(e)
            for line in f8:
                try:
                    json_obj = json.loads(line)
                    json_objs.append(json_obj)
                except Exception as e:
                    print(e)
            for line in f9:
                try:
                    json_obj = json.loads(line)
                    json_objs.append(json_obj)
                except Exception as e:
                    print(e)
            total_num = 4000
            for i in range(0, total_num):
                print('test process : %s [%d/%d]' % (str(i * 100 / total_num) + '%', i, total_num), end='\r')
                json_obj = json_objs[i]
                out_f.write(json.dumps(json_obj, ensure_ascii=False) + '\n')


if __name__ == '__main__':
    # arxiv_init()
    model, tokenizer = init_model_and_tokenizer()
    # start_time = time.time()
    # print("begin rewrite")
    # rewrite_objs(model, tokenizer)
    # end_time = time.time()
    # print("end rewrite: " + str(end_time - start_time))
    # print("begin replace")
    # rewrite_objs(model, tokenizer)
    # end_time = time.time()
    # print("end replace: " + str(end_time - start_time))
    print("begin continue")
    start_time = time.time()
    generate_chat_objs(model, tokenizer, 'continue_1.jsonl' ,continue_content, result_key='continue')
    end_time = time.time()
    print("end continue: " + str(end_time - start_time))
    print("begin academic")
    start_time = time.time()
    generate_chat_objs(model, tokenizer, 'academic_1.jsonl' ,academic_content, result_key='academic')
    end_time = time.time()
    print("end academic: " + str(end_time - start_time))
    print("begin summarize")
    start_time = time.time()
    generate_chat_objs(model, tokenizer, 'summarize_1.jsonl' ,summarize_content, result_key='summarize')
    end_time = time.time()
    print("end summarize: " + str(end_time - start_time))
    # print(chat(model, tokenizer, "Please rewrite the following paragraphs to keep the original meaning intact and prohibit the output of irrelevant content: As the Web matures, an increasing number of dynamic information sources and services come online. Unlike static Web pages, these resources generate their contents dynamically in response to a query. They can be HTML-based, searching the site via an HTML form, or be a Web service. Proliferation of such resources has led to a number of novel applications, including Web-based mashups, such as Google maps and Yahoo pipes, information integration"))

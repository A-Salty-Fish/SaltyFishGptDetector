import json
import random

if __name__ == '__main__':
    text_labels = [
        "medicine",
        "law",
        "computer science",
        "finance",
        "pedagogy",
        "biology",
        "psychology",
        "political",
        "sports",
        "chemistry"
    ]
    #
    # test_datas = []
    # with open('./data/class_chat_data.jsonl', 'r', encoding='utf-8') as test_input:
    #     for line in test_input:
    #         json_obj = json.loads(line)
    #         if json_obj['prompt_template'] != "please pretend to answer a question in the field of {}. about 50 words ":
    #             continue
    #         test_datas.append({
    #             'label': json_obj['label'],
    #             'content': json_obj['content'],
    #             "text_label": json_obj['class_label']
    #         })
    # # to data map
    # label_data_map = {}
    # for text_label in text_labels:
    #     label_data_map[text_label] = []
    # for test_data in test_datas:
    #     if test_data['text_label'] in text_labels:
    #         label_data_map[test_data['text_label']].append(test_data)
    # # random list
    # # limit size
    # for text_label in text_labels:
    #     if len(label_data_map[text_label]) > 100:
    #         label_data_map[text_label] = label_data_map[text_label][0: 100]
    #
    # json_arr_result = []
    #
    # with open('./data/medicine.jsonl.train', 'r', encoding='utf-8') as row_file:
    #     json_arr = json.load(row_file)
    #     for text_label in text_labels:
    #         for data in label_data_map[text_label]:
    #             json_arr_result.append({
    #                 'label': 1,
    #                 'content': data['content']
    #             })
    #     for json_obj in json_arr:
    #         if json_obj['label'] == 1:
    #             # json_arr_result.append(json_obj)
    #             pass
    #         else:
    #             for i in range(0, 5):
    #                 json_arr_result.append(json_obj)
    #     random.shuffle(json_arr_result)
    #     with open('./tmp/train_1/medicine.jsonl.train.1', 'w', encoding='utf-8') as output:
    #         output.write(json.dumps(json_arr_result))

    # label_contents = {}
    # for label in text_labels:
    #     label_contents[label] = []
    # with open('./data/cp_wiki_qa_mix.jsonl', 'r', encoding='utf-8') as f:
    #     for line in f:
    #         json_obj = json.loads(line)
    #         ai_content = json_obj['ai'].replace('\n', '')
    #
    # with open('./label_result', 'w', encoding='utf-8') as f_out:
    #     f_out.write(json.dumps(label_contents, ensure_ascii=False))

    with open('./data/label_result', 'r', encoding='utf-8') as f:
        json_map = json.load(f)
        for text_label in text_labels:
            print(text_label)
            print(len(json_map[text_label]))

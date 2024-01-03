import json

if __name__ == '__main__':
    text_labels= [
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

    test_datas = []
    with open('./data/class_chat_data.jsonl', 'r', encoding='utf-8') as test_input:
        for line in test_input:
            json_obj = json.loads(line)
            test_datas.append({
                'label': json_obj['label'],
                'content': json_obj['content'],
                "text_label": json_obj['class_label']
            })
    # to data map
    label_data_map = {}
    for text_label in text_labels:
        label_data_map[text_label] = []
    for test_data in test_datas:
        if test_data['text_label'] in text_labels:
            label_data_map[test_data['text_label']].append(test_data)
    # random list
    # limit size
    for text_label in text_labels:
        if len(label_data_map[text_label]) > 100:
            label_data_map[text_label] = label_data_map[text_label][0: 100]

    with open('./data/medicine.jsonl.train', 'r', encoding='utf-8') as row_file:
        json_arr = json.load(row_file)
        for text_label in text_labels:
            for data in label_data_map[text_label]:
                json_arr.append({
                    'label': 1,
                    'content': data['content']
                })
        with open('./tmp/train_1/medicine.jsonl.train.1', 'w', encoding='utf-8') as output:
            output.write(json.dumps(json_arr))
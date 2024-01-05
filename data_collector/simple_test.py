import json

if __name__ == '__main__':
    with open('./test_data/quac/train_v0.2.json', 'r', encoding='utf-8') as f:
        json_obj = json.load(f)['data']
        for k in json_obj:
            print(k)

        # print(json.dumps(json_obj, indent=4))
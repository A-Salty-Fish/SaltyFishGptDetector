
import json
import os


def convert_jsonl_to_train_test(jsonl_file, train_rate=0.2):
    json_objs = []
    with open(jsonl_file, 'r', encoding='utf-8') as in_f:
        for line in in_f:
            json_obj = json.loads(line)
            json_objs.append(
                {
                    'label': 0,
                    'content': json_obj['content']
                }
            )
            json_objs.append(
                {
                    'label': 1,
                    'content': json_obj['ai_rewrite']
                }
            )
    with open(jsonl_file + '.train', 'w', encoding='utf-8') as train_f:
        train_f.write(json.dumps(json_objs[0: int(train_rate*len(json_objs))]))
    with open(jsonl_file + '.all.test', 'w', encoding='utf-8') as train_f:
        train_f.write(json.dumps(json_objs[int(train_rate*len(json_objs)):]))


def merge_train_file(train_files, out_file_name):
    results = []
    for train_file in train_files:
        with open(train_file, 'r', encoding='utf-8') as in_f:
            results += json.load(in_f)
    with open(out_file_name, 'w', encoding='utf-8') as out_f:
        out_f.write(json.dumps(results))




if __name__ == '__main__':
    # dirs = [
    #     # './data/nature/glm/',
    #     # './data/nature/mix/',
    #     # './data/nature/qwen/'
    #     './data/adversary/dp/',
    #     './data/adversary/dpo/',
    #     './data/adversary/qwen/',
    # ]
    # for dir in dirs:
    #     for file in os.listdir(dir):
    #         convert_jsonl_to_train_test(dir + file)


    moe_train_files = [
        './data/nature/qwen/7.jsonl.qwen.rewrite.jsonl.train',
        # './data/nature/qwen/8.jsonl.qwen.rewrite.jsonl.train',
        # './data/nature/qwen/9.jsonl.qwen.rewrite.jsonl.train',
        # './data/nature/qwen/10.jsonl.qwen.rewrite.jsonl.train',
        './data/adversary/qwen/7.jsonl.qwen.rewrite.jsonl.qwen.paraphase.jsonl.train',
        # './data/adversary/qwen/8.jsonl.qwen.rewrite.jsonl.qwen.paraphase.jsonl.train',
        # './data/adversary/qwen/9.jsonl.qwen.rewrite.jsonl.qwen.paraphase.jsonl.train',
        # './data/adversary/qwen/10.jsonl.qwen.rewrite.jsonl.qwen.paraphase.jsonl.train',
    ]

    merge_train_file(moe_train_files, 'arxiv_qwen_7.train')
    pass
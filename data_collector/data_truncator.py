# 用来分割数据集数据，避免过长的数据影响benchmark
import json


def truncate_CHEAT(raw_cheat_path, truncated_cheat_path, truncate_length=256, drop_length = 128):
    fusion_file_name = 'ieee-chatgpt-fusion.jsonl'
    generation_file_name = 'ieee-chatgpt-generation.jsonl'
    polish_file_name = 'ieee-chatgpt-polish.jsonl'
    init_file_name = 'ieee-init.jsonl'
    with open(raw_cheat_path + fusion_file_name, 'r', encoding='utf-8') as fusion_file, open(raw_cheat_path + generation_file_name, 'r', encoding='utf-8') as generation_file, open(raw_cheat_path + polish_file_name, 'r', encoding='utf-8') as polish_file, open(raw_cheat_path + init_file_name, 'r', encoding='utf-8') as init_file:
        with open(truncated_cheat_path + fusion_file_name, 'w', encoding='utf-8') as fusion_out_file:
            for line in fusion_file:
                json_obj = json.loads(line)
                truncated_sentences = truncate_content(content=json_obj['abstract'], truncate_length=truncate_length, drop_length=drop_length)
                for truncated_sentence in truncated_sentences:
                    json_obj['abstract'] = truncated_sentence
                    fusion_out_file.write(json.dumps(json_obj) + '\n')
        with open(truncated_cheat_path + generation_file_name, 'w', encoding='utf-8') as generation_out_file:
            for line in generation_file:
                json_obj = json.loads(line)
                truncated_sentences = truncate_content(content=json_obj['abstract'], truncate_length=truncate_length, drop_length=drop_length)
                for truncated_sentence in truncated_sentences:
                    json_obj['abstract'] = truncated_sentence
                    generation_out_file.write(json.dumps(json_obj) + '\n')
        with open(truncated_cheat_path + polish_file_name, 'w', encoding='utf-8') as polish_out_file:
            for line in polish_file:
                json_obj = json.loads(line)
                truncated_sentences = truncate_content(content=json_obj['abstract'], truncate_length=truncate_length, drop_length=drop_length)
                for truncated_sentence in truncated_sentences:
                    json_obj['abstract'] = truncated_sentence
                    polish_out_file.write(json.dumps(json_obj) + '\n')
        with open(truncated_cheat_path + init_file_name, 'w', encoding='utf-8') as init_out_file:
            for line in init_file:
                json_obj = json.loads(line)
                truncated_sentences = truncate_content(content=json_obj['abstract'], truncate_length=truncate_length, drop_length=drop_length)
                for truncated_sentence in truncated_sentences:
                    json_obj['abstract'] = truncated_sentence
                    init_out_file.write(json.dumps(json_obj) + '\n')

def truncate_content(content: str, truncate_length, drop_length):
    words = content.split(' ')
    truncated_sentences = []
    for i in range(0, int(len(words) / truncate_length) - 1):
        sub_words = words[i * truncate_length: (i + 1) * truncate_length]
        truncated_sentences.append(" ".join(sub_words))
    if len(content) % truncate_length >= drop_length:
        sub_words = words[-1 * (len(content) % truncate_length):]
        truncated_sentences.append(" ".join(sub_words))
    else:
        if len(truncated_sentences) == 0:
            truncated_sentences.append(content)
    return truncated_sentences


if __name__ == '__main__':
    truncate_CHEAT('./test_data/CHEAT/', './test_data_truncated/CHEAT/')
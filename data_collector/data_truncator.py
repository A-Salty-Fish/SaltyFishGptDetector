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



def truncate_ghostbuster(raw_ghostbuster_path, truncated_ghostbuster_path, truncate_length=256, drop_length = 128):
    essay_claude_file_name = 'essay_claude.txt'
    essay_gpt_file_name = 'essay_gpt.txt'
    essay_gpt_semantic_file_name = 'essay_gpt_semantic.txt'
    essay_gpt_writing_file_name = 'essay_gpt_writing.txt'
    essay_human_file_name = 'essay_human.txt'
    with open(raw_ghostbuster_path + essay_claude_file_name, 'r', encoding='utf-8') as essay_claude_file, open(
        raw_ghostbuster_path + essay_gpt_file_name, 'r', encoding='utf-8') as essay_gpt_file, open(
        raw_ghostbuster_path + essay_gpt_semantic_file_name, 'r', encoding='utf-8') as essay_gpt_semantic_file, open(
        raw_ghostbuster_path + essay_gpt_writing_file_name, 'r', encoding='utf-8') as essay_gpt_writing_file, open(
        raw_ghostbuster_path + essay_human_file_name, 'r', encoding='utf-8') as essay_human_file:
            with open(truncated_ghostbuster_path + essay_claude_file_name, 'w', encoding='utf-8') as essay_claude_out_file:
                for line in essay_claude_file:
                    truncated_sentences = truncate_content(content=line, truncate_length=truncate_length, drop_length=drop_length)
                    for truncated_sentence in truncated_sentences:
                        essay_claude_out_file.write(truncated_sentence.replace('\n','') + '\n')
            with open(truncated_ghostbuster_path + essay_gpt_file_name, 'w', encoding='utf-8') as essay_gpt_out_file:
                for line in essay_gpt_file:
                    truncated_sentences = truncate_content(content=line, truncate_length=truncate_length, drop_length=drop_length)
                    for truncated_sentence in truncated_sentences:
                        essay_gpt_out_file.write(truncated_sentence.replace('\n','') + '\n')
            with open(truncated_ghostbuster_path + essay_gpt_semantic_file_name, 'w', encoding='utf-8') as essay_gpt_semantic_out_file:
                for line in essay_gpt_semantic_file:
                    truncated_sentences = truncate_content(content=line, truncate_length=truncate_length, drop_length=drop_length)
                    for truncated_sentence in truncated_sentences:
                        essay_gpt_semantic_out_file.write(truncated_sentence.replace('\n','') + '\n')
            with open(truncated_ghostbuster_path + essay_gpt_writing_file_name, 'w', encoding='utf-8') as essay_gpt_writing_out_file:
                for line in essay_gpt_writing_file:
                    truncated_sentences = truncate_content(content=line, truncate_length=truncate_length, drop_length=drop_length)
                    for truncated_sentence in truncated_sentences:
                        essay_gpt_writing_out_file.write(truncated_sentence.replace('\n','') + '\n')
            with open(truncated_ghostbuster_path + essay_human_file_name, 'w', encoding='utf-8') as essay_human_out_file:
                for line in essay_human_file:
                    truncated_sentences = truncate_content(content=line, truncate_length=truncate_length, drop_length=drop_length)
                    for truncated_sentence in truncated_sentences:
                        essay_human_out_file.write(truncated_sentence.replace('\n','') + '\n')


def truncate_hc3_english(raw_hc3_english_path, truncated_hc3_english_path, truncate_length=256, drop_length = 128):
    finance_file_name = 'finance.jsonl'
    medicine_file_name = 'medicine.jsonl'
    open_qa_file_name = 'open_qa.jsonl'
    wiki_csai_file_name = 'wiki_csai.jsonl'
    with open(raw_hc3_english_path + finance_file_name, 'r', encoding='utf-8') as finance_file, open(
            raw_hc3_english_path + medicine_file_name, 'r', encoding='utf-8') as medicine_file, open(
            raw_hc3_english_path + open_qa_file_name, 'r', encoding='utf-8') as open_qa_file, open(
            raw_hc3_english_path + wiki_csai_file_name, 'r', encoding='utf-8') as wiki_csai_file:
        with open(truncated_hc3_english_path + finance_file_name, 'w', encoding='utf-8') as finance_out_file:
            for line in finance_file:
                json_obj = json.loads(line)
                human_truncated_sentences = truncate_content(content=json_obj['human_answers'][0],
                                                             truncate_length=truncate_length,
                                                             drop_length=drop_length)
                json_obj['human_answers'] = human_truncated_sentences
                chatgpt_truncated_sentences = truncate_content(content=json_obj['chatgpt_answers'][0],
                                                               truncate_length=truncate_length,
                                                               drop_length=drop_length)
                json_obj['chatgpt_answers'] = chatgpt_truncated_sentences
                finance_out_file.write(json.dumps(json_obj) + '\n')
        with open(truncated_hc3_english_path + medicine_file_name, 'w', encoding='utf-8') as medicine_out_file:
            for line in medicine_file:
                json_obj = json.loads(line)
                human_truncated_sentences = truncate_content(content=json_obj['human_answers'][0],
                                                             truncate_length=truncate_length,
                                                             drop_length=drop_length)
                json_obj['human_answers'] = human_truncated_sentences
                chatgpt_truncated_sentences = truncate_content(content=json_obj['chatgpt_answers'][0],
                                                               truncate_length=truncate_length,
                                                               drop_length=drop_length)
                json_obj['chatgpt_answers'] = chatgpt_truncated_sentences
                medicine_out_file.write(json.dumps(json_obj) + '\n')
        with open(truncated_hc3_english_path + open_qa_file_name, 'w', encoding='utf-8') as open_qa_out_file:
            for line in open_qa_file:
                json_obj = json.loads(line)
                human_truncated_sentences = truncate_content(content=json_obj['human_answers'][0],
                                                             truncate_length=truncate_length,
                                                             drop_length=drop_length)
                json_obj['human_answers'] = human_truncated_sentences
                chatgpt_truncated_sentences = truncate_content(content=json_obj['chatgpt_answers'][0],
                                                               truncate_length=truncate_length,
                                                               drop_length=drop_length)
                json_obj['chatgpt_answers'] = chatgpt_truncated_sentences
                open_qa_out_file.write(json.dumps(json_obj) + '\n')
        with open(truncated_hc3_english_path + wiki_csai_file_name, 'w', encoding='utf-8') as wiki_csai_out_file:
            for line in wiki_csai_file:
                json_obj = json.loads(line)
                human_truncated_sentences = truncate_content(content=json_obj['human_answers'][0],
                                                             truncate_length=truncate_length,
                                                             drop_length=drop_length)
                json_obj['human_answers'] = human_truncated_sentences
                chatgpt_truncated_sentences = truncate_content(content=json_obj['chatgpt_answers'][0],
                                                               truncate_length=truncate_length,
                                                               drop_length=drop_length)
                json_obj['chatgpt_answers'] = chatgpt_truncated_sentences
                wiki_csai_out_file.write(json.dumps(json_obj) + '\n')


def truncate_hc3_plus_english(raw_hc3_plus_english_path, truncated_hc3_plus_english_path, truncate_length=256, drop_length = 128):
    test_qa = 'test_hc3_QA.jsonl'
    test_si = 'test_hc3_si.jsonl'
    val_qa = 'val_hc3_QA.jsonl'
    val_si = 'val_hc3_si.jsonl'
    with open(raw_hc3_plus_english_path + test_qa, 'r', encoding='utf-8') as test_qa_file, open(
            raw_hc3_plus_english_path + test_si, 'r', encoding='utf-8') as test_si_file, open(
        raw_hc3_plus_english_path + val_qa, 'r', encoding='utf-8') as val_qa_file, open(
        raw_hc3_plus_english_path + val_si, 'r', encoding='utf-8') as val_si_file:
        with open(truncated_hc3_plus_english_path + test_qa, 'w', encoding='utf-8') as test_qa_out_file:
            for line in test_qa_file:
                json_obj = json.loads(line)
                truncated_sentences = truncate_content(content=json_obj['text'],
                                                             truncate_length=truncate_length,
                                                             drop_length=drop_length)
                for truncated_sentence in truncated_sentences:
                    json_obj['text'] = truncated_sentence
                    test_qa_out_file.write(json.dumps(json_obj) + '\n')
        with open(truncated_hc3_plus_english_path + test_si, 'w', encoding='utf-8') as test_si_out_file:
            for line in test_si_file:
                json_obj = json.loads(line)
                truncated_sentences = truncate_content(content=json_obj['text'],
                                                             truncate_length=truncate_length,
                                                             drop_length=drop_length)
                for truncated_sentence in truncated_sentences:
                    json_obj['text'] = truncated_sentence
                    test_si_out_file.write(json.dumps(json_obj) + '\n')
        with open(truncated_hc3_plus_english_path + val_qa, 'w', encoding='utf-8') as val_qa_out_file:
            for line in val_qa_file:
                json_obj = json.loads(line)
                truncated_sentences = truncate_content(content=json_obj['text'],
                                                             truncate_length=truncate_length,
                                                             drop_length=drop_length)
                for truncated_sentence in truncated_sentences:
                    json_obj['text'] = truncated_sentence
                    val_qa_out_file.write(json.dumps(json_obj) + '\n')
        with open(truncated_hc3_plus_english_path + val_si, 'w', encoding='utf-8') as val_si_out_file:
            for line in val_si_file:
                json_obj = json.loads(line)
                truncated_sentences = truncate_content(content=json_obj['text'],
                                                             truncate_length=truncate_length,
                                                             drop_length=drop_length)
                for truncated_sentence in truncated_sentences:
                    json_obj['text'] = truncated_sentence
                    val_si_out_file.write(json.dumps(json_obj) + '\n')


def output_list(list_in: list):
    print(
        f'min:{min(list_in)}, max: {max(list_in)}, mean: {sum(list_in) / len(list_in)}')

def statistic_cheat_length():
    fusion_file_name = 'ieee-chatgpt-fusion.jsonl'
    generation_file_name = 'ieee-chatgpt-generation.jsonl'
    polish_file_name = 'ieee-chatgpt-polish.jsonl'
    init_file_name = 'ieee-init.jsonl'
    raw_cheat_path = './test_data/CHEAT/'
    with open(raw_cheat_path + fusion_file_name, 'r', encoding='utf-8') as fusion_file, open(
            raw_cheat_path + generation_file_name, 'r', encoding='utf-8') as generation_file, open(
            raw_cheat_path + polish_file_name, 'r', encoding='utf-8') as polish_file, open(raw_cheat_path + init_file_name,
                                                                                           'r',encoding='utf-8') as init_file:
        cheat_fusion_lengths = []
        for line in fusion_file:
            json_obj = json.loads(line)
            content = json_obj['abstract']
            cheat_fusion_lengths.append(len(content.split(' ')))
        print("fusion")
        output_list(cheat_fusion_lengths)
        cheat_generation_lengths = []
        for line in generation_file:
            json_obj = json.loads(line)
            content = json_obj['abstract']
            cheat_generation_lengths.append(len(content.split(' ')))
        print("generation")
        output_list(cheat_generation_lengths)
        cheat_polish_lengths = []
        for line in polish_file:
            json_obj = json.loads(line)
            content = json_obj['abstract']
            cheat_polish_lengths.append(len(content.split(' ')))
        print("polish")
        output_list(cheat_polish_lengths)
        cheat_init_lengths = []
        for line in init_file:
            json_obj = json.loads(line)
            content = json_obj['abstract']
            cheat_init_lengths.append(len(content.split(' ')))
        print("polish")
        output_list(cheat_init_lengths)



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
    truncate_CHEAT('./test_data/CHEAT/', './test_data_truncated/CHEAT/', 128, 64)
    truncate_ghostbuster('./test_data/ghostbuster/', './test_data_truncated/ghostbuster/', 128, 64)
    truncate_hc3_english('./test_data/hc3_english/', './test_data_truncated/hc3_english/', 128, 64)
    truncate_hc3_plus_english('./test_data/hc3_plus_english/', './test_data_truncated/hc3_plus_english/', 128, 64)
    # statistic_cheat_length()
    # text = 'Large language models (LLMs) have demonstrated remarkable capability to generate fluent responses to a wide variety of user queries, but this has also resulted in concerns regarding the potential misuse of such texts in journalism, educational, and academic context. In this work, we aim to develop automatic systems to identify machine-generated text and to detect potential misuse. We first introduce a large-scale benchmark M4, which is multi-generator, multi-domain, and multi-lingual corpus for machine-generated text detection. Using the dataset, we experiment with a number of methods and we show that it is challenging for detectors to generalize well on unseen examples if they are either from different domains or are generated by different large language models. In such cases, detectors tend to misclassify machine-generated text as human-written. These results show that the problem is far from solved and there is a lot of room for improvement. We believe that our dataset M4, which covers different generators, domains and languages, will enable future research towards more robust approaches for this pressing societal problem. The M4 dataset is available at this https URL.Large language models (LLMs) have demonstrated remarkable capability to generate fluent responses to a wide variety of user queries, but this has also resulted in concerns regarding the potential misuse of such texts in journalism, educational, and academic context. In this work, we aim to develop automatic systems to identify machine-generated text and to detect potential misuse. We first introduce a large-scale benchmark M4, which is multi-generator, multi-domain, and multi-lingual corpus for machine-generated text detection. Using the dataset, we experiment with a number of methods and we show that it is challenging for detectors to generalize well on unseen examples if they are either from different domains or are generated by different large language models. In such cases, detectors tend to misclassify machine-generated text as human-written. These results show that the problem is far from solved and there is a lot of room for improvement. We believe that our dataset M4, which covers different generators, domains and languages, will enable future research towards more robust approaches for this pressing societal problem. The M4 dataset is available at this https URL. Large language models (LLMs) have demonstrated remarkable capability to generate fluent responses to a wide variety of user queries, but this has also resulted in concerns regarding the potential misuse of such texts in journalism, educational, and academic context. In this work, we aim to develop automatic systems to identify machine-generated text and to detect potential misuse. We first introduce a large-scale benchmark M4, which is multi-generator, multi-domain, and multi-lingual corpus for machine-generated text detection. Using the dataset, we experiment with a number of methods and we show that it is challenging for detectors to generalize well on unseen examples if they are either from different domains or are generated by different large language models. In such cases, detectors tend to misclassify machine-generated text as human-written. These results show that the problem is far from solved and there is a lot of room for improvement. We believe that our dataset M4, which covers different generators, domains and languages, will enable future research towards more robust approaches for this pressing societal problem.'
    # print(len(text.split(' ')))
    # print(len(truncate_content(text, 256, 128)))
    pass
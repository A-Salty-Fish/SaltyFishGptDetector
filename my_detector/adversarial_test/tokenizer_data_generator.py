import json
import random

from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained('microsoft/deberta-v3-small', model_max_length=2048)

from transformers import pipeline

fill_mask_pipe = pipeline("fill-mask", model="microsoft/deberta-v3-small")


def convert_text_to_tokens(text):
    tokenized_input = tokenizer(text.split(' '), is_split_into_words=True)
    tokens = tokenizer.convert_ids_to_tokens(tokenized_input["input_ids"])
    return tokens


def fill_mask(masked_text, pre_word=''):
    if pre_word[0] == '▁':
        pre_word = pre_word[1:]
    results = fill_mask_pipe(masked_text)
    for result in results:
        if result['token_str'] != pre_word:
            return result['token_str']
    return pre_word


def tokens_to_sentense(tokens):
    result = ""
    for token in tokens:
        if token == '[CLS]' or token == '[SEP]':
            continue
        if token[0] == '▁':
            token = token[1:]
        result += token + " "
    return result


def random_fill_token(text, fill_num=1, max_num=20):
    result = []
    tokens = convert_text_to_tokens(text)
    # 第一个和最后一个不考虑
    able_indexes = [i for i in range(0, len(tokens) - 1)]
    for i in range(0, max_num):
        cur_tokens = convert_text_to_tokens(text)
        json_obj = {}
        random_indexes = random.sample(able_indexes, fill_num)
        # token的分类标识
        token_labels = [0 for _ in range(0, len(tokens))]
        json_obj['token_labels'] = token_labels
        json_obj['row_text'] = text
        json_row_tokens = [x for x in cur_tokens]
        json_obj['row_tokens'] = json_row_tokens
        for select_index in random_indexes:
            # 标记标识位
            token_labels[select_index] = fill_num
            pre_word = cur_tokens[select_index]
            cur_tokens[select_index] = '[MASK]'
            sentense = tokens_to_sentense(cur_tokens)
            fill_token = fill_mask(sentense, pre_word)
            cur_tokens[select_index] = fill_token
        json_obj['result_text'] = tokens_to_sentense(cur_tokens)
        result.append(json_obj)
    return result


def tokenize_hc3_data():
    results = []
    i = 0
    hc3_file_names = ['finance', 'medicine', 'open_qa', 'wiki_csai']
    for hc3_file_name in hc3_file_names:
        with open('./data/' + hc3_file_name + '.mix.jsonl', 'r', encoding='utf-8') as f:
            for line in f:
                i += 1
                print('process : %s' % (str(i)), end='\r')
                json_obj = json.loads(line)
                # todo ai only
                human_content = json_obj['human'].replace('\n', '')
                try:
                    fill_results = random_fill_token(human_content, 1, 5)
                    for fill_result in fill_results:
                        results.append({
                            'label': 0,
                            'content': fill_result
                        })
                    results.append({
                        'label': 0,
                        'content': human_content
                    })
                except Exception as e:
                    print(e)

    with open('./hc3_mix_token_fill_1_result', 'w', encoding='utf-8') as f_out:
        f_out.write(json.dumps(results, ensure_ascii=False))


def tokenize_wiki_qa_mix_data():
    results = []
    i = 0
    human_contents = []
    with open('./data/wiki_qa_mix.jsonl', 'r', encoding='utf-8') as f:
        for line in f:

            json_obj = json.loads(line)
            # todo ai only
            human_content = json_obj['human'].replace('\n', '')
            human_contents.append(human_content)
    for human_content in human_contents:
        try:
            i += 1
            print('total : %s' % (str(i/len(human_contents))), end='\r')
            fill_results = random_fill_token(human_content, 1, 5)
            for fill_result in fill_results:
                results.append({
                    'label': 0,
                    'content': fill_result
                })
        except Exception as e:
            print(e)

    with open('./wiki_qa_mix_token_fill_1_result', 'w', encoding='utf-8') as f_out:
        f_out.write(json.dumps(results, ensure_ascii=False))


def fix_tokenized_wiki_qa_mix_result():
    with open('./data/wiki_qa_mix_token_fill_1', 'r' ,encoding='utf-8') as in_f:
        with open('./data/wiki_qa_mix_token_fill_1_result', 'w', encoding='utf-8') as out_f:
            json_arr = json.load(in_f)
            results = []
            for json_obj in json_arr:
                try:
                    results.append(
                        {
                            'label': 0,
                            'content': json_obj['content']['result_text']
                        }
                    )
                except Exception as e:
                    print(e)
            out_f.write(json.dumps(results))



def output_mask_results(input_name, output_name, total_num=10000):
    print("begin 1")
    i = 0
    with open('./data/' + input_name, 'r', encoding='utf-8') as f, open('./' + output_name + '_1.jsonl' , 'w', encoding='utf-8') as out_f:
        for line in f:
            i+=1
            print('test process : %s' % (str(i)), end='\r')
            try:
                json_obj = json.loads(line)
                res_objs = random_fill_token(json_obj['content'], 1, 5)
                for res_obj in res_objs:
                    res_obj['label'] = json_obj['label']
                    out_f.write(json.dumps(res_obj, ensure_ascii=False) + '\n')
            except Exception as e:
                print(e)
    i = 0
    print("begin 2")
    with open('./data/' + input_name, 'r', encoding='utf-8') as f, open('./' + output_name + '_2.jsonl' , 'w', encoding='utf-8') as out_f:
        for line in f:
            i+=1
            print('test process : %s' % (str(i)), end='\r')
            try:
                json_obj = json.loads(line)
                res_objs = random_fill_token(json_obj['content'], 2, 5)
                for res_obj in res_objs:
                    res_obj['label'] = json_obj['label']
                    out_f.write(json.dumps(res_obj, ensure_ascii=False) + '\n')
            except Exception as e:
                print(e)
    i = 0
    print("begin 3")
    with open('./data/' + input_name, 'r', encoding='utf-8') as f, open('./' + output_name + '_3.jsonl' , 'w', encoding='utf-8') as out_f:
        for line in f:
            i+=1
            print('test process : %s' % (str(i)), end='\r')
            try:
                json_obj = json.loads(line)
                res_objs = random_fill_token(json_obj['content'], 3, 5)
                for res_obj in res_objs:
                    res_obj['label'] = json_obj['label']
                    out_f.write(json.dumps(res_obj, ensure_ascii=False) + '\n')
            except Exception as e:
                print(e)


if __name__ == "__main__":
    # masked_text = 'Paris is the [MASK] of France. It [MASK] so good.'
    # print(fill_mask_pipe(masked_text))
    # test_long_sen = 'An overview of text line segmentation methods developed within different projects is  presented in Table 1. The achieved taxonomy consists in six major categories. They are listed  as: projection-based, smearing, grouping, Hough-based, repulsive-attractive network and  stochastic methods. Most of these methods are able to face some image degradations and  writing irregularities specific to historical documents, as shown in the last column of Table 1.  Projection, smearing and Hough-based methods, classically adapted to straight lines and  easier to implement, had to be completed and enriched by local considerations (piecewise  projections, clustering in Hough space, use of a moving window, ascender and descender  skipping), so as to solve some problems including: line proximity, overlapping or even  touching strokes, fluctuating close lines, shape fragmentation occurrences. The stochastic  method (achieved by the Viterbi decision algorithm) is conceptually more robust, but its  implementation requires great care, particularly the initialization phase. As a matter of fact,  text-line images are initially divided into mxn grids (each cell being a node), where the values  of the critical parameters m and n are to be determined according to the estimated average  stroke width in the images. Representing a text line by one or more baselines (RA method,  minima point grouping) must be completed by labeling those pixels not connected to, or  between the extracted baselines. The recurrent nature of the repulsive-attractive method may  induce cascading detecting errors following a unique false or bad line extraction.  Projection and Hough-based methods are suitable for clearly separated lines. Projection-based  methods can cope with few overlapping or touching components, as long text lines smooth  both noise and overlapping effects. Even in more critical cases, classifying the set of blocks  into "one line width" blocks and "several lines width" blocks allows the segmentation process  to get statistical measures so as to segment more surely the "several lines width" blocks. As a  result, the linear separator path may cross overlapping components. However, more accurate  segmentation of the overlapping components can be performed after getting the global or  piecewise straight separator, by looking closely at the so crossed strokes. The stochastic  method naturally avoids crossing overlapping components (if they are not too close): the  resulting non linear paths turn around obstacles. When lines are very close, grouping methods  encounter a lot of conflicting configurations. A wrong decision in an early stage of the  grouping results in errors or incomplete alignments. In case of touching components, making  an accurate segmentation requires additional knowledge (compiled in a dictionary of possible  configurations or represented by logical or fuzzy rules).'
    # print(random_fill_token(test_long_sen, fill_num=1, max_num=5))
    # output_mask_results('medicine_medicine.jsonl.split.acc', 'medicine_masked')
    # tokenize_hc3_data()
    # tokenize_wiki_qa_mix_data()
    fix_tokenized_wiki_qa_mix_result()
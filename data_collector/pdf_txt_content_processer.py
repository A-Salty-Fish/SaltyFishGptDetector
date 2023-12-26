import json
import os


# 把原始的文本先简单分段
def convert_text_to_paragraphs(text_file):
    result = []
    with open(text_file, 'r', encoding='utf-8') as f:
        cur_para = ""
        for line in f:
            line = preprocess_line(line)
            if line == '\n':
                result.append(cur_para)
                cur_para = ""
            else:
                cur_para += line
        result.append(cur_para)
    return result


# 预处理文本，有一些奇怪的数据需要处理
def preprocess_line(line: str):
    line = line.replace('“', '"')
    line = line.replace('”', '"')
    line = line.replace('（', '(')
    line = line.replace('）', ')')
    line = line.replace('，', ',')
    line = line.replace('。', '.')
    line = line.replace('ﬁ', 'fi')
    line = line.replace('ﬂ', 'n')
    line = line.replace('’', '\'')
    # line = line.replace('\\n', '')
    # line = line.replace('\n', '')
    while line.find("  ") != -1:
        line = line.replace("  ", " ")
    while line.find("##") != -1:
        line = line.replace("##", "#")
    # line = line.strip()
    return line

# 无用行判断
def is_useless_line(line):
    line = preprocess_line(line)
    if len(line) < 2:
        return True
    if line == '\n':
        return True
    if line.find('. .') != -1:
        return True
    if line.find('\\u') != -1:
        return True
    if line.find('::') != -1:
        return True
    if line.find(')(') != -1:
        return True
    invalid_characters = ['@', '{', '}', '<', '>']
    for invalid_character in invalid_characters:
        if line.find(invalid_character) != -1:
            return True
    for c in line:
        if not c.isdigit() and not c.isascii():
            if c != ' ' and c != '\n' and c != ',' and c != '.' \
                    and c != '-' and c != '—' and c != '"' and c != '\'' and c != ':' and c != ';' \
                    and c != '(' and c != ')' and c != '•':
                return True
    return False


# 将简单的段落合并成连续的段落，不考虑有无效行的段落
def merge_paragraphs(paragraphs):
    result = []
    for paragraph in paragraphs:
        res_para = ""
        lines = paragraph.split('\n')
        useless_flag = False
        for line in lines:
            if len(line) <= 1:
                continue
            if is_useless_line(line):
                useless_flag = True
                break
            if line[-1] == '-':
                res_para += line[0:-1]
            else:
                res_para += " " + line
        if useless_flag:
            continue
        if len(res_para) < 2:
            continue
        result.append(res_para)
    return result


# 最终的段落生成，除去长度过短的段落
def generate_final_paragraphs(paragraphs, min_len=50, max_len=128):
    result = []
    for paragraph in paragraphs:
        if len(paragraph.split(" ")) < min_len:
            continue
        if len(paragraph.split(" ")) > max_len:
            sentenses = [x for x in paragraph.split('.') if len(x) > 1]
            for sentense in sentenses:
                if len(sentense.split(" ")) < min_len:
                    continue
                if len(sentense.split(" ")) > max_len:
                    continue
                else:
                    sentense = sentense.strip()
                    if len(sentense) < 2:
                        continue
                    result.append(sentense)
                    continue
            else:
                continue
        paragraph = paragraph.strip()
        if len(paragraph) < 2:
            continue
        result.append(paragraph)
    result = [post_process_paragraph(x) for x in result]
    return result


def post_process_paragraph(paragraph):
    paragraph = paragraph.replace('\n', '')
    paragraph = paragraph.replace('\\n', '')
    return paragraph


# 将txt文件处理成段落
def pdf_txt_content_process(pdf_id, min_len=50, max_len=128):
    file_path = f'./row_data/arxiv_text/{pdf_id}.txt'
    init_paras = convert_text_to_paragraphs(file_path)
    merged_paragraphs = (merge_paragraphs(init_paras))
    final_paragraphs = generate_final_paragraphs(merged_paragraphs, min_len=min_len, max_len=max_len)
    return final_paragraphs


def test_is_useless(c: str):
    if not c.isdigit() and not c.isascii():
        if c != ' ' and c != '\n' and c != ',' and c != '.' \
            and c != '-' and c != '—' and c != '"' and c != '\'' and c != ':' and c != ';' \
            and c != '(' and c != ')' and c != '•':
            return True
    return False


if __name__ == '__main__':
    # file_path = './row_data/arxiv_text/0704.2963.txt'
    # init_paras = convert_text_to_paragraphs(file_path)
    #
    # merged_paragraphs = (merge_paragraphs(init_paras))
    # for para in merged_paragraphs:
    #     print(para)

    # test_long_sen = 'An overview of text line segmentation methods developed within different projects is  presented in Table 1. The achieved taxonomy consists in six major categories. They are listed  as: projection-based, smearing, grouping, Hough-based, repulsive-attractive network and  stochastic methods. Most of these methods are able to face some image degradations and  writing irregularities specific to historical documents, as shown in the last column of Table 1.  Projection, smearing and Hough-based methods, classically adapted to straight lines and  easier to implement, had to be completed and enriched by local considerations (piecewise  projections, clustering in Hough space, use of a moving window, ascender and descender  skipping), so as to solve some problems including: line proximity, overlapping or even  touching strokes, fluctuating close lines, shape fragmentation occurrences. The stochastic  method (achieved by the Viterbi decision algorithm) is conceptually more robust, but its  implementation requires great care, particularly the initialization phase. As a matter of fact,  text-line images are initially divided into mxn grids (each cell being a node), where the values  of the critical parameters m and n are to be determined according to the estimated average  stroke width in the images. Representing a text line by one or more baselines (RA method,  minima point grouping) must be completed by labeling those pixels not connected to, or  between the extracted baselines. The recurrent nature of the repulsive-attractive method may  induce cascading detecting errors following a unique false or bad line extraction.  Projection and Hough-based methods are suitable for clearly separated lines. Projection-based  methods can cope with few overlapping or touching components, as long text lines smooth  both noise and overlapping effects. Even in more critical cases, classifying the set of blocks  into "one line width" blocks and "several lines width" blocks allows the segmentation process  to get statistical measures so as to segment more surely the "several lines width" blocks. As a  result, the linear separator path may cross overlapping components. However, more accurate  segmentation of the overlapping components can be performed after getting the global or  piecewise straight separator, by looking closely at the so crossed strokes. The stochastic  method naturally avoids crossing overlapping components (if they are not too close): the  resulting non linear paths turn around obstacles. When lines are very close, grouping methods  encounter a lot of conflicting configurations. A wrong decision in an early stage of the  grouping results in errors or incomplete alignments. In case of touching components, making  an accurate segmentation requires additional knowledge (compiled in a dictionary of possible  configurations or represented by logical or fuzzy rules).'
    #
    # res = (generate_final_paragraphs([test_long_sen]))
    # print(len(res))
    # print(res)
    # print('ф'.isascii())
    # print(test_is_useless('ф'))
    # print(test_is_useless('a'))
    # print(is_useless_line('Аннотация: Классификация метрик и алгоритмов поиска семантически близких слов '))
    for year in range(7, 18):
        print("begin year:" + str(year))
        with open(f'./row_data/arxiv_paras/{year}.jsonl', 'w', encoding='utf-8') as out_file:
            for file in os.listdir('./row_data/arxiv_text'):
                if file == 'empty':
                    continue
                if int(file[0:2]) == year:
                    try:
                        pdf_id = file.replace('.txt', '')
                        pdf_paras = pdf_txt_content_process(pdf_id)
                        for pdf_para in pdf_paras:
                            json_obj = {
                                'pdf_id': pdf_id,
                                'content': pdf_para
                            }
                            out_file.write(json.dumps(json_obj, ensure_ascii=False) + '\n')
                        print(file)
                        pass
                    except Exception as e:
                        print(e)
        print("end year:" + str(year))
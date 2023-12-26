
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
    return line

# 无用行判断
def is_useless_line(line):
    line = preprocess_line(line)
    if len(line) < 2:
        return True
    if line == '\n':
        return True
    invalid_characters = ['@', '{', '}', '<', '>']
    for invalid_character in invalid_characters:
        if line.find(invalid_character) != -1:
            return True
    for c in line:
        if not c.isalpha() and not c.isalnum() and not c.isdigit() \
                and c != ' ' and c != '\n' and c != ',' and c != '.'\
                and c != '-' and c != '—' and c != '"' and c != '\'' and c != ':' and c != ';'\
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
def generate_final_paragraphs(paragraphs, min_len=20, max_len=128):
    result = []
    for paragraph in paragraphs:
        if len(paragraph.split(" ")) < min_len:
            continue
        if len(paragraph.split(" ")) > max_len:
            sentenses = [x for x in paragraph.split('.') if len(x) > 1]
            if len(sentenses) >= 2:
                recursive_sentenses = []
                for sentense in sentenses:
                    if sentense[-1] == '.':
                        recursive_sentenses.append(sentense)
                    else:
                        sentense = sentense + '.'
                        recursive_sentenses.append(sentense)
                result += (generate_final_paragraphs(recursive_sentenses, min_len=min_len, max_len=max_len))
            else:
                continue
        result.append(paragraph)
    return result


# 将txt文件处理成段落
def pdf_txt_content_process(pdf_id, min_len=50, max_len=128):
    file_path = f'./row_data/arxiv_text/{pdf_id}.txt'
    init_paras = convert_text_to_paragraphs(file_path)
    merged_paragraphs = (merge_paragraphs(init_paras))
    final_paragraphs = generate_final_paragraphs(merged_paragraphs, min_len=min_len, max_len=max_len)
    return final_paragraphs



if __name__ == '__main__':
    final_paragraphs = pdf_txt_content_process('0704.1675')
    # print(len(final_paragraphs))
    print(len(final_paragraphs))
    for x in final_paragraphs:
        print(x)

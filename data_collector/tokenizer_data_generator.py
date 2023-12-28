import json

from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained('microsoft/deberta-v3-small', model_max_length=2048)

from transformers import pipeline

fill_mask_pipe = pipeline("fill-mask", model="microsoft/deberta-v3-small")

def convert_text_to_tokens(tokenizer, text):
    tokenized_input = tokenizer(text.split(' '), is_split_into_words=True)
    tokens = tokenizer.convert_ids_to_tokens(tokenized_input["input_ids"])
    return tokens

def fill_mask(masked_text, pre_word = ''):
    results = fill_mask_pipe(masked_text)
    for result in results:
        if result['token_str'] != pre_word:
            return result['token_str']
    return pre_word

def convert_hc3_english_to_tokens():
    with open('./test_data/hc3_english/finance.jsonl', 'r', encoding='utf-8') as finance_file, \
            open('./test_data/hc3_english/medicine.jsonl', 'r', encoding='utf-8') as medicine_file, \
            open('./test_data/hc3_english/wiki_csai.jsonl', 'r', encoding='utf-8') as wiki_file:
        with open('./finance_tokens.jsonl', 'a', encoding='utf-8') as finance_out_file:
            for line in finance_file:
                try:
                    json_obj = json.loads(line)
                    human_obj = {}
                    gpt_obj = {}
                    human_obj['row_label'] = 0
                    gpt_obj['row_label'] = 1
                    human_obj['content'] = json_obj['human_answers'][0]
                    human_obj['tokens'] = convert_text_to_tokens(tokenizer, human_obj['content'])
                    gpt_obj['content'] = json_obj['chatgpt_answers'][0]
                    gpt_obj['tokens'] = convert_text_to_tokens(tokenizer, gpt_obj['content'])
                    finance_out_file.write(json.dumps(human_obj) + '\n')
                    finance_out_file.write(json.dumps(gpt_obj) + '\n')
                except Exception as e:
                    print(e)
        with open('./medicine_token.jsonl', 'a', encoding='utf-8') as medicine_out_file:
            for line in medicine_file:
                try:
                    json_obj = json.loads(line)
                    human_obj = {}
                    gpt_obj = {}
                    human_obj['row_label'] = 0
                    gpt_obj['row_label'] = 1
                    human_obj['content'] = json_obj['human_answers'][0]
                    human_obj['tokens'] = convert_text_to_tokens(tokenizer, human_obj['content'])
                    gpt_obj['content'] = json_obj['chatgpt_answers'][0]
                    gpt_obj['tokens'] = convert_text_to_tokens(tokenizer, gpt_obj['content'])
                    medicine_out_file.write(json.dumps(human_obj) + '\n')
                    medicine_out_file.write(json.dumps(gpt_obj) + '\n')
                except Exception as e:
                    print(e)
        with open('./wiki_token.jsonl', 'a', encoding='utf-8') as wiki_out_file:
            for line in wiki_file:
                try:
                    json_obj = json.loads(line)
                    human_obj = {}
                    gpt_obj = {}
                    human_obj['row_label'] = 0
                    gpt_obj['row_label'] = 1
                    human_obj['content'] = json_obj['human_answers'][0]
                    human_obj['tokens'] = convert_text_to_tokens(tokenizer, human_obj['content'])
                    gpt_obj['content'] = json_obj['chatgpt_answers'][0]
                    gpt_obj['tokens'] = convert_text_to_tokens(tokenizer, gpt_obj['content'])
                    wiki_out_file.write(json.dumps(human_obj) + '\n')
                    wiki_out_file.write(json.dumps(gpt_obj) + '\n')
                except Exception as e:
                    print(e)


if __name__ == "__main__":
    # masked_text = 'Paris is the [MASK] of France.'
    # print(fill_mask(masked_text))
    convert_hc3_english_to_tokens()


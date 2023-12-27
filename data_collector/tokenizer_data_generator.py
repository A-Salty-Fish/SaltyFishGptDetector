from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained('microsoft/deberta-v3-small', model_max_length=512)

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


if __name__ == "__main__":
    masked_text = 'Paris is the [MASK] of France.'
    print(fill_mask(masked_text))


# 各个算法比较

import argparse
from functools import partial

import detect_gpt
import gltr
import hc3_ling
import hc3_single

support_methods = [
    'detect_gpt',
    'gltr',
    'hc3_ling',
    'hc3_single',
    'intrinsic-dim',
    'llmdet',
    'openai-roberta-base',
    'openai-roberta-large',
    'radar-vicuna'
]

support_datasets = [
    'CHEAT',
    'ghostbuster',
    'hc3_english',
    'hc3_plus_english',
    'm4'
]

def get_classifier(method):
    if method == 'detect_gpt':
        model = detect_gpt.init_model()

        def classify(text):
            return detect_gpt.classify_is_human(model, text=text)

        return classify

    if method == 'gltr':
        model = gltr.LM()

        def classify(text):
            return gltr.classify_is_human(model, text=text)

        return classify

    if method == 'hc3_ling':

        model = None

        def classify(text):
            return hc3_ling.classify_is_human(text=text)

        return classify

    if method == 'hc3_single':
        model = hc3_single.init_classifier()
        def classify(text):
            return hc3_single.classify_is_human(model, text=text)

        return classify

    if method == 'intrinsic-dim':
        model = hc3_single.init_classifier()
        def classify(text):
            return hc3_single.classify_is_human(model, text=text)

        return classify

    print("None Method")

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--method', type=str, default='gltr', help='baseline method,')
    parser.add_argument('--test_dataset', type=str, default='hc3_english', help='test dataset')
    parser.add_argument('--test_data_nums', type=int, default=1000)

    args = parser.parse_args()

    if args.method not in support_methods:
        raise ValueError('method not supported')
    if args.test_dataset not in support_datasets:
        raise ValueError('test dataset not supported')

    classify = get_classifier(args.method)
    sentence = "DetectGPT is an amazing method to determine whether a piece of text is written by large language models (like ChatGPT, GPT3, GPT2, BLOOM etc). However, we couldn't find any open-source implementation of it. Therefore this is the implementation of the paper."
    print(classify(sentence))




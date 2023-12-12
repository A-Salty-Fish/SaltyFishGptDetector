# 各个算法比较

import argparse
import random
import time
from functools import partial

import data_convertor
import detect_gpt
import gltr
import hc3_ling
import hc3_single
import llmdet
import openai_roberta_base
import openai_roberta_large
import radar_vicuna

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
    start_time = time.time()

    classifier = None

    if method == 'detect_gpt':
        model = detect_gpt.init_model()

        def classify(text):
            return detect_gpt.classify_is_human(model, text=text)

        classifier = classify

    if method == 'gltr':
        model = gltr.LM()

        def classify(text):
            return gltr.classify_is_human(model, text=text)

        classifier = classify

    if method == 'hc3_ling':
        model = None

        def classify(text):
            return hc3_ling.classify_is_human(text=text)

        classifier = classify

    if method == 'hc3_single':
        model = hc3_single.init_classifier()

        def classify(text):
            return hc3_single.classify_is_human(model, text=text)

        classifier = classify

    if method == 'intrinsic-dim':
        model = hc3_single.init_classifier()

        def classify(text):
            return hc3_single.classify_is_human(model, text=text)

        classifier = classify

    if method == 'llmdet':
        model = llmdet.load_probability()

        def classify(text):
            return llmdet.classify_is_human(text=text)

        classifier = classify

    if method == 'openai-roberta-base':
        model = openai_roberta_base.init_classifier()

        def classify(text):
            return openai_roberta_base.classify_is_human(model, text=text)

        classifier = classify

    if method == 'openai-roberta-large':
        model = openai_roberta_large.init_classifier()

        def classify(text):
            return openai_roberta_large.classify_is_human(model, text=text)

        classifier = classify

    if method == 'radar-vicuna':
        model = radar_vicuna.init_classifier()

        def classify(text):
            return radar_vicuna.classify_is_human(model, text=text)

        classifier = classify

    end_time = time.time()
    print("time to init model and classifier was {} seconds.".format(end_time - start_time))
    if classifier == None:
        print("None Method")

    return classifier


def get_test_data(test_dataset, test_dataset_path, test_data_nums, shuffle=True):
    start_time = time.time()
    result = {
        'human': [],
        'ai': []
    }

    tmp_result = []

    if test_dataset == 'CHEAT':
        tmp_result = data_convertor.convert_CHEAT_dataset(test_dataset_path)
    if test_dataset == 'ghostbuster':
        tmp_result = data_convertor.convert_ghostbuster_dataset(test_dataset_path)
    if test_dataset == 'hc3_english':
        tmp_result = data_convertor.convert_hc3_english(test_dataset_path)
    if test_dataset == 'hc3_plus_english':
        tmp_result = data_convertor.convert_hc3_plus_english(test_dataset_path)
    if test_dataset == 'm4':
        tmp_result = data_convertor.convert_m4(test_dataset_path)

    result['human'] = [x for x in tmp_result if x['label'] == 0]
    result['ai'] = [x for x in tmp_result if x['label'] == 1]

    if shuffle:
        random.shuffle(result['human'])
        random.shuffle(result['ai'])


    result['human'] = result['human'][0: min(test_data_nums, len(result['human']))]
    result['ai'] = result['ai'][0: min(test_data_nums, len(result['ai']))]

    end_time = time.time()
    print("time to load test dataset was {} seconds.".format(end_time - start_time))

    if len(result['human']) == 0 and len(result['ai']) == 0:
        raise ValueError("load test data error: no data, please check the path:" + test_dataset_path)

    return result


def simple_test(method, test_dataset, test_dataset_path, test_data_nums):
    classifier = get_classifier(method)
    data_set = get_test_data(test_dataset, test_dataset_path, test_data_nums)
    test_result = test_classifier_and_dataset(classifier, data_set)
    test_result['method'] = method
    test_result['dataset'] = test_dataset
    test_result['dataset_path'] = test_dataset_path
    return test_result


def test_classifier_and_dataset(classifier, data_set):
    print("test begin")
    start_time = time.time()

    human_true = 0
    human_total = 0
    human_true_rate = 0.0
    ai_true = 0
    ai_total = 0
    ai_true_rate = 0.0
    precision = 0.0
    recall = 0.0
    f1 = 0

    for data in data_set['human'] + data_set['ai']:
        content = data['content']
        label = data['label']
        if label == 0:
            human_total += 1
            if classifier(content):
                human_true += 1
        elif label == 1:
            ai_total += 1
            if not classifier(content):
                ai_true += 1

    if human_total != 0:
        human_true_rate = human_true / human_total
    if ai_total != 0:
        ai_true_rate = ai_true / ai_total

    if ai_total != 0 and human_total != 0:
        precision = ai_true / (ai_true + (human_total - human_true))
        recall = ai_true / ai_total
        f1 = 2 * precision * recall / (precision + recall)

    test_result = {
        "human_true": human_true,
        "human_total": human_total,
        "human_true_rate": human_true_rate,
        "ai_true": ai_true,
        "ai_total": ai_total,
        "ai_true_rate": ai_true_rate,
        "precision": precision,
        "recall": recall,
        "f1": f1
    }

    end_time = time.time()
    print("time to test {} seconds.".format(end_time - start_time))
    return test_result


def multi_test(method, test_datasets, test_dataset_paths, test_data_nums):
    classifier = get_classifier(method)
    data_sets = []
    for i in range(0, min(len(test_datasets), len(test_dataset_paths))):
        data_set = get_test_data(test_datasets[i], test_dataset_paths[i], test_data_nums)
        data_sets.append(data_set)
    multi_test_result = test_classifier_and_datasets(classifier, data_sets)
    for i in range(0, min(len(test_datasets), len(test_dataset_paths))):
        multi_test_result[i]['dataset'] = test_datasets[i]
        multi_test_result[i]['dataset_path'] = test_dataset_paths[i]
    return multi_test_result


def test_classifier_and_datasets(classifier, data_sets):
    result = []
    for data_set in data_sets:
        result.append(test_classifier_and_dataset(classifier, data_set))
    return result


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--method', type=str, default='gltr', help='baseline method,')
    parser.add_argument('--test_dataset', type=str, default='hc3_english', help='test dataset')
    parser.add_argument('--test_dataset_path', type=str, default='../data_collector/test_data/hc3_english',
                        help='test dataset path')
    parser.add_argument('--test_data_nums', type=int, default=1000)

    args = parser.parse_args()

    if args.method not in support_methods:
        raise ValueError('method not supported')
    for td in args.test_dataset.split(','):
        if td not in support_datasets:
            raise ValueError('test dataset not supported:' + td)
    if args.test_data_nums <= 0:
        raise ValueError('test nums must > 0')

    # test load classifier
    # classify = get_classifier(args.method)
    # sentence = "DetectGPT is an amazing method to determine whether a piece of text is written by large language models (like ChatGPT, GPT3, GPT2, BLOOM etc). However, we couldn't find any open-source implementation of it. Therefore this is the implementation of the paper."
    # print(classify(sentence))

    # test load data set
    # test_data_set = get_test_data('CHEAT', '../data_collector/test_data/CHEAT', args.test_data_nums)
    # print(len(test_data_set['human']))
    # print(len(test_data_set['ai']))
    #
    # test_data_set = get_test_data('m4', '../data_collector/test_data/m4', args.test_data_nums)
    # print(len(test_data_set['human']))
    # print(len(test_data_set['ai']))
    #
    # test_data_set = get_test_data('ghostbuster', '../data_collector/test_data/ghostbuster', args.test_data_nums)
    # print(len(test_data_set['human']))
    # print(len(test_data_set['ai']))
    #
    # test_data_set = get_test_data('hc3_english', '../data_collector/test_data/hc3_english', args.test_data_nums)
    # print(len(test_data_set['human']))
    # print(len(test_data_set['ai']))
    #
    # test_data_set = get_test_data('hc3_plus_english', '../data_collector/test_data/hc3_plus_english', args.test_data_nums)
    # print(len(test_data_set['human']))
    # print(len(test_data_set['ai']))

    # test simple test
    # print(simple_test(args.method, args.test_dataset, args.test_dataset_path, args.test_data_nums))

    # test multi test
    print(multi_test(args.method, args.test_dataset.split(','), args.test_dataset_path.split(','), args.test_data_nums))
    # python3 benchmark.py --test_data_nums 10 --method hc3_single --test_dataset CHEAT,m4,ghostbuster,hc3_english,hc3_plus_english --test_dataset_path ../data_collector/test_data/CHEAT,../data_collector/test_data/m4,../data_collector/test_data/ghostbuster,../data_collector/test_data/hc3_english,../data_collector/test_data/hc3_plus_english
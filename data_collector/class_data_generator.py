#!pip install transformers[sentencepiece]
import json
import time

from transformers import pipeline

def init_classifier():


    start_time = time.time()
    classifier = pipeline("zero-shot-classification", model="MoritzLaurer/DeBERTa-v3-base-mnli-fever-anli")
    end_time = time.time()
    print("load utc model success: " + str(end_time - start_time))
    return classifier


def classify(classifier, text, labels):
    result = []
    output = classifier(text, labels, multi_label=False)
    for i in range(0, len(output['labels'])):
        result.append([output['labels'][i], output['scores'][i]])
    if len(result) == 0:
        result.append(['None', 1.00])
    return result


if __name__ == '__main__':
    classifier = init_classifier()
    sequence_to_classify = "Angela Merkel is a politician in Germany and leader of the CDU"

    candidate_labels = ["medicine",
                        "law",
                        "computer science",
                        "finance",
                        "pedagogy",
                        "biology",
                        "psychology",
                        "political" ,
                        "sports" ,
                        "chemistry"
                        ]

    with open('./test_data/hc3_english/medicine.jsonl', 'r', encoding='utf-8') as input_f:
        for line in input_f:
            json_obj = json.loads(line)
            print(classify(classifier, json_obj['human_answers'][0], candidate_labels))
            print(classify(classifier, json_obj['chatgpt_answers'][0], candidate_labels))

#!pip install transformers[sentencepiece]
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
    output = classifier(sequence_to_classify, candidate_labels, multi_label=False)
    for i in range(0, len(output['labels'])):
        result.append([output['labels'][i], output['scores'][i]])
    if len(result) == 0:
        result.append(['None', 1.00])
    return result


if __name__ == '__main__':
    classifier = init_classifier()
    sequence_to_classify = "Angela Merkel is a politician in Germany and leader of the CDU"

    candidate_labels = ["politics", "economy", "entertainment", "environment"]

    print(classify(classifier, sequence_to_classify, candidate_labels))

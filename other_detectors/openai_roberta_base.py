# Use a pipeline as a high-level helper
from transformers import pipeline


def init_classifier():
    pipe = pipeline("text-classification", model="roberta-base-openai-detector")

    return pipe


def classify_is_human(classfier, text, bar=0.50000):
    res_0 = classfier(text)[0]
    if res_0['label'] == 'Real':
        return res_0['score'] >= bar
    else:
        return res_0['score'] < bar


if __name__ == '__main__':
    classifier = init_classifier()
    print(classify_is_human(classifier, "zerogpt is an advanced and reliable chat GPT detector tool designed to analyze text and determine if it was generated by a human or an AI-powered language model. is zerogpt reliable ? "))
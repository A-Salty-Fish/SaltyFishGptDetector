# Use a pipeline as a high-level helper
from transformers import pipeline


def init_translator():

    pipe = pipeline("translation", model="Helsinki-NLP/opus-mt-en-zh")

    return pipe


def translate(translator, context):
    return translator(context)

if __name__ == '__main__':
    translator = init_translator()

    print(translate(translator, "translate english to chinese"))
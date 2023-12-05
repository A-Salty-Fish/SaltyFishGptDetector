from transformers import pipeline

unmasker = pipeline('fill-mask', model='distilbert-base-multilingual-cased')

print(unmasker("Hello I'm a [MASK] model."))

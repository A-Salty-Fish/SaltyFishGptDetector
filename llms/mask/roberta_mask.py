from transformers import pipeline

unmasker = pipeline('fill-mask', model='roberta-large')

print(unmasker("Hello I'm a <mask> model."))

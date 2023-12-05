# Use a pipeline as a high-level helper
from transformers import pipeline

unmasker = pipeline("fill-mask", model="microsoft/deberta-v3-large")

print(unmasker("Hello I'm a [MASK] model."))
# Use a pipeline as a high-level helper
from transformers import pipeline

unmasker = pipeline("fill-mask", model="vinai/phobert-large")

print(unmasker("Hello I'm a <mask> model."))
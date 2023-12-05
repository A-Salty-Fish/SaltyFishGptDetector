from transformers import AutoTokenizer, AutoModelForCausalLM
from transformers import pipeline

tokenizer = AutoTokenizer.from_pretrained("cerebras/Cerebras-GPT-6.7B")
model = AutoModelForCausalLM.from_pretrained("cerebras/Cerebras-GPT-6.7B")

text = "Generative AI is "

pipe = pipeline("text-generation", model=model, tokenizer=tokenizer)
generated_text = pipe(text, max_length=50, do_sample=False, no_repeat_ngram_size=2)[0]
print(generated_text['generated_text'])


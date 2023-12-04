from transformers import AutoTokenizer, AutoModel
from transformers import AutoTokenizer, AutoModelForCausalLM
from transformers import pipeline
# import torch

if __name__ == '__main__':
    # device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    tokenizer = AutoTokenizer.from_pretrained("lmsys/vicuna-7b-v1.5")
    model = AutoModelForCausalLM.from_pretrained("lmsys/vicuna-7b-v1.5")

    text = "Generative AI is "

    pipe = pipeline("text-generation", model=model, tokenizer=tokenizer, device_map="auto")
    generated_text = pipe(text, max_length=50, do_sample=True, no_repeat_ngram_size=2)[0]
    print(generated_text['generated_text'])



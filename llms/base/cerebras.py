from transformers import AutoTokenizer, AutoModelForCausalLM
from transformers import pipeline


def init_model_and_tokenizer():

    tokenizer = AutoTokenizer.from_pretrained("cerebras/Cerebras-GPT-6.7B")
    model = AutoModelForCausalLM.from_pretrained("cerebras/Cerebras-GPT-6.7B")
    return model, tokenizer


def text_generate(model, tokenizer, context):
    pipe = pipeline("text-generation", model=model, tokenizer=tokenizer)
    generated_text = pipe(context, max_length=50, do_sample=False, no_repeat_ngram_size=2)[0]
    return generated_text['generated_text']


if __name__ == '__main__':
    model, tokenizer = init_model_and_tokenizer()
    print(text_generate(model, tokenizer, "Generative AI is "))

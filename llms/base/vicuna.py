from transformers import AutoTokenizer, AutoModel
from transformers import AutoTokenizer, AutoModelForCausalLM
from transformers import pipeline
# import torch


def init_pipe():
    # device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    tokenizer = AutoTokenizer.from_pretrained("lmsys/vicuna-7b-v1.5")
    model = AutoModelForCausalLM.from_pretrained("lmsys/vicuna-7b-v1.5")
    pipe = pipeline("text-generation", model=model, tokenizer=tokenizer, device_map="auto")

    return pipe


def text_generate(pipe, context):
    generated_text = pipe(context, max_length=50, do_sample=True, no_repeat_ngram_size=2)[0]
    return generated_text['generated_text']

if __name__ == '__main__':


    text = "Generative AI is "
    pipe = init_pipe()
    print(text_generate(pipe, text))






# pip install -q transformers accelerate
from transformers import AutoModelForCausalLM, AutoTokenizer

def init_model_and_tokenizer():

    checkpoint = "bigscience/bloomz-7b1-mt"

    tokenizer = AutoTokenizer.from_pretrained(checkpoint)
    model = AutoModelForCausalLM.from_pretrained(checkpoint, torch_dtype="auto", device_map="auto")

    return model, tokenizer

def text_generate(model, tokenizer, context):


    inputs = tokenizer.encode(context, return_tensors="pt").to("cuda")
    outputs = model.generate(inputs)
    return tokenizer.decode(outputs[0])


if __name__ == '__main__':
    model, tokenizer = init_model_and_tokenizer()
    print(text_generate(model, tokenizer, "hello , what's your name? "))


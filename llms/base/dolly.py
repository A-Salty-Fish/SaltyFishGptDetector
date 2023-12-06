import torch
from transformers import pipeline

def init_pipe():

    generate_text = pipeline(model="databricks/dolly-v2-12b", torch_dtype=torch.bfloat16, trust_remote_code=True, device_map="auto")
    return generate_text

def text_generate(pipe, context):

    res = pipe(context)
    return res[0]["generated_text"]


if __name__ == '__main__':
    pipe = init_pipe()
    print(text_generate(pipe, "Explain to me the difference between nuclear fission and fusion."))


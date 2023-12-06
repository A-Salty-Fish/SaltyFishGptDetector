from transformers import AutoTokenizer
import transformers
import torch

def init_pipe():

    model = "TinyLlama/TinyLlama-1.1B-intermediate-step-955k-token-2T"
    tokenizer = AutoTokenizer.from_pretrained(model)
    pipeline = transformers.pipeline(
        "text-generation",
        model=model,
        torch_dtype=torch.float16,
        device_map="auto",
    )
    return pipeline, tokenizer


def text_generate(pipe, tokenizer, context):


    sequences = pipe(
        context,
        do_sample=True,
        top_k=10,
        num_return_sequences=1,
        repetition_penalty=1.5,
        eos_token_id=tokenizer.eos_token_id,
        max_length=500,
    )

    return [seq['generated_text'] for seq in sequences]


if __name__ == '__main__':
    pipe, tokenizer = init_pipe()
    print(text_generate(pipe, tokenizer, 'The TinyLlama project aims to pretrain a 1.1B Llama model on 3 trillion tokens. With some proper optimization, we can achieve this within a span of "just" 90 days using 16 A100-40G GPUs 🚀🚀. The training has started on 2023-09-01.'))


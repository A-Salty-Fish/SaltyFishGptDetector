from transformers import pipeline, set_seed


def init_pipe():


    generator = pipeline('text-generation', model='gpt2-xl')
    set_seed(42)
    return generator


def text_generate(pipe, context):
    return pipe(context, max_length=30, num_return_sequences=5)


if __name__ == '__main__':
    pipe = init_pipe()
    print(text_generate(pipe, "Hello, I'm a language model,"))

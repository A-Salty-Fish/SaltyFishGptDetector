from transformers import pipeline

mask_token = '[MASK]'


def init_unmasker():
    unmasker = pipeline('fill-mask', model='distilbert-base-multilingual-cased')
    return unmasker

def get_mask_token():
    return mask_token

def get_unmask_result(unmasker, context):
    return unmasker(context)


if __name__ == '__main__':
    unmasker = init_unmasker()
    print(get_unmask_result(unmasker, f"Hello I'm a {get_mask_token()} model."))

from transformers import AutoTokenizer, AutoModel


def init_tokenizer_and_model():
    tokenizer = AutoTokenizer.from_pretrained("THUDM/chatglm3-6b-32k", trust_remote_code=True)
    model = AutoModel.from_pretrained("THUDM/chatglm3-6b-32k", trust_remote_code=True).half().cuda()
    model = model.eval()
    return model, tokenizer


def chat(model, tokenizer, context):
    response, history = model.chat(tokenizer, context, history=[])
    return response

if __name__ == '__main__':
    model, tokenizer = init_tokenizer_and_model()
    print(chat(model,tokenizer, "你好呀"))

import torch
from transformers import AutoTokenizer, AutoModelForQuestionAnswering
from transformers import pipeline



def init_qapipe():
    # Use a pipeline as a high-level helper

    ckpt = "mrm8488/longformer-base-4096-finetuned-squadv2"
    tokenizer = AutoTokenizer.from_pretrained(ckpt)
    model = AutoModelForQuestionAnswering.from_pretrained(ckpt)
    nlp = pipeline('question-answering', model=model, tokenizer=tokenizer)
    return nlp

def qa(qapipe, question, context):
    QA_input = {
        'question': question,
        'context': context
    }
    res = qapipe(QA_input)
    return res


# text = "Huggingface has democratized NLP. Huge thanks to Huggingface for this."
# question = "What has Huggingface done ?"
# encoding = tokenizer(question, text, return_tensors="pt")
# input_ids = encoding["input_ids"]
#
# # default is local attention everywhere
# # the forward method will automatically set global attention on question tokens
# attention_mask = encoding["attention_mask"]
#
# start_scores, end_scores = model(input_ids, attention_mask=attention_mask, return_dict=False)
# all_tokens = tokenizer.convert_ids_to_tokens(input_ids[0].tolist())
#
#
# answer_tokens = all_tokens[torch.argmax(start_scores) :torch.argmax(end_scores)+1]
# answer = tokenizer.decode(tokenizer.convert_tokens_to_ids(answer_tokens))
#
# print(answer)
# output => democratized NLP

if __name__ == '__main__':
    text = "Huggingface has democratized NLP. Huge thanks to Huggingface for this."
    question = "What has Huggingface done ?"
    qapipe = init_qapipe()
    print(qa(qapipe, question, text))
from transformers import DistilBertTokenizer, DistilBertForQuestionAnswering
import torch
from transformers import AutoModelForQuestionAnswering, AutoTokenizer, pipeline

def init_qa_pipe():

    tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased-distilled-squad')
    model = DistilBertForQuestionAnswering.from_pretrained('distilbert-base-uncased-distilled-squad')

    nlp = pipeline('question-answering', model=model, tokenizer=tokenizer)
    return nlp

def qa(qapipe, question, context):
    QA_input = {
        'question': question,
        'context': context
    }
    res = qapipe(QA_input)
    return res

# def qa(model, tokenizer, question, context):
#     inputs = tokenizer(question, context, return_tensors="pt")
#     with torch.no_grad():
#         outputs = model(**inputs)
#     answer_start_index = torch.argmax(outputs.start_logits)
#     answer_end_index = torch.argmax(outputs.end_logits)
#     predict_answer_tokens = inputs.input_ids[0, answer_start_index: answer_end_index + 1]
#     return (tokenizer.decode(predict_answer_tokens))


if __name__ == '__main__':

    question, text = "Who was Jim Henson?", "Jim Henson was a nice puppet"

    qapipe = init_qa_pipe()

    print(qa(qapipe, question, text))









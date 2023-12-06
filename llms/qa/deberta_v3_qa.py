from transformers import AutoModelForQuestionAnswering, AutoTokenizer, pipeline


def init_qapipe():

    model_name = "deepset/deberta-v3-base-squad2"
    # a) Get predictions
    # b) Load model & tokenizer
    model = AutoModelForQuestionAnswering.from_pretrained(model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    nlp = pipeline('question-answering', model=model, tokenizer=tokenizer)
    return nlp


def qa(qapipe, question, context):

    QA_input = {
        'question': question,
        'context': context
    }
    res = qapipe(QA_input)
    return res


if __name__ == '__main__':

    qapipe = init_qapipe()
    print(qa(qapipe, 'Why is model conversion important?', 'The option to convert models between FARM and transformers gives freedom to the user and let people easily switch between frameworks.'))


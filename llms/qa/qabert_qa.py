from transformers import AutoTokenizer, AutoModelForQuestionAnswering
from transformers import pipeline


def init_qapipe():

    QAtokenizer = AutoTokenizer.from_pretrained("SRDdev/QABERT-small")

    QAmodel = AutoModelForQuestionAnswering.from_pretrained("SRDdev/QABERT-small")

    ask = pipeline("question-answering", model= QAmodel , tokenizer = QAtokenizer)

    return ask

def qa(qapipe, question, context):
    QA_input = {
        'question': question,
        'context': context
    }
    res = qapipe(QA_input)
    return res


if __name__ == '__main__':

    context = '''Extractive Question Answering is the task of extracting an answer from a text given a question. An example of a
    question-answering dataset is the SQuAD dataset, which is entirely based on that task. If you would like to fine-tune
    a model on a SQuAD task, you may leverage the examples/pytorch/question-answering/run_squad.py script.
    '''
    question = "What is a good example of a question answering dataset?"
    # result = ask(question="What is a good example of a question answering dataset?", context=context)

    qapipe = init_qapipe()

    print(qa(qapipe, question, context))
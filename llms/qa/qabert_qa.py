from transformers import AutoTokenizer, AutoModelForQuestionAnswering
from transformers import pipeline

QAtokenizer = AutoTokenizer.from_pretrained("SRDdev/QABERT-small")

QAmodel = AutoModelForQuestionAnswering.from_pretrained("SRDdev/QABERT-small")

context = '''Extractive Question Answering is the task of extracting an answer from a text given a question. An example of a
question-answering dataset is the SQuAD dataset, which is entirely based on that task. If you would like to fine-tune
a model on a SQuAD task, you may leverage the examples/pytorch/question-answering/run_squad.py script.
'''

ask = pipeline("question-answering", model= QAmodel , tokenizer = QAtokenizer)

result = ask(question="What is a good example of a question answering dataset?", context=context)

print(f"Answer: '{result['answer']}'")
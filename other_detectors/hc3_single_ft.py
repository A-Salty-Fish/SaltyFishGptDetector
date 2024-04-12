import json
import time

from datasets import load_dataset
# fine tune the radar vicuna

from transformers import AutoModelForSequenceClassification, TrainingArguments, Trainer, AutoTokenizer, pipeline
from sklearn.metrics import accuracy_score, f1_score

def init_model_and_tokenizer():

    model = AutoModelForSequenceClassification.from_pretrained(
        "Hello-SimpleAI/chatgpt-detector-roberta", num_labels=2,
    )

    tokenizer = AutoTokenizer.from_pretrained("Hello-SimpleAI/chatgpt-detector-roberta", trust_remote_code=True, max_length=512)

    return model, tokenizer

def compute_metrics(pred):
    labels = pred.label_ids
    preds = pred.predictions.argmax(-1)
    f1 = f1_score(labels, preds, average="weighted")
    acc = accuracy_score(labels, preds)
    return {"accuracy": acc, "f1": f1}

def load_local_dataset(train_file_paths):
    start_time = time.time()

    local_dataset = load_dataset('json', data_files={
        'train': train_file_paths,
    })
    end_time = time.time()
    print("load dataset successful: " + str(end_time - start_time))
    return local_dataset

def tokenize_data(tokenizer, local_dataset):
    start_time = time.time()

    def tokenize(batch):
        return tokenizer(batch["content"], padding=True, truncation=True, max_length=512)

    tokenized_data = local_dataset.map(tokenize, batched=True, batch_size=None)
    tokenized_data.set_format("torch", columns=["input_ids", "attention_mask", "label"])

    end_time = time.time()
    print("tokenize dataset successful: " + str(end_time - start_time))
    return tokenized_data

def train(model, tokenizer, train_file_path , output_dif):
    training_args = TrainingArguments(
        output_dir=output_dif,
        learning_rate=1e-5,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=16,
        num_train_epochs=5,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        save_total_limit=2,
        load_best_model_at_end=True,
        push_to_hub=False,
    )

    local_dataset = load_local_dataset(train_file_path)
    tokenized_data = tokenize_data(tokenizer, local_dataset)

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_data["train"],
        eval_dataset=tokenized_data["train"],
        compute_metrics=compute_metrics,
    )

    trainer.train()


def prepare_train_data():
    with open('../my_detector/roberta_test/data/hc3_row.train', 'r', encoding='utf-8') as hc3_file, \
        open('../my_detector/roberta_test/data/hc3_mix_multi_prompt.train', 'r', encoding='utf-8') as hc3_adv_file:
        all_json_objs = json.load(hc3_adv_file) + json.load(hc3_file)
        with open('./hc3_single_ft.train', 'w', encoding='utf-8') as out_f:
            out_f.write(json.dumps(all_json_objs))


def init_classifier(model_path='./hc3_single_ft_adt/checkpoint-905/'):
    classifier = pipeline("sentiment-analysis", model=model_path,
                          tokenizer=AutoTokenizer.from_pretrained("Hello-SimpleAI/chatgpt-detector-roberta", trust_remote_code=True, max_length=512))
    return classifier


def classify_is_human(classifier, text, bar=0.5):

    raw_output = classifier(text)[0]
    # print(raw_output)
    if raw_output['label'] == 'Human':
        return raw_output['score'] >= bar
    else:
        return raw_output['score'] < bar



if __name__ == '__main__':

    model, tokenizer = init_model_and_tokenizer()
    # prepare_train_data()
    # train(model, tokenizer, ['../my_detector/roberta_test/data/hc3_row.train'], 'hc3_single_ft_adt')


    moe_train_files = [
        '../my_detector/moe_test/data/7_m4_chatGPT.json.train',
        # '../my_detector/moe_test/data/nature/mix/7.jsonl.rewrite.jsonl.train',
        # '../my_detector/moe_test/data/nature/qwen/7.jsonl.qwen.rewrite.jsonl.train',
        # '../my_detector/moe_test/data/nature/qwen/8.jsonl.qwen.rewrite.jsonl.train',
        # '../my_detector/moe_test/data/nature/qwen/9.jsonl.qwen.rewrite.jsonl.train',
        # '../my_detector/moe_test/data/nature/qwen/10.jsonl.qwen.rewrite.jsonl.train',
        # '../my_detector/moe_test/data/adversary/qwen/7.jsonl.qwen.rewrite.jsonl.qwen.paraphase.jsonl.train',
        # '../my_detector/moe_test/data/adversary/qwen/8.jsonl.qwen.rewrite.jsonl.qwen.paraphase.jsonl.train',
        # '../my_detector/moe_test/data/adversary/qwen/9.jsonl.qwen.rewrite.jsonl.qwen.paraphase.jsonl.train',
        # '../my_detector/moe_test/data/adversary/qwen/10.jsonl.qwen.rewrite.jsonl.qwen.paraphase.jsonl.train',
    ]
    train(model, tokenizer, moe_train_files, 'hc3_single_ft_moe_4')



    # human_text = "I know this question has a lot of answers already, but I feel the answers are phrased either strongly against, or mildly for, co-signing. What it amounts down to is that this is a personal choice. You cannot receive reliable information as to whether or not co-signing this loan is a good move due to lack of information. The person involved is going to know the person they would be co-signing for, and the people on this site will only have their own personal preferences of experiences to draw from. You know if they are reliable, if they will be able to pay off the loan without need for the banks to come after you.  This site can offer general theories, but I think it should be kept in mind that this is wholly a personal decision for the person involved, and them alone to make based on the facts that they know and we do not."
    # ai_text = "Co-signing a personal loan for a friend or family member can be a risky proposition. When you co-sign a loan, you are agreeing to be responsible for the loan if the borrower is unable to make the payments. This means that if your friend or family member defaults on the loan, you will be on the hook for the remaining balance.There are a few things to consider before co-signing a personal loan for someone:Do you trust the borrower to make the payments on time and in full? If you are not confident that the borrower will be able to make the payments, it may not be a good idea to co-sign the loan.Can you afford to make the payments if the borrower defaults? If you are unable to make the payments, co-signing the loan could put your own financial stability at risk.What is the purpose of the loan? If the borrower is using the loan for a risky or questionable venture, it may not be worth the risk to co-sign.Is there another way for the borrower to get the loan without a co-signer? If the borrower has a good credit score and is able to qualify for a loan on their own, it may not be necessary for you to co-sign.In general, it is important to carefully consider the risks and potential consequences before co-signing a loan for someone. If you do decide to co-sign, it is a good idea to have a conversation with the borrower about their plans for making the loan payments and to have a clear understanding of your responsibilities as a co-signer."
    #
    #
    # classifier = init_classifier()
    # print(classifier(human_text))
    # print(classifier(ai_text))
    #
    # print(classify_is_human(classifier, human_text))
    # print(classify_is_human(classifier, ai_text))

    pass
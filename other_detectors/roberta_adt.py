import time

import torch
from torch import nn
from transformers import RobertaTokenizer, RobertaModel

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def init_model_and_tokenizer(state_dict_path):
    start_time = time.time()
    tokenizer = RobertaTokenizer.from_pretrained('roberta-base')
    model = RobertaModel.from_pretrained('roberta-base')

    model.load_state_dict(torch.load(state_dict_path))
    model.eval()
    # device = get_best_device()
    model.to(device)
    classifier = nn.Linear(model.config.hidden_size, 2).to(device)
    end_time = time.time()
    print("load model successful: " + str(end_time - start_time))
    return model, tokenizer, classifier


def classify_is_human(model, tokenizer, classifier, text):

    inputs = tokenizer([text], padding=True, truncation=True, return_tensors="pt").to(device)

    outputs = model(**inputs)[0][:, 0, :]
    # 计算原始的分类器损失
    logits = classifier(outputs)
    # print(logits)
    _, predicted = torch.max(logits, 1)
    return predicted[0].item()


if __name__ == '__main__':
    state_dict_path = './roberta_result/model_epoch_2.pt'
    model, tokenizer, classifier = init_model_and_tokenizer(state_dict_path)
    human_text = "I know this question has a lot of answers already, but I feel the answers are phrased either strongly against, or mildly for, co-signing. What it amounts down to is that this is a personal choice. You cannot receive reliable information as to whether or not co-signing this loan is a good move due to lack of information. The person involved is going to know the person they would be co-signing for, and the people on this site will only have their own personal preferences of experiences to draw from. You know if they are reliable, if they will be able to pay off the loan without need for the banks to come after you.  This site can offer general theories, but I think it should be kept in mind that this is wholly a personal decision for the person involved, and them alone to make based on the facts that they know and we do not."
    ai_text = "Co-signing a personal loan for a friend or family member can be a risky proposition. When you co-sign a loan, you are agreeing to be responsible for the loan if the borrower is unable to make the payments. This means that if your friend or family member defaults on the loan, you will be on the hook for the remaining balance.There are a few things to consider before co-signing a personal loan for someone:Do you trust the borrower to make the payments on time and in full? If you are not confident that the borrower will be able to make the payments, it may not be a good idea to co-sign the loan.Can you afford to make the payments if the borrower defaults? If you are unable to make the payments, co-signing the loan could put your own financial stability at risk.What is the purpose of the loan? If the borrower is using the loan for a risky or questionable venture, it may not be worth the risk to co-sign.Is there another way for the borrower to get the loan without a co-signer? If the borrower has a good credit score and is able to qualify for a loan on their own, it may not be necessary for you to co-sign.In general, it is important to carefully consider the risks and potential consequences before co-signing a loan for someone. If you do decide to co-sign, it is a good idea to have a conversation with the borrower about their plans for making the loan payments and to have a clear understanding of your responsibilities as a co-signer."
    print(classify_is_human(model, tokenizer, classifier, human_text))
    print(classify_is_human(model, tokenizer, classifier, ai_text))



import nltk
import numpy as np
# nltk.download('stopwords')
# nltk.download('punkt')

# https://huggingface.co/lucadiliello/BLEURT-20
# pip install git+https://github.com/lucadiliello/bleurt-pytorch.git
# pip install rouge

from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize


def preprocess_text(text):
    stop_words = set(stopwords.words('english'))
    word_tokens = word_tokenize(text.lower())
    filtered_text = [word for word in word_tokens if word.isalnum() and word not in stop_words]
    return " ".join(filtered_text)


def calculate_cosine_similarity(text1, text2):
    preprocessed_text1 = preprocess_text(text1)
    preprocessed_text2 = preprocess_text(text2)

    vectorizer = CountVectorizer().fit_transform([preprocessed_text1, preprocessed_text2])
    vectors = vectorizer.toarray()

    cosine_sim = cosine_similarity(vectors)

    return cosine_sim[0][1]


def calculate_euclidean_distance(text1, text2):
    preprocessed_text1 = preprocess_text(text1)
    preprocessed_text2 = preprocess_text(text2)

    tfidf_vectorizer = TfidfVectorizer()
    tfidf_matrix = tfidf_vectorizer.fit_transform([preprocessed_text1, preprocessed_text2])

    tfidf_array = tfidf_matrix.toarray()
    euclidean_dist = np.linalg.norm(tfidf_array[0] - tfidf_array[1])

    return euclidean_dist


def calculate_edit_distance(text1, text2):
    m = len(text1)
    n = len(text2)

    # 初始化动态规划矩阵
    dp = [[0 for j in range(n + 1)] for i in range(m + 1)]

    # 初始化第一行和第一列
    for i in range(1, m + 1):
        dp[i][0] = i
    for j in range(1, n + 1):
        dp[0][j] = j

    # 计算动态规划矩阵
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if text1[i - 1] == text2[j - 1]:
                dp[i][j] = dp[i - 1][j - 1]
            else:
                dp[i][j] = 1 + min(dp[i - 1][j], dp[i][j - 1], dp[i - 1][j - 1])

    return dp[m][n]


# bleu score part

import torch
from bleurt_pytorch import BleurtConfig, BleurtForSequenceClassification, BleurtTokenizer

def init_bleu_model_and_tokenizer():

    config = BleurtConfig.from_pretrained('lucadiliello/BLEURT-20')
    model = BleurtForSequenceClassification.from_pretrained('lucadiliello/BLEURT-20')
    tokenizer = BleurtTokenizer.from_pretrained('lucadiliello/BLEURT-20')



    model.eval()

    return  model, tokenizer

def get_bleu_score(model, tokenizer, text1, text2):
    references = [text1]
    candidates = [text2]

    with torch.no_grad():
        inputs = tokenizer(references, candidates, padding='longest', return_tensors='pt')
        res = model(**inputs).logits.flatten().tolist()
        return res[0]


def get_all_score(bleu_model, bleu_tokenizer, text1, text2):
    cosine_score = calculate_cosine_similarity(text1, text2)
    euclidean_score = calculate_euclidean_distance(text1, text2)
    edit_distance = calculate_edit_distance(text1, text2)
    bleu_score = get_bleu_score(bleu_model, bleu_tokenizer, text1, text2)
    return {
        'cosine': cosine_score,
        'euclidean': euclidean_score,
        'edit_distance': edit_distance,
        'bleu': bleu_score,
    }

# rouge
from rouge import Rouge

def init_rouge_scorer():
    rouge = Rouge()
    return rouge

def get_rouge_score(rouge_scorer, text1, text2):
    # text1: hypothesis
    # text2: reference
    # result:
    # [
    #     {
    #         "rouge-1": {
    #             "f": 0.4786324739396596,
    #             "p": 0.6363636363636364,
    #             "r": 0.3835616438356164
    #         },
    #         "rouge-2": {
    #             "f": 0.2608695605353498,
    #             "p": 0.3488372093023256,
    #             "r": 0.20833333333333334
    #         },
    #         "rouge-l": {
    #             "f": 0.44705881864636676,
    #             "p": 0.5277777777777778,
    #             "r": 0.3877551020408163
    #         }
    #     }
    # ]
    return rouge_scorer.get_scores(hypothesis, reference)[0]['rouge-1']



if __name__ == '__main__':

    text1 = "This is the first text"
    text2 = "This is the second text"

    cosine_sim = calculate_cosine_similarity(text1, text2)
    print("Cosine Similarity between the two texts:", cosine_sim)

    euclidean_dist = calculate_euclidean_distance(text1, text2)
    print("Euclidean Distance between the two texts:", euclidean_dist)

    edit_dist = calculate_edit_distance(text1, text2)
    print("Minimum Edit Distance between the two texts:", edit_dist)

    model,tokenizer = init_bleu_model_and_tokenizer()
    print(get_all_score(model, tokenizer, text1, text2))

    hypothesis = "the #### transcript is a written version of each day 's cnn student news program use this transcript to he    lp students with reading comprehension and vocabulary use the weekly newsquiz to test your knowledge of storie s you     saw on cnn student news"
    reference = "this page includes the show transcript use the transcript to help students with reading comprehension and     vocabulary at the bottom of the page , comment for a chance to be mentioned on cnn student news . you must be a teac    her or a student age # # or older to request a mention on the cnn student news roll call . the weekly newsquiz tests     students ' knowledge of even ts in the news"
    rouge_scorer = init_rouge_scorer()
    print(get_rouge_score(rouge_scorer, hypothesis, reference))

    pass
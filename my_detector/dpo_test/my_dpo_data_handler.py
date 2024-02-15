import nltk
import numpy as np
# nltk.download('stopwords')
# nltk.download('punkt')

# https://huggingface.co/lucadiliello/BLEURT-20
# pip install git+https://github.com/lucadiliello/bleurt-pytorch.git

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



if __name__ == '__main__':

    text1 = "This is the first text"
    text2 = "This is the second text"

    cosine_similarity = calculate_cosine_similarity(text1, text2)
    print("Cosine Similarity between the two texts:", cosine_similarity)

    euclidean_distance = calculate_euclidean_distance(text1, text2)
    print("Euclidean Distance between the two texts:", euclidean_distance)

    edit_distance = calculate_edit_distance(text1, text2)
    print("Minimum Edit Distance between the two texts:", edit_distance)

    model,tokenizer = init_bleu_model_and_tokenizer()
    print(get_all_score(model, tokenizer, text1, text2))
    pass
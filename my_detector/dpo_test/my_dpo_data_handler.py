import json

import nltk
import numpy as np
import transformers
# nltk.download('stopwords')
# nltk.download('punkt')

# https://huggingface.co/lucadiliello/BLEURT-20
# pip install git+https://github.com/lucadiliello/bleurt-pytorch.git
# pip install rouge

from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from tqdm import tqdm

transformers.logging.set_verbosity_error()

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
        inputs = tokenizer(references, candidates, truncation=True, return_tensors='pt', max_length=512)
        res = model(**inputs).logits.flatten().tolist()
        return res[0]

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
    return rouge_scorer.get_scores(text1, text2)[0]['rouge-1']['f']

def get_all_score(bleu_model, bleu_tokenizer, rouge_scorer, text1, text2):
    cosine_score = calculate_cosine_similarity(text1, text2)
    euclidean_score = calculate_euclidean_distance(text1, text2)
    edit_distance = calculate_edit_distance(text1, text2)
    bleu_score = get_bleu_score(bleu_model, bleu_tokenizer, text1, text2)
    rouge_score = get_rouge_score(rouge_scorer, text1, text2)
    return {
        'cosine': cosine_score,
        'euclidean': euclidean_score,
        'edit_distance': edit_distance,
        'bleu': bleu_score,
        'rouge': rouge_score
    }


def output_list_metricts(scores):
    print(f"mean: {np.mean(scores)}, max: {np.max(scores)}, min: {np.min(scores)}, var: {np.var(scores)}")

def test_qwen_score():
    model,tokenizer = init_bleu_model_and_tokenizer()
    rouge_scorer = init_rouge_scorer()
    row_texts = []
    rewrite_texts = []
    with open('./qwen/cheat_generation.test.qwen.jsonl', 'r', encoding='utf-8') as in_f:
        for line in in_f:
            json_obj = json.loads(line)
            try:
                row_texts.append(json_obj['ai'])
                rewrite_texts.append(json_obj['ai_rewrite'])
            except Exception as e:
                print(json_obj)
                print(e)
    all_scores = []
    for i in tqdm(range(0, len(row_texts))):
        try:
            all_score = get_all_score(model, tokenizer, rouge_scorer, row_texts[i], rewrite_texts[i])
            all_scores.append(all_score)
        except Exception as e:
            print(e)
            print(row_texts[i])
            print(rewrite_texts[i])
    cosines = [x['cosine'] for x in all_scores]
    euclideans = [x['euclidean'] for x in all_scores]
    edit_distances = [x['edit_distance'] for x in all_scores]
    bleus = [x['bleu'] for x in all_scores]
    rouges = [x['rouge'] for x in all_scores]

    print("cosines")
    output_list_metricts(cosines)
    print("euclideans")
    output_list_metricts(euclideans)
    print("edit_distances")
    output_list_metricts(edit_distances)
    print("bleus")
    output_list_metricts(bleus)
    print("rouges")
    output_list_metricts(rouges)



if __name__ == '__main__':

    # text1 = "This is the first text"
    # text2 = "This is the second text"
    #
    # cosine_sim = calculate_cosine_similarity(text1, text2)
    # print("Cosine Similarity between the two texts:", cosine_sim)
    #
    # euclidean_dist = calculate_euclidean_distance(text1, text2)
    # print("Euclidean Distance between the two texts:", euclidean_dist)
    #
    # edit_dist = calculate_edit_distance(text1, text2)
    # print("Minimum Edit Distance between the two texts:", edit_dist)
    #
    model,tokenizer = init_bleu_model_and_tokenizer()
    #
    # hypothesis = "the #### transcript is a written version of each day 's cnn student news program use this transcript to he    lp students with reading comprehension and vocabulary use the weekly newsquiz to test your knowledge of storie s you     saw on cnn student news"
    # reference = "this page includes the show transcript use the transcript to help students with reading comprehension and     vocabulary at the bottom of the page , comment for a chance to be mentioned on cnn student news . you must be a teac    her or a student age # # or older to request a mention on the cnn student news roll call . the weekly newsquiz tests     students ' knowledge of even ts in the news"
    # rouge_scorer = init_rouge_scorer()
    # print(get_rouge_score(rouge_scorer, hypothesis, reference))
    #
    # print(get_all_score(model, tokenizer, rouge_scorer, text1, text2))

    text1 = '''Historical price-to-earnings (P/E) ratios for small-cap and large-cap stocks can vary significantly over time and may not be directly comparable due to the different characteristics of these two categories of stocks.Small-cap stocks, which are defined as stocks with a market capitalization of less than $2 billion, tend to be riskier and more volatile than large-cap stocks, which have a market capitalization of $10 billion or more. As a result, investors may be willing to pay a higher price for the potential growth opportunities offered by small-cap stocks, which can lead to higher P/E ratios.On the other hand, large-cap stocks tend to be more established and stable, with a longer track record of earnings and revenue growth. As a result, these stocks may trade at lower P/E ratios, as investors may be less willing to pay a premium for their growth potential.It is important to note that P/E ratios are just one factor to consider when evaluating a stock and should not be used in isolation. Other factors, such as the company's financial health, industry trends, and macroeconomic conditions, can also impact a stock's P/E ratio.'''
    text2 = '''Historically, P/\E divergencebetween small-cap and large-cap stocks isnobsoletefact.P/\E ratios Small-cap(under $2 billion market cap) differ fundamentallyfromlarge-cap($10 billion or above) due to disparities ingrowth potential, risk, and steadiness.Riskier Small-Caps:

1. Volatility: Bigger price swings and unpredictability.
2. Lack of history: Shorter revenue/earnings streams.

Reasons why:
a. Younger businesses, nascent industries, etc.
b. Sensitive to market fluctuations.

Premium Paid:
Investors crave growth possibilities, willing to overpay = Higher P/\E ratios.

Large-Caps (Established):

1. Predictability: Less volatility and more stability.
2. Longer History: Consistent Revenue/Earnings growth for decades.

Reasons why:
a. Well-established enterprises.
b. Solid financials and industry dominance.

Lower P/\E Ratios:
Investors less eager to overpay for less potential = Lower P/\E ratios.

Overall:

1. P/\E ratios should complement, NOT dictate, decision-making.
2. Company health, industry, and economy matter.

Example: Two similar companies, X with a P/\E 20 vs Y with a P/\E 40:
Instead of assuming X is undervalued and Y overvalued, analyze:

X: Why the underperformance?
Y: Why the outperformance?
Bear in mind, P/\E comparisons only provide a partial insight!'''
    print(get_bleu_score(model, tokenizer, text1, text1))
    print(len(text2.split(' ')))
    text1_char_set = []
    for x in text1:
        if x not in text1_char_set:
            text1_char_set.append(x)
    print([x for x in text2 if x not in text1_char_set])
    text2 = text2.replace('\\', '')
    text2 = text2.replace('\n', '')
    print(get_bleu_score(model, tokenizer, text1, text2))
    print(get_bleu_score(model, tokenizer, text2, " ".join(text2.split(' ')[0:400])))
    print(get_bleu_score(model, tokenizer, text1, " ".join(text2.split(' ')[0:400])))


    # test_qwen_score()

    pass
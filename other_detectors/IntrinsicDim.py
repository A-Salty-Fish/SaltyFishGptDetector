import numpy as np

# https://github.com/ArGintum/GPTID
# 论文：Intrinsic Dimension Estimation for Robust Detection of AI-Generated Texts

from scipy.spatial.distance import cdist
from threading import Thread

MINIMAL_CLOUD = 47


def prim_tree(adj_matrix, alpha=1.0):
    infty = np.max(adj_matrix) + 10

    dst = np.ones(adj_matrix.shape[0]) * infty
    visited = np.zeros(adj_matrix.shape[0], dtype=bool)
    ancestor = -np.ones(adj_matrix.shape[0], dtype=int)

    v, s = 0, 0.0
    for i in range(adj_matrix.shape[0] - 1):
        visited[v] = 1
        ancestor[dst > adj_matrix[v]] = v
        dst = np.minimum(dst, adj_matrix[v])
        dst[visited] = infty

        v = np.argmin(dst)
        s += (adj_matrix[v][ancestor[v]] ** alpha)

    return s.item()


def process_string(sss):
    return sss.replace('\n', ' ').replace('  ', ' ')


class PHD():
    def __init__(self, alpha=1.0, metric='euclidean', n_reruns=3, n_points=7, n_points_min=3):


        '''
        Initializes the instance of PH-dim computer
        Parameters:
            1) alpha --- real-valued parameter Alpha for computing PH-dim (see the reference paper). Alpha should be chosen lower than
        the ground-truth Intrinsic Dimensionality; however, Alpha=1.0 works just fine for our kind of data.
            2) metric --- String or Callable, distance function for the metric space (see documentation for Scipy.cdist)
            3) n_reruns --- Number of restarts of whole calculations (each restart is made in a separate thread)
            4) n_points --- Number of subsamples to be drawn at each subsample
            5) n_points_min --- Number of subsamples to be drawn at larger subsamples (more than half of the point cloud)
        '''
        self.alpha = alpha
        self.n_reruns = n_reruns
        self.n_points = n_points
        self.n_points_min = n_points_min
        self.metric = metric
        self.is_fitted_ = False


    def _sample_W(self, W, nSamples):
        n = W.shape[0]
        random_indices = np.random.choice(n, size=nSamples, replace=False)
        return W[random_indices]


    def _calc_ph_dim_single(self, W, test_n, outp, thread_id):
        lengths = []
        for n in test_n:
            if W.shape[0] <= 2 * n:
                restarts = self.n_points_min
            else:
                restarts = self.n_points

            reruns = np.ones(restarts)
            for i in range(restarts):
                tmp = self._sample_W(W, n)
                reruns[i] = prim_tree(cdist(tmp, tmp, metric=self.metric), self.alpha)

            lengths.append(np.median(reruns))
        lengths = np.array(lengths)

        x = np.log(np.array(list(test_n)))
        y = np.log(lengths)
        N = len(x)
        outp[thread_id] = (N * (x * y).sum() - x.sum() * y.sum()) / (N * (x ** 2).sum() - x.sum() ** 2)


    def fit_transform(self, X, y=None, min_points=50, max_points=512, point_jump=40):


        '''
        Computing the PH-dim
        Parameters:
            1) X --- point cloud of shape (n_points, n_features),
            2) y --- fictional parameter to fit with Sklearn interface
            3) min_points --- size of minimal subsample to be drawn
            4) max_points --- size of maximal subsample to be drawn
            5) point_jump --- step between subsamples
        '''
        ms = np.zeros(self.n_reruns)
        test_n = range(min_points, max_points, point_jump)
        threads = []

        for i in range(self.n_reruns):
            threads.append(Thread(target=self._calc_ph_dim_single, args=[X, test_n, ms, i]))
            threads[-1].start()

        for i in range(self.n_reruns):
            threads[i].join()

        m = np.mean(ms)
        return 1 / (1 - m)


import numpy as np
import pandas as pd

import torch
import torch.nn as nn

from sklearn.model_selection import cross_validate
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from transformers import RobertaTokenizer, RobertaModel

from scipy.spatial.distance import cdist
from skdim.id import MLE

from tqdm import tqdm

from IntrinsicDim import PHD

model_path = 'projecte-aina/roberta-base-ca-cased-ner'
tokenizer_path = model_path

### Loading the model
tokenizer = RobertaTokenizer.from_pretrained(tokenizer_path)
model = RobertaModel.from_pretrained(model_path)


"""
Our method (PHD) is stochastic, here are some magic constants for it. They are chosen specifically for text data. If you plan to use this code for something different, consider testing other values.

MIN_SUBSAMPLE       --- the size of the minimal subsample to be drawn in procedure. Lesser values yields less statisitcally stable predictions.
INTERMEDIATE_POINTS --- number of sumsamples to be drawn. The more this number is, the more stable dimension estimation for single text is; however,  the computational time is higher, too. 7 is, empirically, the best trade-off.
"""
MIN_SUBSAMPLE = 40
INTERMEDIATE_POINTS = 7

'''
Auxillary function. Clear text from linebreaks and odd whitespaces, because they seem to interfer with LM quite a lot.
Replace with a more sophisticated cleaner, if needed.
'''

def preprocess_text(text):
    return text.replace('\n', ' ').replace('  ', ' ')


'''
Get PHD for one text
Parameters:
        text  --- text
        solver --- PHD computator

Returns:
    real number or NumPy.nan  --- Intrinsic dimension value of the text in the input data
                                                    estimated by Persistence Homology Dimension method.'''


def get_phd_single(text, solver):
    inputs = tokenizer(preprocess_text(text), truncation=True, max_length=512, return_tensors="pt")
    with torch.no_grad():
        outp = model(**inputs)

    # We omit the first and last tokens (<CLS> and <SEP> because they do not directly correspond to any part of the)
    mx_points = inputs['input_ids'].shape[1] - 2

    mn_points = MIN_SUBSAMPLE
    step = (mx_points - mn_points) // INTERMEDIATE_POINTS

    return solver.fit_transform(outp[0][0].numpy()[1:-1], min_points=mn_points, max_points=mx_points - step, \
                                point_jump=step)


'''
Get PHD for all texts in df[key] Pandas DataSeries (PHD method)
Parameters:
        df  --- Pandas DataFrame
        key --- Name of the column
        is_list --- Check if the elements of the df[key] are lists (appears in some data)

        alpha --- Parameter alpha for PHD computattion

Returns:
    numpy.array of shape (number_of_texts, 1) --- Intrinsic dimension values for all texts in the input data
                                                    estimated by Persistence Homology Dimension method.
'''


def get_phd(df, key='text', is_list=False, alpha=1.0):
    dims = []
    PHD_solver = PHD(alpha=alpha, metric='euclidean', n_points=9)
    for s in tqdm(df[key]):
        if is_list:
            text = s[0]
        else:
            text = s
        dims.append(get_phd_single(text, PHD_solver))

    return np.array(dims).reshape(-1, 1)


'''
Get MLE for one text
Parameters:
        text  --- text
        solver --- MLE computator

Returns:
    real number or NumPy.nan  --- Intrinsic dimension value of the text in the input data
                                                    estimated by Maximum Likelihood Estimation method.'''


def get_mle_single(text, solver):
    inputs = tokenizer(preprocess_text(text), truncation=True, max_length=512, return_tensors="pt")
    with torch.no_grad():
        outp = model(**inputs)

    return solver.fit_transform(outp[0][0].numpy()[1:-1])


'''
Get PHD for all texts in df[key] Pandas DataSeries (PHD method)
Parameters:
        df  --- Pandas DataFrame
        key --- Name of the column
        is_list --- Check if the elements of the df[key] are lists (appears in some data)

Returns:
    numpy.array of shape (number_of_texts, 1) --- Intrinsic dimension values for all texts in the input data
                                                    estimated by Maximum Likelihood Estimation method.
'''


def get_mle(df, key='text', is_list=False):
    dims = []
    MLE_solver = MLE()
    for s in tqdm(df[key]):
        if is_list:
            text = s[0]
        else:
            text = s
        dims.append(get_mle_single(text, MLE_solver))

    return np.array(dims).reshape(-1, 1)


#Training subset

# reddit_data = pd.read_json("opt_13b_reddit.jsonl_pp", lines=True)
#
# human_phd_train_en = get_phd(reddit_data.iloc, 'gold_completion')
# opt_phd_train_en = get_phd(reddit_data.iloc, 'gen_completion',is_list=True)

sample_text = "Speaking of festivities, there is one day in China that stands unrivaled - the first day of the Lunar New Year, commonly referred to as the Spring Festival. Even if you're generally uninterested in celebratory events, it's hard to resist the allure of the family reunion dinner, a quintessential aspect of the Spring Festival. Throughout the meal, family members raise their glasses to toast one another, expressing wishes for happiness, peace, health, and prosperity in the upcoming year."

# print("PHD estimation of the Intrinsic dimension of sample text is ", get_phd_single(sample_text, PHD( metric='euclidean', n_points=9)))


# sample_text = "Speaking of festivities, there is one day in China that stands unrivaled - the first day of the Lunar New Year, commonly referred to as the Spring Festival. Even if you're generally uninterested in celebratory events, it's hard to resist the allure of the family reunion dinner, a quintessential aspect of the Spring Festival. Throughout the meal, family members raise their glasses to toast one another, expressing wishes for happiness, peace, health, and prosperity in the upcoming year."
# print("MLE estimation of the Intrinsic dimension of sample text is ", get_mle_single(sample_text, MLE()))
#Training subset

# reddit_data = pd.read_json("data/chatgpt_reddit.jsonl_pp", lines=True)

# human_mle_train_en = get_phd(reddit_data.iloc[train_idx], 'gold_completion')
# chatgpt_mle_train_en = get_phd(reddit_data.iloc[train_idx], 'gen_completion',is_list=False)


def classify_is_human(text, bar=5):
    phd = PHD(metric='euclidean', n_points=9)
    score = get_phd_single(sample_text, phd)
    if score > bar:
        return False
    else:
        return True


if __name__ == '__main__':
    print(classify_is_human(sample_text))
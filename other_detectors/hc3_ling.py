# Copyright (c) Hello-SimpleAI Org. 2023.
# Licensed under the Apache License, Version 2.0.

# 源码地址：https://huggingface.co/spaces/Hello-SimpleAI/chatgpt-detector-ling/tree/main

import os
import pickle
import re
from typing import Callable, List, Tuple

from nltk.data import load as nltk_load
import numpy as np
from sklearn.linear_model import LogisticRegression
import torch
from transformers.utils import cached_file
from transformers import GPT2LMHeadModel, GPT2Tokenizer


AUTH_TOKEN = os.environ.get("access_token")
DET_LING_ID = 'Hello-SimpleAI/chatgpt-detector-ling'


def download_file(filename):
    return cached_file(DET_LING_ID, filename, use_auth_token=AUTH_TOKEN)


NLTK = nltk_load(download_file('english.pickle'))
sent_cut_en = NLTK.tokenize
LR_GLTR_EN, LR_PPL_EN, LR_GLTR_ZH, LR_PPL_ZH = [
    pickle.load(open(download_file(f'{lang}-gpt2-{name}.pkl'), 'rb'))
    for lang, name in [('en', 'gltr'), ('en', 'ppl'), ('zh', 'gltr'), ('zh', 'ppl')]
]

NAME_EN = 'gpt2'
TOKENIZER_EN = GPT2Tokenizer.from_pretrained(NAME_EN)
MODEL_EN = GPT2LMHeadModel.from_pretrained(NAME_EN)

NAME_ZH = 'IDEA-CCNL/Wenzhong-GPT2-110M'
TOKENIZER_ZH = GPT2Tokenizer.from_pretrained(NAME_ZH)
MODEL_ZH = GPT2LMHeadModel.from_pretrained(NAME_ZH)


# code borrowed from https://github.com/blmoistawinde/HarvestText
def sent_cut_zh(para: str) -> List[str]:
    para = re.sub('([。！？\?!])([^”’)\]）】])', r"\1\n\2", para)  # 单字符断句符
    para = re.sub('(\.{3,})([^”’)\]）】….])', r"\1\n\2", para)  # 英文省略号
    para = re.sub('(\…+)([^”’)\]）】….])', r"\1\n\2", para)  # 中文省略号
    para = re.sub('([。！？\?!]|\.{3,}|\…+)([”’)\]）】])([^，。！？\?….])', r'\1\2\n\3', para)
    # 如果双引号前有终止符，那么双引号才是句子的终点，把分句符\n放到双引号后，注意前面的几句都小心保留了双引号
    para = para.rstrip()  # 段尾如果有多余的\n就去掉它
    # 很多规则中会考虑分号;，但是这里我把它忽略不计，破折号、英文双引号等同样忽略，需要的再做些简单调整即可。
    sentences = para.split("\n")
    sentences = [sent.strip() for sent in sentences]
    sentences = [sent for sent in sentences if len(sent.strip()) > 0]
    return sentences


CROSS_ENTROPY = torch.nn.CrossEntropyLoss(reduction='none')


def gpt2_features(
    text: str, tokenizer: GPT2Tokenizer, model: GPT2LMHeadModel, sent_cut: Callable
) -> Tuple[List[int], List[float]]:
    # Tokenize
    input_max_length = tokenizer.model_max_length - 2
    token_ids, offsets = list(), list()
    sentences = sent_cut(text)
    for s in sentences:
        tokens = tokenizer.tokenize(s)
        ids = tokenizer.convert_tokens_to_ids(tokens)
        difference = len(token_ids) + len(ids) - input_max_length
        if difference > 0:
            ids = ids[:-difference]
        offsets.append((len(token_ids), len(token_ids) + len(ids)))  # 左开右闭
        token_ids.extend(ids)
        if difference >= 0:
            break

    input_ids = torch.tensor([tokenizer.bos_token_id] + token_ids)
    logits = model(input_ids).logits
    # Shift so that n-1 predict n
    shift_logits = logits[:-1].contiguous()
    shift_target = input_ids[1:].contiguous()
    loss = CROSS_ENTROPY(shift_logits, shift_target)

    all_probs = torch.softmax(shift_logits, dim=-1)
    sorted_ids = torch.argsort(all_probs, dim=-1, descending=True)  # stable=True
    expanded_tokens = shift_target.unsqueeze(-1).expand_as(sorted_ids)
    indices = torch.where(sorted_ids == expanded_tokens)
    rank = indices[-1]
    counter = [
        rank < 10,
        (rank >= 10) & (rank < 100),
        (rank >= 100) & (rank < 1000),
        rank >= 1000
    ]
    counter = [c.long().sum(-1).item() for c in counter]


    # compute different-level ppl
    text_ppl = loss.mean().exp().item()
    sent_ppl = list()
    for start, end in offsets:
        nll = loss[start: end].sum() / (end - start)
        sent_ppl.append(nll.exp().item())
    max_sent_ppl = max(sent_ppl)
    sent_ppl_avg = sum(sent_ppl) / len(sent_ppl)
    if len(sent_ppl) > 1:
        sent_ppl_std = torch.std(torch.tensor(sent_ppl)).item()
    else:
        sent_ppl_std = 0

    mask = torch.tensor([1] * loss.size(0))
    step_ppl = loss.cumsum(dim=-1).div(mask.cumsum(dim=-1)).exp()
    max_step_ppl = step_ppl.max(dim=-1)[0].item()
    step_ppl_avg = step_ppl.sum(dim=-1).div(loss.size(0)).item()
    if step_ppl.size(0) > 1:
        step_ppl_std = step_ppl.std().item()
    else:
        step_ppl_std = 0
    ppls = [
        text_ppl, max_sent_ppl, sent_ppl_avg, sent_ppl_std,
        max_step_ppl, step_ppl_avg, step_ppl_std
    ]
    return counter, ppls  # type: ignore


def lr_predict(
    f_gltr: List[int], f_ppl: List[float], lr_gltr: LogisticRegression, lr_ppl: LogisticRegression,
    id_to_label: List[str]
) -> List:
    x_gltr = np.asarray([f_gltr])
    gltr_label = lr_gltr.predict(x_gltr)[0]
    gltr_prob = lr_gltr.predict_proba(x_gltr)[0, gltr_label]
    x_ppl = np.asarray([f_ppl])
    ppl_label = lr_ppl.predict(x_ppl)[0]
    ppl_prob = lr_ppl.predict_proba(x_ppl)[0, ppl_label]
    return [id_to_label[gltr_label], gltr_prob, id_to_label[ppl_label], ppl_prob]


def predict_en(text: str) -> List:
    with torch.no_grad():
        feat = gpt2_features(text, TOKENIZER_EN, MODEL_EN, sent_cut_en)
    out = lr_predict(*feat, LR_GLTR_EN, LR_PPL_EN, ['Human', 'ChatGPT'])
    return out


def predict_zh(text: str) -> List:
    with torch.no_grad():
        feat = gpt2_features(text, TOKENIZER_ZH, MODEL_ZH, sent_cut_zh)
    out = lr_predict(*feat, LR_GLTR_ZH, LR_PPL_ZH, ['人类', 'ChatGPT'])
    return out


text = "zerogpt is an advanced and reliable chat GPT detector tool designed to analyze text and determine if it was generated by a human or an AI-powered language model. is zerogpt reliable ? "


def classify_is_human(text, bar=0.5000):
    predict_res = predict_en(text)
    if predict_res[1] >= bar:
        return True
    else:
        return False


if __name__ == '__main__':
    print(classify_is_human(text))

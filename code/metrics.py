from loading import get_sentence, load_model, load_from_json, save_in_json, load_matrix
import os
import numpy as np
import pandas as pd
from scipy.spatial.distance import jensenshannon

def comp_noop(a, n):
    a_prob = np.array(a)
    b_prob = np.zeros(n)
    b_prob[-1] = 1
    return jensenshannon(a_prob, b_prob)


def comp_me(a, n, ind_token, bert_tokens):
    a_prob = np.array(a)
    b_prob = np.zeros(n)
    inds = list()

    for i in range(len(bert_tokens)):
        if bert_tokens[i] == bert_tokens[ind_token]:
            inds.append(i)

    w = 1 / len(inds)
    for i in inds:
        b_prob[i] = w

    return jensenshannon(a_prob, b_prob)


def comp_back(a, n, ind_token, bert_tokens):
    a_prob = np.array(a)
    b_prob = np.zeros(n)
    inds = list()

    for i in range(len(bert_tokens)):
        if i < ind_token:
            if bert_tokens[i] == bert_tokens[ind_token]:
                inds.append(i)

    
    if len(inds) == 0:
        b_prob[-1] = 1
        return jensenshannon(a_prob[:-1], b_prob[:-1]), False
    else:
        w = 1 / len(inds)
        for i in inds:
            b_prob[i] = w

    return jensenshannon(a_prob[:-1], b_prob[:-1]), True

def comp_jsd(a, n):
    a_prob = np.array(a)
    # print(a_prob)
    b_prob = np.array([1 / n] * n)
    # print(b_prob)
    return jensenshannon(a_prob, b_prob)
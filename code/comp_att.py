import torch
import transformers
import numpy as np
import pandas as pd
from loading import load_matrix, save_in_json, load_from_json
from scipy.spatial.distance import jensenshannon


def select_sub_matrix_for_token(out_dir, id_sent, layer, head, token, bert_tokens):
    mtx = load_matrix(out_dir, id_sent, layer, head)
    sel = True
    ha = ''
    frams = list()
    index = None

    index = sel_index_by_token(token, bert_tokens)
    # print('COMP ATT', index)
    if index is None:
        token = '##' + token
        ha = '##'
        index = sel_index_by_token(token, bert_tokens)

    if index is not None:
        frams.append(mtx.iloc[index])

    j = 1
    t = '' + token
    token = t + '.' + str(j)
    while sel:
        index = sel_index_by_token(t, bert_tokens, j)
        if index is not None:
            frams.append(mtx.iloc[index])
            j = j + 1
            token = t + '.' + str(j)
        else:
            sel = False

    return frams, j, ha

def sel_index_by_token(token, bert_tokens, j=0):
    for i in range(len(bert_tokens)):
        if bert_tokens[i] == token:
            if j > 0:
                j = j - 1
                # print('j rimossa ', str(j))
            else:
                return i

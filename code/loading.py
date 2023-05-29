from transformers import  BertTokenizer, BertModel
import torch
import os
import json
import pandas as pd


def load_model(bert_path):
    #config = BertConfig.from_pretrained(bert_path, output_hidden_states=True, output_attentions=True)
    tokenizer = BertTokenizer.from_pretrained(bert_path)
    #.from_pretrained('biobert_f/biobert_v1.1_pubmed/vocab.txt')
    model = BertModel.from_pretrained(bert_path)
    return tokenizer, model

def comp_matrix(tokenizer, model, sentence):
    max_tokens=model.config.max_position_embeddings
    e = tokenizer.encode(sentence, truncation=True,max_length=max_tokens,add_special_tokens=True)
    output = model(torch.tensor([e]),output_attentions=True, output_hidden_states=True)
    attentions = output.attentions  #output[3]
    tokens = tokenizer.convert_ids_to_tokens(e)
    hidden_states = output.hidden_states #output[2]
    # print(len(hidden_states))
    # print(hidden_states[0].shape)
    return attentions, tokens, hidden_states

def create_environment_and_path(out_dir, name):
    mtx_path = os.path.join(out_dir, name)
    max_path = os.path.join(mtx_path, "max.json")
    sent_path = os.path.join(mtx_path, "sentence.json")
    if not os.path.isdir(mtx_path):
        os.mkdir(mtx_path)
    return mtx_path, max_path, sent_path

def save_in_json(name, path_file):
    out_file = open(path_file, "w", encoding="utf-8")
    json.dump(name, out_file, indent=6, ensure_ascii=False)
    out_file.close()

def save_sentence(sentence, sent_id, sent_path):
    dic_sent = dict()
    dic_sent['sentence'] = sentence
    dic_sent['sent_id'] = sent_id
    save_in_json(dic_sent, sent_path)

def save_hidden_state(dir_path, hidden, tokens):
    # dimension (1,n,768)
    for i in range(len(hidden)):
        np_h = hidden[i][0].detach().numpy()
        df = pd.DataFrame(data=np_h, index=tokens)
        file_name = dir_path + "/" + "hid_state_layer-" + str(i) + ".csv"
        df.to_csv(file_name, index=False)

def save_matrix(model,dir_path, tokens, attentions):
    dic_max = dict()
    h=model.config.num_attention_heads
    l=model.config.num_hidden_layers
    for i in range(l):
        for j in range(h):
            np_attention = attentions[i][0][j].detach().numpy()
            df = pd.DataFrame(data=np_attention, columns=tokens)
            file_name = dir_path + "/" + "att-mtx_layer-" + str(i + 1) + "_head-" + str(j + 1) + ".csv"
            df.to_csv(file_name, index=False)
            dic_max[str((i + 1, j + 1))] = df.max().max() + 0
    save_in_json(dic_max, dir_path + "/max.json")

def load(sentence, sent_id, name, tokenizer, model, out_dir):
    attentions, tokens, hidden = comp_matrix(tokenizer, model, sentence)
    mtx_path, max_path, sent_path = create_environment_and_path(out_dir, name)
    save_sentence(sentence, sent_id, sent_path)
    save_hidden_state(mtx_path, hidden, tokens)
    save_matrix(model, mtx_path, tokens, attentions)

def load_from_json(path_file):
    f = open(path_file, 'r')
    data = json.load(f)
    f.close()
    return data

def get_sentence(out_dir, name):
    dic_sent = load_from_json(os.path.join(os.path.join(out_dir, name), 'sentence.json'))
    return dic_sent['sentence']

def comp_token(tokenizer, model, sentence):
    max_tokens=model.config.max_position_embeddings
    e = tokenizer.encode(sentence,truncation=True,max_length=max_tokens, add_special_tokens=True)
    return tokenizer.convert_ids_to_tokens(e)

def get_bert_tokens(mtx_dir, tokenizer, model, sentence):
    bert_path = os.path.join(mtx_dir, "bert_tokens.json")

    if not os.path.exists(bert_path):
        bert_tokens = comp_token(tokenizer, model, sentence)
        save_in_json(bert_tokens, bert_path)
    else:
        bert_tokens = load_from_json(bert_path)

    return bert_tokens

def load_matrix(out_dir, id_sent, layer, head):
    if head is None:
        file_path = out_dir + "/" + id_sent + "/" + "layer-" + layer + ".csv"
    else:
        file_path = out_dir + "/" + id_sent + "/" + "att-mtx_layer-" + layer + "_head-" + head + ".csv"

    return pd.read_csv(file_path)

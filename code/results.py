from nltk.corpus import wordnet as wn
from nltk.corpus.reader import Synset
import os
import spacy
import nltk
import pickle
from typing import Dict, List
import numpy as np
from loading import load_model
import re
import json
import random
from scipy.stats import wilcoxon
random.seed(10)

def same_token(first: str, second: str):
    if first == second:
        return True
    if np.abs(len(first)-len(second)) > 2:
        return False
    if len(first) >= len(second):
        min_len = len(second)
    else:
        min_len = len(first)
    differences = 0
    for i in range(min_len):
        if first[i] != second[i]:
            differences += 1
    if differences <= 2:
        return True
    return False

def complete_token(index: int, tokens: List[str]): 
    token = tokens[index] 
    #print(f"Indexed {token}")
    original_index = index 
    if '##' in tokens[index]: 
        forward_index = index 
        while '##' in token: 
            previous_token: str = tokens[index - 1]
            #print(f"Precedente {previous_token} e attuale {token}")
            token = previous_token + token.replace('##', '') 
            #print(f"Completo {token}")
            index = index - 1 #scendo di indice
        if forward_index + 1>= len(tokens):
            return token
        next_token = tokens[forward_index + 1] 
        while '##' in next_token: 
            #print(f"Prossimo {next_token} e attuale {token}")
            token = token + next_token.replace('##', '')
            #print(f"Completo {token}")
            forward_index += 1 
            if forward_index + 1>= len(tokens):
                break
            next_token = tokens[forward_index +1]
    else:
        forward_index = original_index
        if forward_index + 1>= len(tokens):
            return token
        next_token = tokens[forward_index +1 ]
        while '##' in next_token:
            #print(f"Prossimo {next_token} e attuale {token}")
            token = token + next_token.replace('##', '')
            forward_index += 1  # salgo di indice
            if forward_index + 1 >= len(tokens):
                break
            next_token = tokens[forward_index + 1]
    return token

def count_duplicates(lista_totale):
    num = '[0-9]+'
    punt='[.,;:?!\/]'
    lista={}
    for i in lista_totale:
        if  not (re.match(num, i[0]) or re.match(punt, i[0]) or re.match(num, i[1]) or re.match(punt, i[1])):
            if (i[0],i[2]) not in lista:
                lista[(i[0],i[2])]=1
            else:
                lista[(i[0],i[2])]+=1
    duplicates = {}
    for i in lista:
        if lista[i]>1:
            duplicates[i]=lista[i]
    return duplicates


path_modello="bert-base-uncased" # bert-base-uncased, /home/lorenzoserina/MaterialeLuca/liberty/BERT/modelli/bluebert-base-uncased, bert-base-multilingual-cased google/bert_uncased_L-8_H-512_A-8
path_directory="" #directory of "collecter.py" results
results_path=""
nlp= spacy.load('en_core_web_lg')
folder = path_directory
reports = os.listdir(folder)
wup_syn = list()
wup_random = list()
dati={}

tokenizer, model = load_model(path_modello)

heads=model.config.num_attention_heads
layers=model.config.num_hidden_layers
for he in range(heads):
    for layer in range(layers):
        h='('+str(he)+', '+str(layer)+')'
        #print("Entro in for")
        lista_totale=[]
        dati[h]={}
        
        for i,r in enumerate(reports):
            lista=[]
            path_sent = folder + '/' + r + '/sentence.json'
            with open(path_sent) as json_file:
                data = json.load(json_file)
            dati[h][data["sent_id"]]={}
            dati[h][data["sent_id"]]["sentence"]=data["sentence"]
            path = folder + '/' + r + '/smear_results.pkl'
            f = open(path, 'rb')
            results: Dict = pickle.load(f)
            json_file = open(folder + '/' + r + '/bert_tokens.json')
            TOKENS: List = json.load(json_file)
            #dati[h][data["sent_id"]]["tokens"]=TOKENS
            #print(TOKENS)
            head: Dict = results[h] 
            for i in head.keys(): 
                token = head[i]['token'] 
                if token not in ['[CLS]', '[SEP]', '.']:
                    dati[h][data["sent_id"]][token]={}
                    #dati[h][data["sent_id"]][token]["linked"]=[]
                    clust = head[i]['clusters'] 
                    max_avg = 0
                    max_key = 0
                    for k, c in clust.items(): #numero cluster, lista di tuple
                        avg = np.average([e[1] for e in c]) #media delle correlazioni
                        if avg > max_avg:#prendo il cluster con la media più alta
                            max_key = k
                            max_avg = avg
                    last_cluster = clust[max_key]
                    dati[h][data["sent_id"]][token]["cluster"]={}
                    dati[h][data["sent_id"]][token]["cluster"]["valori"]=last_cluster
                    dati[h][data["sent_id"]][token]["cluster"]["media_pesi"]=max_avg
                    dati[h][data["sent_id"]][token]["cluster"]["numero_token"]=len(last_cluster)

                    pattern = '.+\.[0-9]+'
                    
                    for e in last_cluster:
                        result = re.match(pattern, e[0])
                        if result:
                            occurrence = int(e[0].split('.')[-1]) 
                            if '..' in e[0]:
                                linked_token = '.'
                            else:#è un token
                                linked_token = e[0].split('.')[0]
                            possibilities = [j for j in range(len(TOKENS)) if TOKENS[j] == linked_token]
                            linked_index = possibilities[occurrence]
                        else:
                            linked_token = e[0] 
                            linked_index = TOKENS.index(linked_token) 
                        if not same_token(token, linked_token) and linked_token not in ['[CLS]', '[SEP]', '.']: 
                            linked_token = complete_token(linked_index, TOKENS)
                            res = nlp(linked_token)
                           
                            linked_token=res[0].lemma_
                            linked_label=""
                            if len(res.ents)>0:
                                linked_label=res.ents[0].label_
                            actual_index = i 
                            
                           
                            actual_token = complete_token(actual_index, TOKENS)
                            res = nlp(actual_token)
                            actual_token=res[0].lemma_
                            actual_label=""
                            if len(res.ents)>0:
                                actual_label=res.ents[0].label_
                            print(f"actual token lemma: {actual_token}")
                            wordnet_token: List[Synset] = wn.synsets(actual_token)#, lang='spa')
                            print(f"wordnet token: {wordnet_token}")
                            
                            wordnet_linked_token: List[Synset] = wn.synsets(linked_token)#, lang='spa')
                            print(f"wordnet linked token: {wordnet_linked_token}")
                            max_similarity = 0
                            max_wup = 0
                            for s in wordnet_token:
                                for t in wordnet_linked_token:
                                    sim = s.path_similarity(t)
                                    if sim > max_similarity:
                                        max_similarity = sim
                                    wup = s.wup_similarity(t)
                                    if wup > max_wup:
                                        max_wup = wup
                            if max_wup == 0:
                                max_wup = 0.01
                            valori=(actual_token, actual_label, linked_token, linked_label,str(f"Wordnet: {max_wup}"), str(f"Smear: {e[1]}"))
                            lista.append(valori)
                            lista_totale.append(valori)
                            wup_syn.append(max_wup)

            lista.sort(key=lambda x: x[2], reverse=True)
            dati[h][data["sent_id"]]["risultati"]=lista
        lista_totale.sort(key=lambda x: x[2], reverse=True)
        dati[h]["risultati_totali"]=lista_totale
        dati[h]["duplicati"]=str(count_duplicates(lista_totale)) 
        dati[h]["media_simil"]=np.mean(wup_syn)

                    
with open(results_path , "w", encoding="utf-8") as outfile:
    json.dump(dati, outfile, indent=4, ensure_ascii=False)

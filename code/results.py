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

# nltk.download('wordnet')
# nltk.download('omw-1.4')

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

def complete_token(index: int, tokens: List[str]): #prendo indice e lista di token
    token = tokens[index] #prendo il token relativo all'indice
    #print(f"Indexed {token}")
    original_index = index #ricordo l'indice originale
    if '##' in tokens[index]: #se il token è una sottostringa di un token più lungo
        forward_index = index #ricordo l'indice
        while '##' in token: #finché il token è una sottostringa
            previous_token: str = tokens[index - 1]#prendo il token precedente
            #print(f"Precedente {previous_token} e attuale {token}")
            token = previous_token + token.replace('##', '') #concateno il token precedente al token senza ##
            #print(f"Completo {token}")
            index = index - 1 #scendo di indice
        if forward_index + 1>= len(tokens):
            return token
        next_token = tokens[forward_index + 1] #prendo il token successivo
        while '##' in next_token: #finché il token è una sottostringa
            #print(f"Prossimo {next_token} e attuale {token}")
            token = token + next_token.replace('##', '')#concateno il token successivo al token senza ##
            #print(f"Completo {token}")
            forward_index += 1 #salgo di indice
            if forward_index + 1>= len(tokens):
                break
            next_token = tokens[forward_index +1]#TODO: +1 modificato #prendo il token successivo
    else:#se il token non è una sottostringa
        forward_index = original_index
        if forward_index + 1>= len(tokens):
            return token
        next_token = tokens[forward_index +1 ]#TODO: +1 #prendo il token successivo
        while '##' in next_token:# finché il token è una sottostringa
            #print(f"Prossimo {next_token} e attuale {token}")
            token = token + next_token.replace('##', '')#concateno il token successivo al token senza ##
            #print(f"Completo {token}")            
            forward_index += 1  # salgo di indice
            if forward_index + 1 >= len(tokens):
                break
            next_token = tokens[forward_index + 1]#prendo il token successivo
    return token#ritorno il token completo

def count_duplicates(lista_totale):
    num = '[0-9]+'#pattern per le occorrenze
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


#INIZIO

path_modello="bert-base-uncased"# bert-base-uncased, /home/lorenzoserina/MaterialeLuca/liberty/BERT/modelli/bluebert-base-uncased, bert-base-multilingual-cased google/bert_uncased_L-8_H-512_A-8
path_directory="/home/lorenzoserina/MaterialeLuca/liberty/BERT/dati/risultati_trial/contesto/cased/synonyms"
results_path="/home/lorenzoserina/MaterialeLuca/liberty/BERT/dati/risultati_results/contesto/cased/synonyms.json"
#for path_dati in os.listdir(path_directory):

    #nlp = spacy.load('es_core_news_lg')
nlp= spacy.load('en_core_web_lg')
#folder = '/home/lorenzoserina/MaterialeLuca/liberty/dati/news_esp_results/'
folder = path_directory#+path_dati
#print("Inizio")
reports = os.listdir(folder)
wup_syn = list()
wup_random = list()
#lista=[]
dati={}
# heads=['(2, 6)', '(3, 0)', '(11, 8)', '(1, 11)','(10, 10)'] #[2,8],[10,1],[2,11],[11,4],[7,0],  bert-uncased [2,6],[3,0],[11,8],[1,11],[10,10]   BETO online [2,7],[3,2],[11,5],[11,1],[4,10]
# for h in heads:

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
            # if i==10:
            #     break
            print(r)
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
            head: Dict = results[h] #(2, 6); (11, 8); (3, 0); (1, 11)
            #print(head.keys())
            for i in head.keys(): #il numero dei tokens
                token = head[i]['token'] #il token, string
                if token not in ['[CLS]', '[SEP]', '.']:
                    dati[h][data["sent_id"]][token]={}
                    #dati[h][data["sent_id"]][token]["linked"]=[]
                    clust = head[i]['clusters'] #dizionario di cluster, lista di tuple (token, correlazione)
                    #print(clust)
                    max_avg = 0
                    max_key = 0
                    for k, c in clust.items(): #numero cluster, lista di tuple
                        avg = np.average([e[1] for e in c]) #media delle correlazioni
                        if avg > max_avg:#prendo il cluster con la media più alta
                            max_key = k
                            max_avg = avg
                    last_cluster = clust[max_key]#prendo il cluster con la media più alta
                    #dati[h][data["sent_id"]][token]["cluster_val"]={}
                    dati[h][data["sent_id"]][token]["cluster"]={}
                    dati[h][data["sent_id"]][token]["cluster"]["valori"]=last_cluster
                    dati[h][data["sent_id"]][token]["cluster"]["media_pesi"]=max_avg
                    dati[h][data["sent_id"]][token]["cluster"]["numero_token"]=len(last_cluster)

                    pattern = '.+\.[0-9]+'#pattern per le occorrenze
                    
                    for e in last_cluster:#prendo le tuple del cluster
                        result = re.match(pattern, e[0])#vedo quanti match ci sono
                        if result:
                            occurrence = int(e[0].split('.')[-1])  #prendo l'occorrenza
                            if '..' in e[0]:#è un punto
                                linked_token = '.'
                            else:#è un token
                                linked_token = e[0].split('.')[0]
                            possibilities = [j for j in range(len(TOKENS)) if TOKENS[j] == linked_token]#prendo tutti i token uguali a quello che sto analizzando
                            linked_index = possibilities[occurrence]#scelgo quello relativo all'occorrenza
                        else:#non è un token con più occorrenze
                            linked_token = e[0] #prendo il token
                            linked_index = TOKENS.index(linked_token) #prendo l'indice del token
                        if not same_token(token, linked_token) and linked_token not in ['[CLS]', '[SEP]', '.']: #se token in iziale e quello linkato non sono uguali e non sono punti o token speciali
                            linked_token = complete_token(linked_index, TOKENS) #controllo se si può unire il token con quello precedente
                            #print(f"linked token: {linked_token}")
                            res = nlp(linked_token)
                            #print(res)
                            #print(res.ents)
                            linked_token=res[0].lemma_
                            linked_label=""
                            if len(res.ents)>0:
                                linked_label=res.ents[0].label_
                            #print(f"linked token lemma: {linked_token}")
                            actual_index = i 
                            
                            #if actual_index in head.keys():
                                #print(actual_index,TOKENS[actual_index])
                            actual_token = complete_token(actual_index, TOKENS)
                            #print(f"actual token: {actual_token}")
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
                            #print(actual_token, linked_token, max_wup) fai una lista con queste cose
                            if max_wup == 0:
                                max_wup = 0.01
                            valori=(actual_token, actual_label, linked_token, linked_label,str(f"Wordnet: {max_wup}"), str(f"Smear: {e[1]}"))
                            lista.append(valori)
                            lista_totale.append(valori)
                            wup_syn.append(max_wup)
                            # one = TOKENS[random.randint(1, len(TOKENS)-1)]
                            # two = TOKENS[random.randint(1, len(TOKENS) - 1)]
                            # while one == two:
                            #     two = TOKENS[random.randint(1, len(TOKENS) - 1)]
                            # wordnet_token: List[Synset] = wn.synsets(one)
                            # wordnet_linked_token: List[Synset] = wn.synsets(two)
                            # max_wup = 0
                            # for s in wordnet_token:
                            #     for t in wordnet_linked_token:
                            #         wup = s.wup_similarity(t)
                            #         if wup > max_wup:
                            #             max_wup = wup
                            # # print('WUP', one, two, max_wup)
                            # if max_wup == 0:
                            #     max_wup = 0.01
                            # wup_random.append(max_wup)
            lista.sort(key=lambda x: x[2], reverse=True)
            dati[h][data["sent_id"]]["risultati"]=lista
        lista_totale.sort(key=lambda x: x[2], reverse=True)
        dati[h]["risultati_totali"]=lista_totale
        dati[h]["duplicati"]=str(count_duplicates(lista_totale)) 
        dati[h]["media_simil"]=np.mean(wup_syn)
    # dati[h]["duplicati"]={}
    # dati[h]["duplicati"]=str(count_duplicates(lista_totale))
    #test = wilcoxon(wup_syn, wup_random)
    #dati[h]["wilcoxon"]=str(test)
    
    # with open("/home/user/MaterialeLuca/liberty/risultati.json", "w") as outfile:
    #     json.dump(dati, outfile, indent=4)
    #break

                    
with open(results_path , "w", encoding="utf-8") as outfile:
    json.dump(dati, outfile, indent=4, ensure_ascii=False)
        #print(lista)
        # lista.sort(key=lambda x: x[2], reverse=True)
        # #print(lista)
        # test = wilcoxon(wup_syn, wup_random)
        # print(test)

    #Conta le coppie che appaiono più volte in tutte le head                                                        FATTO
    #Togli coppie con lo stesso token                                                                               FATTO
    #Trovare un modo per trovare i termini medici                                                                   MANCA   
    #Vedere in smear results che peso hanno queste coppie, se c'è correlazione tra similiarità alta e peso alto     FATTO
    #Leva coppie con numeri e punteggiatura                                                                         FATTO
    #Calcola media delle similiarità per ogni head                                                                  FATTO                                     
    #Lista per singolo documento(dizionario: referto - lista - pesi del cluster - numero di elementi nel cluster )  FATTO
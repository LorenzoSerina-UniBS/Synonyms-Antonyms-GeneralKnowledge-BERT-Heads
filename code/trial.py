import os
import pickle
from loading import load
from smear import smear
from loading import load_model
import spacy
from spacy import displacy
from pathlib import Path

# path_dati="/home/lorenzoserina/MaterialeLuca/liberty/dati/nazioni_capitali_dict_single.pkl"
# path_modello="roberta-base"#"dccuchile/bert-base-spanish-wwm-cased", bert-base-uncased
# prova_path="/home/lorenzoserina/MaterialeLuca/liberty/dati/roberta_nazioni_capitali_dict_single"

path_directory="/home/lorenzoserina/MaterialeLuca/liberty/BERT/dati/datasets/contesto/synonyms.pkl"
path_modello="/home/lorenzoserina/MaterialeLuca/liberty/BERT/modelli/bluebert-base-uncased"# bert-base-uncased, bert-base-multilingual-cased, /home/lorenzoserina/MaterialeLuca/liberty/BERT/modelli/bluebert-base-uncased, "google/bert_uncased_L-8_H-512_A-8"
prova_path_base="/home/lorenzoserina/MaterialeLuca/liberty/BERT/dati/risultati_trial/contesto/blue/synonyms/"

#for path_dati in os.listdir(path_directory):
prova_path=prova_path_base#+path_dati
os.mkdir(prova_path)
path_dati=path_directory#+"/"+path_dati
f = open(path_dati, 'rb')
#nlp = spacy.load('en_core_web_lg')
datasets = pickle.load(f)
dataset=datasets["text"]
print("Dataset caricato")

print(len(datasets))
tokenizer, model = load_model(path_modello)
heads=model.config.num_attention_heads
layers=model.config.num_hidden_layers
max_tokens=model.config.max_position_embeddings

for i in range(len(dataset)):
    print(f"Documento {i}/{len(dataset)}")
    document = dataset[i]
    #print(document)
    
    #doc = nlp(document)
    #svg = displacy.render(doc, style="dep")
    #print(f"Svg:{svg}")
    #output_path = Path("sentence.svg")
    #output_path.open("w", encoding="utf-8").write(svg)
    if len(document) == 0:
        continue
    if document[-1] != '.':
        document += '.'
    tokenized = tokenizer.encode(document, truncation=True,max_length=max_tokens, add_special_tokens=True)
    tokens = tokenizer.convert_ids_to_tokens(tokenized)
    points = [j for j in range(len(tokens)) if tokens[j] == '.']
    start = 0
    sentences = list()
    for p in points:
        sentences.append((start, p))
        start = p + 1
    if points[-1] < len(tokens) - 1:
        sentences.append((start, len(tokens)-1))
    id = 0
    sentence_dict = dict()
    for low, high in sentences:
        x = dict()
        sentence_tokens = [tokens[i] for i in range(low, high+1)]
        x['tokens'] = tokens
        x['low'] = low
        x['high'] = high
        # x = {'text': s.text, 'tokens': tokens}
        sentence_dict[id] = x
        id += 1
    os.mkdir(prova_path+'/train-'+str(i))
    f = open(prova_path+'/train-'+str(i)+'/sentence_tokens.pkl', 'wb')
    pickle.dump(sentence_dict, f)
    f.close()
    load(document, 'train-' + str(i), 'train-' + str(i), tokenizer, model, prova_path)
    smear_heads = dict()
    #heads=[[2,6],[3,0],[11,8],[1,11],[10,10]]#bert-uncased[[2,6],[3,0],[11,8],[1,11],[10,10]] beto [2,8],[10,1],[2,11],[11,4],[7,0] beto_online [[2,7],[3,2],[11,5],[11,1],[4,10]]
    for h in range(heads):
        for l in range(layers):
    #for h in heads:
            layer=l#[0]
            head=h#h[1]
            print(f"{layer}, {head}")
            #print(layer, head)
            smear_results = dict()
            for j in range(len(tokens)):
                #print(j, len(tokens))
                t = tokens[j]
                
                means, clust = smear(head=str(head + 1), layer=str(layer + 1), name='train-' + str(i), token=t,
                                        out_dir=prova_path,
                                        tokenizer=tokenizer, model=model, verbose=False, id_token=0)
                smear_results[j] = {'token': t, 'clusters': clust}
            smear_heads['('+str(layer)+', '+str(head)+')'] = smear_results
    f = open(prova_path+'/train-'+str(i)+'/smear_results.pkl', 'wb')
    pickle.dump(smear_heads, f)
    f.close()

#[(2, 6, 0.5278422980197197, 0.7889637686084245), (3, 0, 0.575731358834471, 0.7601693575249139), (11, 8, 0.5793305489361382, 0.8314534542996405), (1, 11, 0.6183630101777527, 0.8241823870782435), (10, 10, 0.6421073110425511, 0.6575927758820771)]
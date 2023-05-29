import os
import pickle
from loading import load
from linker import linker
from loading import load_model
import spacy
from spacy import displacy
from pathlib import Path



path_directory="" #dataset pickle
path_modello=""# bert-base-uncased, bert-base-multilingual-cased, bluebert-base-uncased, "google/bert_uncased_L-8_H-512_A-8"
prova_path_base="" #path results

prova_path=prova_path_base
os.mkdir(prova_path)
path_dati=path_directory
f = open(path_dati, 'rb')
datasets = pickle.load(f)
dataset=datasets["text"]

tokenizer, model = load_model(path_modello)
heads=model.config.num_attention_heads
layers=model.config.num_hidden_layers
max_tokens=model.config.max_position_embeddings

for i in range(len(dataset)):
    print(f"Documento {i}/{len(dataset)}")
    document = dataset[i]
    
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
        sentence_dict[id] = x
        id += 1
    os.mkdir(prova_path+'/train-'+str(i))
    f = open(prova_path+'/train-'+str(i)+'/sentence_tokens.pkl', 'wb')
    pickle.dump(sentence_dict, f)
    f.close()
    load(document, 'train-' + str(i), 'train-' + str(i), tokenizer, model, prova_path)
    smear_heads = dict()
    for h in range(heads):
        for l in range(layers):
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


from loading import comp_matrix, load_model
import pickle
from typing import List
import numpy as np
from metrics import comp_me, comp_noop, comp_back, comp_jsd
from transformers import BertTokenizer, BertModel, RobertaTokenizer, RobertaModel
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import json

def heatmap(data, row_labels, ax=None, path=None):
    fig, ax = plt.subplots()
    im = ax.imshow(data)

    # Show all ticks and label them with the respective list entries
    ax.set_xticks(np.arange(len(row_labels)), labels=row_labels)
    ax.set_yticks(np.arange(len(row_labels)), labels=row_labels)

    # Rotate the tick labels and set their alignment.
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
            rotation_mode="anchor")

    # Loop over data dimensions and create text annotations.
    for i in range(len(row_labels)):
        for j in range(len(row_labels)):
            text = ax.text(j, i, data[i, j],
                        ha="center", va="center", color="w")

    ax.set_title("Head attention heatmap")
    fig.tight_layout()
    plt.savefig(path)
    
print("Inizio")
f = open('/home/lorenzoserina/MaterialeLuca/liberty/BERT/dati/datasets/medicine.pkl', 'rb')
dataset = pickle.load(f)
dataset=dataset["text"]
tokenizer, model=load_model("bert-base-multilingual-cased")#bert-base-uncased, google/bert_uncased_L-8_H-512_A-8 , bert-base-multilingual-cased  /home/lorenzoserina/MaterialeLuca/liberty/BERT/modelli/bluebert-base-uncased/
occorrenze={}
oc_back={}
heads=model.config.num_attention_heads
layers=model.config.num_hidden_layers
max_tokens=model.config.max_position_embeddings
tuple_tot=[]
diag_dict={}
jsd_dict={}
for i in range(layers):
    for j in range(heads):
        occorrenze[(i,j)]=0
        oc_back[(i,j)]=0
        diag_dict[(i,j)]=0
        jsd_dict[(i,j)]=0

for i in range(len(dataset)):
    #print("Analizzo nota", i)
    tuples = list()
    text = dataset[i]
    #text = note.text
    tokenized = tokenizer.encode(text, add_special_tokens=True, truncation=True, max_length=max_tokens)
    tokens = tokenizer.convert_ids_to_tokens(tokenized)
    if len(tokens) > max_tokens:
        continue
    attention_tensor = comp_matrix(tokenizer, model, text)[0]
    #print(comp_matrix(tokenizer, model, text))
    for j in range(layers):
        layer_tensor = attention_tensor[j][0]
        for k in range(heads):
            #print("Analizzo head", k, "layer", j)
            head_tensor = layer_tensor[k].detach().numpy()
    # L'attention data si trova sulle righe
    # sum = np.sum(head_tensor, axis=1)
            me_metrics = list()
            noop_metrics = list()
            back_metrics = list()
            jsd_metrics = list()
            for w in range(len(head_tensor)):
                attention_vector = head_tensor[w]
                argmax = np.argmax(attention_vector)
                me = comp_me(attention_vector, len(head_tensor), w, tokens)
                
                noop = comp_noop(attention_vector, len(head_tensor))
                back, check=comp_back(attention_vector, len(head_tensor), w, tokens)
                jsd=comp_jsd(attention_vector, len(head_tensor))
                if check==True:
                    back_metrics.append(back)
                noop_metrics.append(noop)
                me_metrics.append(me)
                jsd_metrics.append(jsd)
            avg_jsd=np.average(jsd_metrics)
            avg = np.average(me_metrics)
            diag_dict[(j,k)]+=avg
            jsd_dict[(j,k)]+=avg_jsd
            avg_noop = np.average(noop_metrics)
            if len(back_metrics)==0:
                avg_back=1
            else:
                avg_back = np.average(back_metrics)
            tuples.append((j, k, avg, avg_noop, avg_back))
    # heatmap(np.array([float(f'{t[2]:.2f}') for t in tuples]).reshape(12, 12), [str(i) for i in range(12)], path=f"/home/lorenzoserina/MaterialeLuca/liberty/BERT/diagonal_metric/heatmap_diag_{i}.png")
    # heatmap(np.array([float(f'{t[4]:.2f}') for t in tuples]).reshape(12, 12), [str(i) for i in range(12)], path=f"/home/lorenzoserina/MaterialeLuca/liberty/BERT/diagonal_metric/heatmap_back_{i}.png")
            
            
    
    tuple_tot.append(tuples)
    tuples.sort(key=lambda a: a[2])
    
    for d in tuples[:10]:
        occorrenze[(d[0],d[1])]+=1
        
    tuples.sort(key=lambda a: a[4])
    for b in tuples[:10]:

        oc_back[(b[0],b[1])]+=1

# vettore=[]
# for i in np.mean(tuple_tot, axis=0):
#     vettore.append(tuple(i))
  
#Create an array of the mean of each value of diag_dict
for i in range(layers):
    for j in range(heads):
        diag_dict[(i,j)]=diag_dict[(i,j)]/len(dataset)
        jsd_dict[(i,j)]=jsd_dict[(i,j)]/len(dataset)
        
        
diag_dict = {str(k): v for k, v in diag_dict.items()}
jsd_dict = {str(k): v for k, v in jsd_dict.items()}
#Create an heatmap of the mean of each value of diag_dict
#heatmap(np.array([float(f'{t:.2f}') for t in diag_dict.values()]).reshape(12, 12), [str(i) for i in range(12)], path=f"/home/lorenzoserina/MaterialeLuca/liberty/BERT/diagonal_metric/heatmap_unc.png")

        
#Sort the dictionary by value descending
diag_dict = {k: v for k, v in sorted(diag_dict.items(), key=lambda item: item[1])}
jsd_dict = {k: v for k, v in sorted(jsd_dict.items(), key=lambda item: item[1])}
#Save the dictionary in a json file
with open('/home/lorenzoserina/MaterialeLuca/liberty/BERT/diagonal_metric/diag_medicine_multi.json', 'w') as fp:
    json.dump(diag_dict, fp)
#with open('/home/lorenzoserina/MaterialeLuca/liberty/BERT/diagonal_metric/jsd_17.json', 'w') as fp:
    #json.dump(jsd_dict, fp)
    
    
print("Diagonal metric")
for i in range(layers):
    for j in range(heads):
        if occorrenze[(i,j)]!=0:
            print("Layer: "+str(i)+" Head: "+str(j)+" Occorrenze: "+str(occorrenze[(i,j)]))
            
print("Back metric")
for i in range(layers):
    for j in range(heads):
        if oc_back[(i,j)]!=0:
            print("Layer: "+str(i)+" Head: "+str(j)+" Occorrenze: "+str(oc_back[(i,j)]))
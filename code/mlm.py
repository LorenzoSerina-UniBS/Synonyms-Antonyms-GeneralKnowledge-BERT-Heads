from transformers import pipeline
from tqdm import tqdm
from difflib import SequenceMatcher
import json

def is_number(s):
    try:
        float(s)
        return True
    except ValueError:
        return False
    
def read_json(filepath):
  with open(filepath) as f:
        return json.load(f)
syno=read_json("/home/lorenzoserina/MaterialeLuca/liberty/BERT/dati/synonyms_wn.json")
print(len(syno))
dataset_syno={}
prompt="[Y] has the same meaning as [X]."# if you would ask me to describe it i could say that it is [Y], or, in other words, it is [X].
pipe=pipeline("fill-mask", model="bert-base-uncased", device=0)
batch_mask={}
corrette=0
for k in syno:
  parola=(k.split(":"))[0]
  #print(type(syno[k]))
  contrari= syno[k].replace("|",";").split(";")
  # chiavi=list(dataset_syno.keys())
  # if len(chiavi)>100:
  #   break
  for c in contrari:
    if len(parola.replace("-"," ").split())==1 and len(c.replace("-"," ").split())==1 and not is_number(parola) and not is_number(c):
      if SequenceMatcher(None, parola.lower(), c.lower()).ratio()<0.7:
        frase=prompt.replace("[Y]", parola).replace("[X]","[MASK]")
        batch_mask[f"{parola}, {c}"]=frase
batch=[]
for k in tqdm(batch_mask):
  pred=pipe(batch_mask[k])
  for p in pred:
    for k in batch_mask:
      parola, c= k.split(", ")
      if batch_mask[k].lower()== p["sequence"]:
          corrette+=1
          if f"{parola.lower()}, {c.lower()}" not in dataset_syno and f"{c.lower()}, {parola.lower()}" not in dataset_syno:
            #if pred["score"]>0.1:
            dataset_syno[f"{parola}, {c}"]=p["score"]
            #print(len(list(dataset_syno.keys())))
print(corrette/len(syno))
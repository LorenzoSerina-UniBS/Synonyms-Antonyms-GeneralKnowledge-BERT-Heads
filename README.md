# Synonyms-Antonyms-GeneralKnowledge-BERT-Heads
Github repository for the paper "Synonyms, Antonyms and General Knowledge in BERT Heads".

In order to reproduce the experiments, ypu have to follow these steps:
1. Create a dataset with only sentences in a column ["Text"] in pickle format.
2. Run the *collector.py* algorithm.
3. Run the *results.py* algorithm on the results of *collector.py".
4. Run *self_metric.py* to obtain the important heads.
5. Collect all the results and use *BERT_Knowledge_utils.ipynb* to obtain the results.

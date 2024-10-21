#!/usr/bin/env python
# coding: utf-8

import os
from os.path import join
import json
from itertools import chain
from setfit import SetFitModel
import random
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import pandas as pd
from tqdm import tqdm

## DIRS AND PATHS
project_dir = join('/work', 'SMI-CM')
data_dir = join(project_dir, 'data')
modelling_dir = join(project_dir, 'modelling')
modules_dir = join(modelling_dir, 'modules')
logs_dir = join(modelling_dir, 'logs')
model_dir = join(modelling_dir, 'models')
output_dir = join(modelling_dir, 'output')
model_dir = join(modelling_dir, 'models')

os.makedirs(output_dir, exist_ok = True)

data_p = join(data_dir, 'rep-speech_model-data.json')
model_p = join(model_dir, 'rep_speech_model')

## READ DATA
with open(data_p, 'r') as f:
    rep_speech_data = json.load(f)


## DATA USED FOR TRAINING
seed_n = 4102
random.seed(seed_n)

n_texts = 25 # number of texts used
data_used = random.sample(rep_speech_data, n_texts)
test_eval_prop = 0.3 # proportion of docs used for test/eval
ids = [rep_speech.get('id') for rep_speech in data_used] # doc ids used
train_ids_used, temp_ids = train_test_split(ids, test_size = test_eval_prop, random_state = seed_n) # ids used for training
test_ids_used, eval_ids_used = train_test_split(temp_ids, test_size = 0.5, random_state = seed_n) # ids used for test and eval


## GENERATE TEST DATA

# Test ids
test_ids = [rep_speech.get('id') for rep_speech in rep_speech_data if rep_speech.get('id') not in train_ids_used + eval_ids_used]

# Test sentences and negatives
test_sentences = list(chain(*[rep.get('rep_speech') for rep in rep_speech_data if rep.get('id') in test_ids]))
test_negatives = list(chain(*[rep.get('negatives') for rep in rep_speech_data if rep.get('id') in test_ids]))

# Convert to data frame
test_data = pd.DataFrame(
    {
        'sentence': test_sentences + test_negatives,
        'label': ["reported speech"] * len(test_sentences) + ["not reported speech"] * len(test_negatives)})


## EVALUATE MODEL
# Load rep-speech model
rep_speech_model = SetFitModel.from_pretrained(model_p)

# Predictions
preds = rep_speech_model.predict(list(test_data['sentence']))

# Classification report
class_report = classification_report(list(test_data['label']), preds, output_dict=True)

# Save classification report
with open(join(output_dir, 'model_eval.json'), 'w') as f:
    json.dump(class_report, f)


## EVALUATE PROBABILITY THRESHOLD
### Brug evt. nedenstående funktion til at tage udgangspunkt i.
def predict_thres(texts, model = rep_speech_model, threshold = 0.50):

    labels = sorted(model.labels) # predict_proba output matches sorted labels (index 0: not reported speech, index 1: reported speech)
    probas = model.predict_proba(texts).tolist()

    preds = []

    for probs in (probas):

        max_prop = max(probs)

        index_max = probs.index(max_prop)

        if index_max == 0:
            label_out = "not reported speech"
        elif index_max == 1:
            if max_prop < threshold:
                label_out = "not reported speech"
            else:
                label_out = "reported speech"
        
        preds.append(label_out)
    
    return(preds)

### data til at evaluere på
test_data_sentences = list(test_data['sentence'])
test_data_labels = list(test_data['label'])

## NEDENSTÅENDE SKAL ITERERES OVER FORSKELLIGE THRESHOLD (0.5-0.9 med 0.05 stigninger)
results = []

threshold = [x / 100.0 for x in range(50, 90, 5)]
for x in tqdm(threshold):
    preds = predict_thres(test_data_sentences, threshold=x)
    class_report = classification_report(test_data_labels, preds, output_dict=True) 
    class_report['threshold'] = x
    results.append(class_report)

# Save classification report
with open(join(output_dir, 'model_threshold.json'), 'w') as f:
    json.dump(results, f)
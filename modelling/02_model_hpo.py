#!/usr/bin/env python
# coding: utf-8

import sys
import os
from os.path import join



import json
from itertools import chain
from setfit import SetFitModel, Trainer
from datasets import Dataset
import random
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_recall_fscore_support, accuracy_score
import torch
import logging
import optuna # backend for setfit HPO
from optuna import Trial # backend for setfit HPO

## DIRS AND PATHS
project_dir = join('/work', 'SMI-CM')
data_dir = join(project_dir, 'data')
modelling_dir = join(project_dir, 'modelling')
modules_dir = join(modelling_dir, 'modules')
logs_dir = join(modelling_dir, 'logs')
output_dir = join(modelling_dir, 'output')
model_dir = join(modelling_dir, 'models')

data_p = join(data_dir, 'rep-speech_model-data.json')


## LOGGING SETUP
logging.basicConfig(
    filename=join(logs_dir, 'hpo.log'),  # Log file name
    filemode='a',        # Append mode
    format='%(asctime)s - %(levelname)s - %(message)s',  # Log format
    datefmt='%Y-%m-%d %H:%M:%S',
    level=logging.INFO   # Log level
)


## READ DATA
with open(data_p, 'r') as f:
    rep_speech_data = json.load(f)


## SELECT DATA FOR TRAINING
seed_n = 4102
random.seed(seed_n)

### determine number of texts to use (will be further split into training, eval, test)
### NOTE: Number of texts; not sentences! (55 texts in total)
n_texts = 25
data_use = random.sample(rep_speech_data, n_texts)


## TRAIN/TEST SPLIT (on whole documents)

### ids for training and test data
test_eval_prop = 0.3 # proportion of docs used for test/eval

### split data based on ids
ids = [rep_speech.get('id') for rep_speech in data_use]
train_ids, temp_ids = train_test_split(ids, test_size = test_eval_prop, random_state = seed_n)
test_ids, eval_ids = train_test_split(temp_ids, test_size = 0.5, random_state = seed_n)

#### train
train_sentences = list(chain(*[rep.get('rep_speech') for rep in data_use if rep.get('id') in train_ids]))
train_negatives = list(chain(*[rep.get('negatives') for rep in data_use if rep.get('id') in train_ids]))

#### eval
eval_sentences = list(chain(*[rep.get('rep_speech') for rep in data_use if rep.get('id') in eval_ids]))
eval_negatives = list(chain(*[rep.get('negatives') for rep in data_use if rep.get('id') in eval_ids]))

#### test
test_sentences = list(chain(*[rep.get('rep_speech') for rep in data_use if rep.get('id') in test_ids]))
test_negatives = list(chain(*[rep.get('negatives') for rep in data_use if rep.get('id') in test_ids]))

### Convert to Dataset class (using text labels)
train_data = Dataset.from_dict({'text': train_sentences + train_negatives, 'label': ["reported speech"] * len(train_sentences) + ["not reported speech"] * len(train_negatives)})
eval_data = Dataset.from_dict({'text': eval_sentences + eval_negatives, 'label': ["reported speech"] * len(eval_sentences) + ["not reported speech"] * len(eval_negatives)})
test_data = Dataset.from_dict({'text': test_sentences + test_negatives, 'label': ["reported speech"] * len(test_sentences) + ["not reported speech"] * len(test_negatives)})


## MODEL SETUP
### Model name
model_name = 'intfloat/multilingual-e5-large'
### Set device
device = "cuda" if torch.cuda.is_available() else "cpu"

### Function to compute additional metrics
def compute_metrics(y_pred, y_test):
    accuracy = accuracy_score(y_test, y_pred)
    precision, recall, f1, _ = precision_recall_fscore_support(y_test, y_pred, average='binary')
    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
    }


## HYPERPARAMETER OPTIMIZATION
### Functions for HP-opt
#### model init
def model_init(params):
    params = params or {}
    max_iter = params.get("max_iter", 100)
    solver = params.get("solver", "liblinear")
    params = {
        "head_params": {
            "max_iter": max_iter,
            "solver": solver,
        }
    }

    model_use = SetFitModel.from_pretrained(
        model_name,
        labels=["not reported speech","reported speech"],
        **params).to(device)

    return(model_use)

#### HP space
def hp_space(trial):
    return {
        "body_learning_rate": trial.suggest_float("body_learning_rate", 1e-6, 1e-2, log=True),
        "num_epochs": trial.suggest_int("num_epochs", 2, 6),
        "batch_size": trial.suggest_categorical("batch_size", [16, 32, 48, 64]),
        "max_iter": trial.suggest_int("max_iter", 50, 300),
        "solver": trial.suggest_categorical("solver", ["lbfgs", "liblinear"]),
    }

# Setfit Trainer
trainer = Trainer(
    train_dataset=train_data,
    eval_dataset=eval_data,
    model_init = model_init,
    metric=compute_metrics,
    column_mapping={"text": "text", "label": "label"}  # Map dataset columns to text/label expected by trainer
)

# Run HP opt
logging.info('Running hyperparemeter-optimization...')
best_run = trainer.hyperparameter_search(direction="maximize", hp_space=hp_space, n_trials=15)

with open(join(output_dir, 'hpo_bestrun.txt'), 'w') as f:
    f.write(str(best_run))

logging.info(f'Best run: {json.dumps(best_run.hyperparameters, indent=4)}') # might throw error

## NOTE: Results from HPO copied manually from nohup log hpo/nohup_hpo.out to output/hpo-results.json
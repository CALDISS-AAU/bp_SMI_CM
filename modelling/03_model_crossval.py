#!/usr/bin/env python
# coding: utf-8

from os.path import join
import json
from itertools import chain
from setfit import SetFitModel, Trainer, TrainingArguments
from datasets import Dataset
import random
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.metrics import precision_recall_fscore_support, accuracy_score
import torch
import logging
import numpy as np


## DIRS AND PATHS
project_dir = join('/work', 'SMI-CM')
data_dir = join(project_dir, 'data')
modelling_dir = join(project_dir, 'modelling')
modules_dir = join(modelling_dir, 'modules')
logs_dir = join(modelling_dir, 'logs')
model_dir = join(modelling_dir, 'models')

data_p = join(data_dir, 'rep-speech_model-data.json')


## LOGGING SETUP
logging.basicConfig(
    filename=join(logs_dir, 'cross-val_update.log'),  # Log file name
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


## SENTENCES FOR TRAINING/CROSS-VAL

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
test_data = Dataset.from_dict({'text': test_sentences + test_negatives, 'label': ["reported speech"] * len(test_sentences) + ["not reported speech"] * len(test_negatives)})

### Combine train and eval for cross-validation
rep_sent_cv = train_sentences + eval_sentences
neg_sent_cv = train_negatives + eval_negatives

all_sent_cv = rep_sent_cv + neg_sent_cv
all_labels = ["reported speech"] * len(rep_sent_cv) + ["not reported speech"] * len(neg_sent_cv)


## CROSS-VAL SETUP
logging.info('Starting cross-validation...')
### Model name
model_name = 'intfloat/multilingual-e5-large'
### Device
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

n_splits = 5 # n folds
kf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed_n)

### List for storing metrics
metrics_per_fold = []

for fold, (train_index, val_index) in enumerate(kf.split(all_sent_cv, all_labels), start = 1):
    print(f"Fold {fold}/{n_splits}")
    
    # Split trænings- og valideringsdata for denne fold
    train_texts = [all_sent_cv[i] for i in train_index]
    train_labels = [all_labels[i] for i in train_index]
    val_texts = [all_sent_cv[i] for i in val_index]
    val_labels = [all_labels[i] for i in val_index]
    
    # Opret Dataset for trænings- og valideringsdata
    fold_train_data = Dataset.from_dict({'text': train_texts, 'label': train_labels})
    fold_val_data = Dataset.from_dict({'text': val_texts, 'label': val_labels})
    
    # Indlæs modellen på ny for hver fold
    model = SetFitModel.from_pretrained(
        model_name, 
        labels=["not reported speech", "reported speech"],
        ### Hyperparameters (best run from HPO)
        head_params={
                'max_iter': 300, 
                'solver': 'lbfgs'
            }
    ).to(device)
    
    # Initialiser træningsargumenter
    args = TrainingArguments(
        ### Hyperparameters (best run from HPO - see hpo/hpo_bestrun.txt)
        batch_size=32,
        num_epochs=6,
        evaluation_strategy="epoch",
        body_learning_rate = 1.0770502781075495e-06,
        save_strategy="epoch",
        load_best_model_at_end=True
    )
    
    # Træn modellen på træningsdata for denne fold
    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=fold_train_data,
        eval_dataset=fold_val_data,
        metric=compute_metrics,
        column_mapping={"text": "text", "label": "label"}
    )
    
    # Train model
    trainer.train()
    metrics = trainer.evaluate(test_data)
    metrics_per_fold.append(metrics)

    # Write fold metrics to log
    logging.info(f'Metrics for fold {fold}: %s', json.dumps(metrics, indent=4))

# Avg. metrics
avg_metrics = {metric: np.mean([m[metric] for m in metrics_per_fold]) for metric in metrics_per_fold[0]}
logging.info(f"Average metrics across {n_splits} folds: %s", json.dumps(avg_metrics, indent=4))

# NOTE: Results are copied manuelly from log logs/cross-val_update.log to output/cross-val-results.json
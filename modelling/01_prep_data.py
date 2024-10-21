#!/usr/bin/env python
# coding: utf-8

import sys
import os
from os.path import join
import json
import re
from itertools import chain
import random

project_dir = join('/work', 'SMI-CM')
modelling_dir = join(project_dir, 'rep-speech-model')
modules_dir = join(modelling_dir, 'modules')
sys.path.append(modules_dir)

from smi_sm_funs import * # contains get_sentences function for sampling positive and negative sentences from data

## DIRS AND PATHS
data_dir = join(project_dir, 'data')

data_p = join(data_dir, 'annotated', 'smi-cm_annotated_2024-07-08.jsonl')

# read data
with open(data_p, 'r') as f:
    data = [json.loads(line) for line in f]

# extract sentences
rep_speech_sentences = [get_sentences(entry, seed_no=4102) for entry in data]

## exclude texts with no sentences
rep_speech_sentences = [rep for rep in rep_speech_sentences if rep.get('rep_speech')]

## export data
out_p = join(data_dir, 'rep-speech_model-data.json')

with open(out_p, 'w') as f:
    json.dump(rep_speech_sentences, f)
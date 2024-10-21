#!/usr/bin/env python
# coding: utf-8

from pysbd import Segmenter
import random
from itertools import chain

## Function for extracting labelled sentences based on text indexes
def get_sentences(entry, seed_no, seg = Segmenter(language='da', clean=False)):
    
    ## set seed
    random.seed(seed_no)

    ## extract
    labels_list = entry.get('label')
    text = entry.get('text')

    ## index pairs for labels
    index_pairs = [(label[0], label[1]) for label in labels_list]

    ## list of all label indexes
    label_indexes = list(chain(*[list(range(pair[0], pair[1])) for pair in index_pairs]))

    ## text excluding labelled text (for negative samples)
    text_nolabel = ''.join([char for i, char in enumerate(list(text)) if i not in label_indexes])
    
    ## sentences from text without labels
    negative_sentences = seg.segment(text_nolabel)

    ## number of negative samples (based on number of positives)
    n_negatives = len(labels_list)

    ## draw negative samples
    negatives = random.sample(negative_sentences, n_negatives)

    sentences = []
    for pair in index_pairs:
        sentence = text[pair[0]:pair[1]]

        sentences.append(sentence)

    entry_return = {
        'id': entry.get('id'),
        'rep_speech': sentences,
        'negatives': negatives
    }
    
    return(entry_return)
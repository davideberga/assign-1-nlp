import random
from functools import reduce

from nltk.corpus import genesis
from sklearn.model_selection import train_test_split
import math

from coroutine import *
from pipeline_utils import *


# Compute corpus infos
corpus = genesis
tokens = 0
vocabulary = set()

for sentence in corpus.sents():
    for word in sentence:
        tokens+=1
        vocabulary.add(word)

print(">> Number of tokens: " + str(tokens))
print(">> Vocabulary dimension: " + str(len(vocabulary)))

print(">> Pipeline starting")

prepare_and_tag_data(corpus, targets=[
    randomize_docs_set_pipeline(targets=[
        attr_label_split_pipeline(targets=[
            train_test_split_pipeline(targets=[
                learning_pipeline(targets=[
                    test_pipeline(targets=[preformance_evaluation()]),
                ])
            ])
        ])
    ])
])
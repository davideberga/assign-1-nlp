import random
from functools import reduce

from nltk.corpus import genesis
from sklearn.model_selection import train_test_split
import math

from coroutine import *
from pipeline_utils import *

prepare_and_tag_data(genesis, targets=[
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
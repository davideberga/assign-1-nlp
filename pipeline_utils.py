import random
from functools import reduce

from nltk.corpus import genesis
from sklearn.model_selection import train_test_split
import math

from coroutine import *

classes = [0, 1]
trainPartionDim = 0.8
alpha = 1

##### Utility lamda functions #####

nDocsIn = lambda cls, setOfItems: reduce(lambda acc, current: acc + (current[1] == cls), enumerate(setOfItems), 0)

createText = lambda cls, attributes, labels: reduce(lambda acc, current: acc + current[1], filter(lambda doc: labels[doc[0]] == cls, enumerate(attributes)), [])

countWordIn = lambda word, bagOfWords: reduce(lambda acc, current: acc + (current == word), bagOfWords, 0);

##### END  

def prepare_and_tag_data(corpus, targets):
    docsSet = []
    for file in genesis.fileids():
        clazz = classes[0] if 'english' in file else classes[1]
        for sentence in genesis.sents([file]):
            withoutDuplicates = list(dict.fromkeys(sentence)) # remove duplicates within each doc
            docsSet.append([clazz, withoutDuplicates])
    print(str(len(docsSet)))
    for t in targets:
            t.send(docsSet)

@coroutine
def randomize_docs_set_pipeline(targets):
    while True:
        randomize = (yield)
        random.shuffle(randomize)
        for target in targets:
            target.send(randomize)

@coroutine
def attr_label_split_pipeline(targets):
    while True:
        taggedSet = (yield)
        docsSetInputs = []
        outputs = []
        for sentence in taggedSet:
            outputs.append(sentence[0])
            docsSetInputs.append(sentence[1])
        for target in targets:
            target.send([docsSetInputs, outputs])

@coroutine
def train_test_split_pipeline(targets):
    while True:
        [X, Y] = (yield)
        x_train, x_test, y_train, y_test = train_test_split(X, Y, train_size=trainPartionDim)
        for target in targets:
            target.send([x_train, x_test, y_train, y_test])

@coroutine
def learning_pipeline(targets):
     while True:
        [x_train, x_test, y_train, y_test] = (yield)
        totalTrainDocs = len(y_train)
        docsTrainInEnglish = nDocsIn(classes[0], y_train)
        docsTrainNotInEnglish = totalTrainDocs - docsTrainInEnglish
        print("Train docs english " + str(docsTrainInEnglish))
        print("Train docs not wnglish " + str(docsTrainNotInEnglish))

        # Calc relative frequencies for every class [P(c_j)]
        pEnglish = math.log(docsTrainInEnglish/totalTrainDocs);
        pNotEnglish = math.log(docsTrainNotInEnglish/totalTrainDocs);

        # Extract vocabulary for train set (must be the globally accepted as it is)
        vocabulary = set()
        for sentence in x_train:
            for word in sentence:
                vocabulary.add(word)

        textEnglish = createText(classes[0], x_train, y_train)
        textNotEnglish = createText(classes[1], x_train, y_train)

        how_many_en = dict()
        how_many_nen = dict()
        freq_english = dict()
        freq_not_english = dict()

        for word in textEnglish:
            how_many_en[word] = how_many_en.get(word, 0) + 1
        for word in textNotEnglish:
            how_many_nen[word] = how_many_nen.get(word, 0) + 1

        for word in vocabulary:
            w_k = (how_many_en.get(word, 0) + alpha)/(len(textEnglish) + alpha * len(vocabulary))
            freq_english[word] = math.log(w_k)
            w_k = (how_many_nen.get(word, 0) + alpha)/(len(textNotEnglish) + alpha * len(vocabulary))
            freq_not_english[word] = math.log(w_k)

        for target in targets:
            target.send([x_test, y_test, vocabulary, pEnglish, pNotEnglish, freq_english, freq_not_english])

@coroutine
def test_pipeline(targets):
    while True:
        [x_test, y_test, vocabulary, pEnglish, pNotEnglish, freq_english, freq_not_english] = (yield)
        confMatrix = [[0, 0], [0, 0]]
        for testSentence in enumerate(x_test):
            goldLabel = y_test[testSentence[0]]
            sentence = list(dict.fromkeys(testSentence[1]))
            sumEnglish, sumNotEnglish = pEnglish, pNotEnglish
            for word in sentence:
                if word in vocabulary:
                    sumEnglish += freq_english.get(word)
                    sumNotEnglish += freq_not_english.get(word)
            outputLabel = classes[0] if(sumEnglish > sumNotEnglish) else classes[1]
            confMatrix[goldLabel][outputLabel] += 1
        for target in targets:
            target.send(confMatrix)

@coroutine
def preformance_evaluation():
    while True:
        confMatrix = (yield)
        TP = confMatrix[0][0]
        TN = confMatrix[1][1]
        FP = confMatrix[0][1]
        FN = confMatrix[1][0]

        precision = TP/(TP+FP)
        recall = TP/(TP+FN)
        accuracy = (TP+ TN)/(TP+FP+TN+FN)
        f1Measure = (2*precision*recall)/(precision + recall)
        print('Confusion Matrix: ' + str(confMatrix))
        print("Precision " + str(precision))
        print("Recall " + str(recall))
        print("Accuracy " + str(accuracy))
        print("F1 measure " + str(f1Measure))


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
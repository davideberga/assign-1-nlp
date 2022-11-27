import random
from functools import reduce
import re

from nltk import NaiveBayesClassifier
from nltk import FreqDist
from sklearn.metrics import confusion_matrix
from nltk.corpus import genesis
from sklearn.model_selection import train_test_split
import math

from coroutine import *

classes = [0, 1]

trainPartionDim = 0.7
alpha = 1

##### Utility lamda functions #####

removePunctuation = lambda sentence: filter(lambda word: re.search("[^\.\"!?\-)\(_;,:0-9]", word), sentence)

nDocsIn = lambda cls, setOfItems: reduce(lambda acc, current: acc + (current[1] == cls), enumerate(setOfItems), 0)

createText = lambda cls, attributes, labels: reduce(lambda acc, current: acc + current[1], filter(lambda doc: labels[doc[0]] == cls, enumerate(attributes)), [])

countWordIn = lambda word, bagOfWords: reduce(lambda acc, current: acc + (current == word), bagOfWords, 0);

##### END  

def prepare_and_tag_data(corpus, targets):
    docsSet = []
    for file in genesis.fileids():
        clazz = classes[0] if 'english' in file or 'lolcat' in file else classes[1]
        for sentence in genesis.sents([file]):
            sentence[0] = sentence[0].lower() # to lower case
            sentence = removePunctuation(sentence)
            withoutDuplicates = list(dict.fromkeys(sentence)) # remove duplicates within each doc
            docsSet.append([clazz, withoutDuplicates]) 
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
        print("Train docs not english " + str(docsTrainNotInEnglish))

        # Calc relative frequencies for every class [P(c_j)]
        pEnglish = math.log(docsTrainInEnglish/totalTrainDocs);
        pNotEnglish = math.log(docsTrainNotInEnglish/totalTrainDocs);

        # Extract vocabulary for train set
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
def features_ext_train_test_split_pipeline_nltk(targets):
    while True:
        [X, Y] = (yield)
        all_words = []
        for d in X:
            for w in d:
                all_words.append(w)
        all_words = FreqDist(all_words)
        # Take in consideration only the most frequent words
        word_features = list(all_words.keys())[:1800]
        
        def _format_features(d, label):
            words = set(d)
            features = {}
            for w in words:
                features[w] = (w in word_features)
            return (features, label)

        labeled_featureset = [_format_features(doc[1], Y[doc[0]]) for doc in enumerate(X)]
        boundary = int(trainPartionDim * len(labeled_featureset))

        for target in targets:
            target.send([labeled_featureset[:boundary], labeled_featureset[boundary:]])

@coroutine
def learning_pipeline_nltk(targets):
     while True:
        [training_set, test_set] = (yield)
        
        model = NaiveBayesClassifier.train(training_set)

        for target in targets:
            target.send([model, test_set])

@coroutine
def test_pipeline_nltk(targets):
     while True:
        [model, test_set] = (yield)
        confMatrix = [[0, 0], [0, 0]]
        # Compute manually the confusion matrix
        for test in test_set:
            (feat, label) =  test
            y_predicted = model.classify(feat)
            confMatrix[y_predicted][label] += 1
        for target in targets:
            target.send(confMatrix)

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
            confMatrix[outputLabel][goldLabel] += 1
            #if(goldLabel != outputLabel):
            #    print(" >>> Should be " + str(goldLabel) + " predicted: " + str(outputLabel))
            #    print("     >>> Sum english " + str(sumEnglish) + " sum not english: " + str(sumNotEnglish))
            #    print("         " + str(sentence))
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





import random
from functools import reduce

from nltk.corpus import genesis
from sklearn.model_selection import train_test_split
import math

from coroutine import *



documentsPerLanguage = dict()
classes = [0, 1]
trainPartionDim = 0.8

print(genesis.sents(['english-kjv.txt']))

exit 

for file in genesis.fileids():
    print(file)
    if file == 'english-kjv.txt' or file == 'english-web.txt':
        print(len(genesis.sents([file])))
        documentsPerLanguage['english'] = genesis.sents([file])
    else:
        documentsPerLanguage[file] = genesis.sents([file])

docsSet = []
docsSetInputs = []
docsSetOutputs = []

# remove also duplicates
for lang in documentsPerLanguage.keys():
    clazz = classes[0] if lang == 'english' else classes[1]
    for sentence in documentsPerLanguage[lang]:
        withoutDuplicates = list(dict.fromkeys(sentence))
        docsSet.append([clazz, withoutDuplicates])

random.shuffle(docsSet)
print(str(len(docsSet)))

for sentence in docsSet:
    docsSetOutputs.append(sentence[0])
    docsSetInputs.append(sentence[1])

x_train, x_test, y_train, y_test = train_test_split(docsSetInputs, docsSetOutputs, train_size=trainPartionDim)

nDocsIn = lambda cls, setOfItems: reduce(lambda acc, current: acc + (current[1] == cls), enumerate(setOfItems), 0)

createText = lambda cls, attributes, labels: reduce(lambda acc, current: acc + current[1], filter(lambda doc: labels[doc[0]] == cls, enumerate(attributes)), [])

totalTrainDocs = len(y_train)
docsTrainInEnglish = nDocsIn(classes[0], y_train)
docsTrainNotInEnglish = totalTrainDocs - docsTrainInEnglish

print("Train docs english " + str(docsTrainInEnglish))
print("Train docs not wnglish " + str(docsTrainNotInEnglish))

# Calc relative frequencies for every class [P(c_j)]
pEnglish = math.log(docsTrainInEnglish/totalTrainDocs);
pNotEnglish = math.log(docsTrainNotInEnglish/totalTrainDocs);

print("Log prob english " + str(pEnglish))
print("Log prob not english " + str(pNotEnglish))

#Extract vocabulary for train set (must be the globally accepted as it is)
vocabulary = set()

for sentence in x_train:
    for word in sentence:
        vocabulary.add(word) 

# Text_english with all docs in english
textEnglish = createText(classes[0], x_train, y_train)
textNotEnlish = createText(classes[1], x_train, y_train)

alpha = 5

frequencies_english = dict()
frequencies_not_english = dict()

# Learning

print('Start learning')
print('Words in english: ' + str(len(textEnglish)));
print('Words in not english: ' + str(len(textNotEnlish)));

for word in textEnglish:
    frequencies_english[word] = frequencies_english.get(word, 0) + 1

for word in textNotEnlish:
    frequencies_not_english[word] = frequencies_not_english.get(word, 0) + 1

for word in vocabulary:
    w_k = (frequencies_english.get(word, 0) + alpha)/(len(textEnglish) + alpha * len(vocabulary))
    frequencies_english[word] = math.log(w_k)
    w_k = (frequencies_not_english.get(word, 0) + alpha)/(len(textNotEnlish) + alpha * len(vocabulary))
    frequencies_not_english[word] = math.log(w_k)
    
# Testing

confMatrix = [[0, 0], [0, 0]]

print('Start testing')

for testSentence in enumerate(x_test):
    goldLabel = y_test[testSentence[0]]
    sentence = testSentence[1]
    sumEnglish = pEnglish
    sumNotEnglish = pNotEnglish
    sentence = list(dict.fromkeys(sentence))
    for word in sentence:
        if word in vocabulary:
            sumEnglish += frequencies_english.get(word)
            sumNotEnglish += frequencies_not_english.get(word)
    # print("sumEnglish: " + str(sumEnglish) + " sumNotEnglish: " + str(sumNotEnglish))
    outputLabel = classes[0] if(sumEnglish > sumNotEnglish) else classes[1]
    # print("Output: " + str(outputLabel) + " gold label: " + str(goldLabel))
    confMatrix[outputLabel][goldLabel] += 1

print(confMatrix)

TP = confMatrix[0][0]
TN = confMatrix[1][1]
FP = confMatrix[0][1]
FN = confMatrix[1][0]

precision = TP/(TP+FP)
recall = TP/(TP+FN)
accuracy = (TP+ TN)/(TP+FP+TN+FN)
f1Measure = (2*precision*recall)/(precision + recall)

print("Precision " + str(precision))
print("Recall " + str(recall))
print("Accuracy " + str(accuracy))
print("F1 measure " + str(f1Measure))





# Extract vocabulary
#






# Wordlist corpus

#if nltk.download('genesis') & nltk.download('punkt'): 
#    print('Genesis downloaded successfully')

#nltk.corpus.gutenberg.fileids()

   

# Extract some corpus in english
# Extract some corpus in other languages
# Label each word
# Put all the word in array
# Randomize it
# 70 - 30 (maybe delete the anseen words in 70 and duplicates)
# Write the model
# Apply the model to the dataset
# Create the confusion matrix



#sent_text = sent_tokenize(nltk.corpus.genesis) # this gives us a list of sentences
# now loop over each sentence and tokenize it separately









print('Finish')
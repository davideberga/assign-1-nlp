# A basic Naive Bayes classifier
Davide Bergamasco

Repository url: [url](https://github.com/davideberga/assign-1-nlp)

To run this model you need:
- nltk, sklearn packages
- the genesis corpus

## An initial note

To better understand how a Naive Bayes Classifier works I have written a custom version of this algorithm from scratch (nltk independent). Because it has the same performance of the nltk one I let it in the code but only the nltk version is runnable (unless you de-comment that part).

## Corpus

I built this classifier using the NLTK package and I chose the **genesis** corpus as the training dataset. It contains 315302 tokens and a vocabulary composed of 25832 types. Its size seems to be appropriate for the task and I used it also because it contains documents written in different languages.
Here's the list of the different corpora involved and how I labeled the docuemnts inside them:

| Text            | Label       |
|-----------------|-------------|
| english-kjv.txt | English     |
| english-web.txt | English     |
| finnish.txt     | Not English |
| french.txt      | Not English |
| german.txt      | Not English |
| lolcat.txt      | English     |
| portuguese.txt  | Not English |
| swedish.txt     | Not English |

### Genesis Provenance (from its README.md)

This corpus has been prepared from several web sources; formatting,
markup and verse numbers have been stripped.

#### CONTENTS

english-kjv.txt - Genesis, King James version (Project Gutenberg)
english-web.txt - Genesis, World English Bible (Project Gutenberg)
finnish.txt - Genesis, Suomen evankelis-luterilaisen kirkon kirkolliskokouksen vuonna 1992 käyttöön ottama suomennos
french.txt - Genesis, Louis Segond 1910
german.txt - Genesis, Luther Translation
lolcat.txt - Genesis, Lolcat version http://www.lolcatbible.com/
portuguese.txt - Genesis, Brazilian Portuguese version - http://www.bibliaonline.com.br
swedish.txt - Genesis, Gamla och Nya Testamentet, 1917 (Project Runeberg)

## Train/test dataset split

Once the data has been transformed into a standard format the dataset has been split with a train_size of 0.7. This results typically in: about *3200 documents* in ENGLISH and *6300 documents* labeled as NOT-ENGLISH, composing the dataset used during the learning process. 

## Performance indicators

I used the following performance indicators:

- **Accuracy**: % of items correctly identified (meaningless)
- **Recall**: % of items actually present in the input that were correctly identified by the system. 
- **Precision**: % of items the system detected that are actually positive
- **F1 measure**: gives the information about precision and recall in a single number

In the development process, I used to check mainly the *recall* and *precision* indicators because they give me an immediate measure of where the model is failing. 

Given also the following performance on unseen data (test set) I decided that the process of cross-validation was not necessary.

```
Confusion Matrix: [[1317, 1], [2, 2771]]
Precision 0.9992412746585736
Recall 0.9984836997725549
Accuracy 0.9992666829626008
F1 measure 0.9988623435722412
```

## Usage as a Probabilistic Language Model

A model that can compute the probability of a sentence (sequence of words) is a probabilistic language model. When the naive Bayes model, discussed before, has been learned the ENGLISH class can be seen as a unigram language model. In fact, given that one is possible to compute the following probabilities:

- P( word | ENGLISH)
- P( sentence | ENGLISH)



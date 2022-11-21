# A basic Naive Bayes classifier

## Corpus

I built this classifier using the NLTK package and I chose the **genesis** corpus as training dataset. It contains 315302 tokens and a vocabulary composed by 25832 types. Its size seems to me appropriate for the task and I used it also because it contains documents written in different languages.
Here's the list of the different texts envolved and how I labeled them:

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

## Train/test dataset split

Once the data has been trasformed in a standard format the dataset has been splitted with a train_size of 0.7. This results typically in: about *3200 documents* in ENGLISH and *6300 documents* labeled as NOT-ENGLISH, composing the dataset used during the learning process. 

## Performance indicators

I used the following performance indicators:

- **Accuracy**: 
- **Recall**:
- **Precision**:
- **F1 measure**:

In the development process I used to check mainly the *recall* and *precision* indicators because they gave me an immediate measure where the model was failing.

```
Confusion Matrix: [[1317, 1], [2, 2771]]
Precision 0.9992412746585736
Recall 0.9984836997725549
Accuracy 0.9992666829626008
F1 measure 0.9988623435722412
```



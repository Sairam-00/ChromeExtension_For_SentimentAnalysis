#!/usr/bin/env python
# coding: utf-8

import numpy as np
import pandas as pd
import random
import matplotlib.pyplot as plt
import re 
import string 
import seaborn as sns
from mpl_toolkits.mplot3d import Axes3D
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt # plotting
from wordcloud import WordCloud, STOPWORDS
import collections
from collections import defaultdict
import os # accessing directory structure
import warnings
warnings.filterwarnings('ignore')
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
import nltk
import inflect
import contractions
from bs4 import BeautifulSoup
import re, string, unicodedata
from nltk import word_tokenize, sent_tokenize
from nltk.corpus import stopwords
from nltk.stem import LancasterStemmer, WordNetLemmatizer
from IPython.display import display
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn import feature_extraction, linear_model, model_selection, preprocessing
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn import metrics
import itertools
from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import RandomForestClassifier
from sklearn import svm
from sklearn.linear_model import LogisticRegression




data = pd.read_csv("training.csv")
test = pd.read_csv("test.csv")



data.head(10)
test.head(10)


# ### Pre Processing

# #### 0 - sadness, 1 - joy,  2 - love, 3 - anger, 4 - fear, 5 - surprise

labels_dict = {0:'sadness', 1:'joy', 2:'love', 3:'anger', 4:'fear', 5:'surprise'}
data['description'] = data['label'].map(labels_dict )
data.head(10)


data['description'].value_counts(normalize=True)

sns.countplot(data['description'],order = data['description'].value_counts(normalize=True).index)

total = data.isnull().sum()
total


def text_preprocessing_platform(data,text_col,remove_stopwords=True):

    def denoise_text(text):
        soup = BeautifulSoup(text, "html.parser")
        text = soup.get_text()
        text = contractions.fix(text)
        return text
    
    
    def remove_non_ascii(words):
        """Remove non-ASCII characters from list of tokenized words"""
        new_words = []
        for word in words:
            new_word = unicodedata.normalize('NFKD', word).encode('ascii', 'ignore').decode('utf-8', 'ignore')
            new_words.append(new_word)
        return new_words
    
    
    def to_lowercase(words):
        """Convert all characters to lowercase from list of tokenized words"""
        new_words = []
        for word in words:
            new_word = word.lower()
            new_words.append(new_word)
        return new_words
    
    
    def remove_punctuation(words):
        """Remove punctuation from list of tokenized words"""
        new_words = []
        for word in words:
            new_word = re.sub(r'[^\w\s]', '', word)
            if new_word != '':
                new_words.append(new_word)
        return new_words
    
    
    def replace_numbers(words):
        """Replace all interger occurrences in list of tokenized words with textual representation"""
        p = inflect.engine()
        new_words = []
        for word in words:
            if word.isdigit():
                new_word = p.number_to_words(word)
                new_words.append(new_word)
            else:
                new_words.append(word)
        return new_words
    
    
    def remove_stopwords(words):
        """Remove stop words from list of tokenized words"""
        new_words = []
        for word in words:
            if word not in stopwords.words('english'):
                new_words.append(word)
        return new_words
    
    
    def stem_words(words):
        """Stem words in list of tokenized words"""
        stemmer = LancasterStemmer()
        stems = []
        for word in words:
            stem = stemmer.stem(word)
            stems.append(stem)
        return stems
    
    
    def lemmatize_verbs(words):
        """Lemmatize verbs in list of tokenized words"""
        lemmatizer = WordNetLemmatizer()
        lemmas = []
        for word in words:
            lemma = lemmatizer.lemmatize(word, pos='v')
            lemmas.append(lemma)
        return lemmas
    
    
    ### A wrap-up function for normalization
    def normalize_text(words, remove_stopwords):
        words = remove_non_ascii(words)
        words = to_lowercase(words)
        words = remove_punctuation(words)
        words = replace_numbers(words)
        if remove_stopwords:
            words = remove_stopwords(words)
        #words = stem_words(words)
        words = lemmatize_verbs(words)
        return words

    def tokenize(text):
        return nltk.word_tokenize(text)

    def text_prepare(text):
        text = denoise_text(text)
        text = ' '.join([x for x in normalize_text(tokenize(text), remove_stopwords)])
        return text
    
    data[text_col] = [text_prepare(x) for x in data[text_col]]
    
    return data

print("Before Text Preprocessing")
display(data.head()[['text']])
processed_data = text_preprocessing_platform(data, 'text', remove_stopwords=False)
print("After Text Preprocessing")
display(processed_data.head()[['text']])


# ### Data Visualization

data['text_length'] = data['text'].astype(str).apply(len)
data['text_word_count'] = data['text'].apply(lambda x: len(str(x).split()))

sns.distplot(data['text_length'])
plt.xlim([0, 512]);
plt.xlabel('Text Length');


sns.boxplot(x="description", y="text_word_count", data=data)

col = 'description'
fig, (ax1, ax2)  = plt.subplots(nrows=1, ncols=2, figsize=(12,8))
explode = list((np.array(list(data[col].dropna().value_counts()))/sum(list(data[col].dropna().value_counts())))[::-1])[:10]
labels = list(data[col].dropna().unique())[:10]
sizes = data[col].value_counts()[:10]
ax2.pie(sizes,  explode=explode, startangle=60, labels=labels,autopct='%1.0f%%', pctdistance=0.9)
ax2.add_artist(plt.Circle((0,0),0.6,fc='white'))
sns.countplot(y =col, data = data, ax=ax1)
ax1.set_title("Count of each emotion")
ax2.set_title("Percentage of each emotion")
plt.show()



def print_word_cloud(data, description):

    print("Word cloud of most frequent words for the sentiment : {}".format(description))

    temp_data = data[data['description']==description]
    print("Number of Rows : ", len(temp_data))

    corpus = ''
    for text in temp_data.text:
        text = str(text)
        corpus += text
             
    total = 0
    count = defaultdict(lambda: 0)
    for word in corpus.split(" "):
        total += 1
        count[word] += 1
        
    top20pairs = sorted(count.items(), key=lambda kv: kv[1], reverse=True)[:20]
    top20words = [i[0] for i in top20pairs]
    top20freq = [i[1] for i in top20pairs]
    
    xs = np.arange(len(top20words))
    width = 0.5

    fig = plt.figure(figsize=(10,6))                                                               
    ax = fig.gca()  #get current axes
    ax.bar(xs, top20freq, width, align='center')

    ax.set_xticks(xs)
    ax.set_xticklabels(top20words)
    plt.xticks(rotation=45)
    
    
    stopwords = set(STOPWORDS)
    # lower max_font_size, change the maximum number of word and lighten the background:
    wordcloud = WordCloud(max_font_size=50, max_words=50,stopwords=stopwords, background_color="white").generate(corpus)
    plt.figure(figsize = (12, 12), facecolor = None)
    plt.imshow(wordcloud, interpolation="bilinear")
    plt.axis("off")
    plt.show()

print_word_cloud(data, 'sadness')

print_word_cloud(data, 'surprise')

y = data.description
x = data.text

x.head()
y.head()


def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, cm[i, j],
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')


X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.4, random_state=42)

dct = dict()

# Naive Bayes

NB_classifier = MultinomialNB()
pipe = Pipeline([('vect', CountVectorizer()),
                 ('tfidf', TfidfTransformer()),
                 ('model', NB_classifier)])

model = pipe.fit(X_train, y_train)
prediction = model.predict(X_test)
print("Accuracy: {}%".format(round(accuracy_score(y_test, prediction)*100,2)))

dct['Naive Bayes'] = round(accuracy_score(y_test, prediction)*100,2)


cm = metrics.confusion_matrix(y_test, prediction)
plot_confusion_matrix(cm, classes=[])

# Random Forest

pipe = Pipeline([('vect', CountVectorizer()),
                 ('tfidf', TfidfTransformer()),
                 ('model', RandomForestClassifier(n_estimators=50, criterion="entropy"))])

model = pipe.fit(X_train, y_train)
prediction = model.predict(X_test)
print("Accuracy: {}%".format(round(accuracy_score(y_test, prediction)*100,2)))
dct['Random Forest'] = round(accuracy_score(y_test, prediction)*100,2)


cm = metrics.confusion_matrix(y_test, prediction)
plot_confusion_matrix(cm, classes=[])


# svm Classifier

clf = svm.SVC(kernel='linear') # Linear Kernel

pipe = Pipeline([('vect', CountVectorizer()),
                 ('tfidf', TfidfTransformer()),
                 ('model', clf)])

svm_model = pipe.fit(X_train, y_train)
svm_prediction = svm_model.predict(X_test)
print("accuracy: {}%".format(round(accuracy_score(y_test, svm_prediction)*100,2)))
dct['SVM'] = round(accuracy_score(y_test, svm_prediction)*100,2)

cm = metrics.confusion_matrix(y_test, svm_prediction)
plot_confusion_matrix(cm, classes=[])

# Logistic Regression

pipe = Pipeline([('vect', CountVectorizer()),
                 ('tfidf', TfidfTransformer()),
                 ('model', LogisticRegression())])

# Fitting the model
model = pipe.fit(X_train, y_train)

# Accuracy
prediction = model.predict(X_test)
print("accuracy: {}%".format(round(accuracy_score(y_test, prediction)*100,2)))
dct['Logistic Regression'] = round(accuracy_score(y_test, prediction)*100,2)

cm = metrics.confusion_matrix(y_test, prediction)
plot_confusion_matrix(cm, classes=[])

# Comparing classifiers

import matplotlib.pyplot as plt
plt.figure(figsize=(7,7))
plt.bar(list(dct.keys()),list(dct.values()))
plt.ylim(30,90)
plt.yticks((30, 60, 90))


# ###  The best accurate model is SVM with 85.08% accuracy

# ### Check accuracy on Test Data

test.head(10)

print("Before Text Preprocessing")
display(test.head()[['text']])
processed_test = text_preprocessing_platform(test, 'text', remove_stopwords=False)
print("After Text Preprocessing")
display(processed_test.head()[['text']])

# predicting test data

model.fit(data['text'], data['label'])
model.predict(test['text'])
labels_dict = {0:'sadness', 1:'joy', 2:'love', 3:'anger', 4:'fear', 5:'surprise'}
test['description'] = test['label'].map(labels_dict )
test.head(10)

# predicting sentiment for user input

svm_model.fit(data['text'], data['label'])
string = ["i am teriffied"]
pred = svm_model.predict(string)
labels_dict = {0:'sadness', 1:'joy', 2:'love', 3:'anger', 4:'fear', 5:'surprise'}
output = labels_dict[pred[0]]
print(output)

# function for predicting

def predict(svm_model,text):
    pred = svm_model.predict(string)
    labels_dict = {0:'sadness', 1:'joy', 2:'love', 3:'anger', 4:'fear', 5:'surprise'}
    output = labels_dict[pred[0]]
    return output

#########################################################################################################

input1 = "my friend gifted me a bike i am so happy"
input2 = pd.DataFrame()
input2["text"] = np.array([input1])
text_col = ["text"]
processed_input = test_preprocessing_platform(input2,text_col,remove_stopwords=True)
print(processed_input)
#print(input2)


def test_preprocessing_platform(data,text_col,remove_stopwords=True):

    def denoise_text(text):
        soup = BeautifulSoup(text, "html.parser")
        text = soup.get_text()
        text = contractions.fix(text)
        return text
    
    
    def remove_non_ascii(words):
        """Remove non-ASCII characters from list of tokenized words"""
        new_words = []
        for word in words:
            new_word = unicodedata.normalize('NFKD', word).encode('ascii', 'ignore').decode('utf-8', 'ignore')
            new_words.append(new_word)
        return new_words
    
    
    def to_lowercase(words):
        """Convert all characters to lowercase from list of tokenized words"""
        new_words = []
        for word in words:
            new_word = word.lower()
            new_words.append(new_word)
        return new_words
    
    
    def remove_punctuation(words):
        """Remove punctuation from list of tokenized words"""
        new_words = []
        for word in words:
            new_word = re.sub(r'[^\w\s]', '', word)
            if new_word != '':
                new_words.append(new_word)
        return new_words
    
    
    def replace_numbers(words):
        """Replace all interger occurrences in list of tokenized words with textual representation"""
        p = inflect.engine()
        new_words = []
        for word in words:
            if word.isdigit():
                new_word = p.number_to_words(word)
                new_words.append(new_word)
            else:
                new_words.append(word)
        return new_words
    
    
    def remove_stopwords(words):
        """Remove stop words from list of tokenized words"""
        new_words = []
        for word in words:
            if word not in stopwords.words('english'):
                new_words.append(word)
        return new_words
    
    
    def stem_words(words):
        """Stem words in list of tokenized words"""
        stemmer = LancasterStemmer()
        stems = []
        for word in words:
            stem = stemmer.stem(word)
            stems.append(stem)
        return stems
    
    
    def lemmatize_verbs(words):
        """Lemmatize verbs in list of tokenized words"""
        lemmatizer = WordNetLemmatizer()
        lemmas = []
        for word in words:
            lemma = lemmatizer.lemmatize(word, pos='v')
            lemmas.append(lemma)
        return lemmas
    
    
    ### A wrap-up function for normalization
    def normalize_text(words, remove_stopwords):
        words = remove_non_ascii(words)
        words = to_lowercase(words)
        words = remove_punctuation(words)
        words = replace_numbers(words)
        if remove_stopwords:
            words = remove_stopwords(words)
        #words = stem_words(words)
        words = lemmatize_verbs(words)
        return words

    def tokenize(text):
        return nltk.word_tokenize(text)

    def text_prepare(text):
        text = denoise_text(text)
        text = ' '.join([x for x in normalize_text(tokenize(text), remove_stopwords)])
        return text
    
    data[text_col] = [text_prepare(x) for x in data[text_col]]
    
    return data


for x in data["text"]:
    print(x)
# NLP Natural Language Processing

# Targets:
# NLU Natural Language Understand
# NLG Natural Language Generation

# Approximations:
# Linguistik Approximation
# Statistic
# Hybrit

# The most using Libraries
# NLTK  :   nltk.org
# SpaCy :   spacy.io
# Stanford NLP
# OpenNLP : Apache : opennlp.apache.org
# Rapid Automatic Keyword Extract (RAKE)
# Amueller Word Cloud
# Tensor Flow : Word2Vec


# Turkish Libraries
# Zemberek
# İTÜ : tools.nlp.itu.edu.tr
# Tspell
# Yıldız Teknik Üniversitesi : Kemik
# Wordnet (Balkanet)
# TrMorph
# TSCorpus
# Metu- Sabanci Tree Bank ve ITU Doğrulama Kümesi


# NLP Steps
# Preprocessing Stop Words, Case, Parsers(html)
# Feature Extraction

import numpy as np
import pandas as pd

#Preprocessing ----------------------------------------------
#Loading data sets
comments = pd.read_csv('data/Restaurant_Reviews.csv', on_bad_lines='skip')
comments.dropna(inplace=True)

#NOTE : Sparks Matrices
# https://en.wikipedia.org/wiki/Spark_(mathematics)
# Matrices consisting of empty spaces are called sparks matrices.


# filtering alphanumeric data and punctuation
import re
import nltk

from nltk.stem.porter import PorterStemmer
ps = PorterStemmer()


nltk.download('stopwords')
from nltk.corpus import stopwords

compilation = []
for i in comments.index:
    comment = re.sub('[^a-zA-Z]',' ',comments['Review'][i])
    '''regular expression [^a-zA-Z]'''

    # Convert Lowercase and Uppercase
    comment = comment.lower()
    comment = comment.split()

    '''For each word in the comment we select, 
    if it is not in the stopwords list we defined in nltk, 
    create a set and find the stem, that is, the code snippet that finds the body of the word. 
    Ex: comment : ['wow', 'loved', 'this', 'place'] 
                apply stem method
        comment : ['wow', 'love', 'place']
    '''
    comment = [ps.stem(word) for word in comment if not word in set(stopwords.words('english'))]
    comment = ' '.join(comment)
    compilation.append(comment)

print(compilation)


#Feature Extraction (Bag of Words BOW)----------------------------------------------

from sklearn.feature_extraction.text import CountVectorizer
cv = CountVectorizer(max_features=1000)

X = cv.fit_transform(compilation).toarray() #independent value
y = comments.iloc[:,1].values #depent value


#Machine Learning ----------------------------------------------
#Split test and train variables ----------------------------------------------

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=0)

# Gaussian Naive Bayes
from sklearn.naive_bayes import GaussianNB
gnb = GaussianNB()
gnb.fit(X_train,y_train)

y_pred = gnb.predict(X_test)

#Confusion Matrix 
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test,y_pred)
print(cm)

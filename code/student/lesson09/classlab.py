# ________   .__          __                            
# \______ \  |__|  ____  |  | __  ______  ____    ____  
#  |    |  \ |  |_/ ___\ |  |/ / /  ___/ /  _ \  /    \ 
#  |    `   \|  |\  \___ |    <  \___ \ (  <_> )|   |  \
# /_______  /|__| \___  >|__|_ \/____  > \____/ |___|  /
#         \/          \/      \/     \/              \/ 
# GA DataScience Classwork
# Date: 2014-05-26

import os

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Don't show deprecation warnings
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning) 

# Set some Pandas options
pd.set_option('max_columns', 30)
pd.set_option('max_rows', 20)

# Store data in a consistent place
DATA_DIR = '../../../data/'

from sklearn import datasets, metrics
from matplotlib import pyplot as plt

iris = datasets.load_iris()

from sklearn.naive_bayes import GaussianNB
gnb = GaussianNB()

y_pred = gnb.fit(iris.data, iris.target).predict(iris.data)

print("Number of mislabeled points : %d" % (iris.target != y_pred).sum())
print "\n\n"

"""
Problem: Classify text as either an insult or not an insult
Data: Training data with text that decides if the input was an insult or not.
soooo.... this is a supervised learning Problem
Classifier: Naive Bayes.
"""
# IMPORT MODULES
from sklearn import naive_bayes, cross_validation, metrics
from sklearn.feature_extraction.text import CountVectorizer

######## LOAD DATA
train = pd.read_csv(DATA_DIR + 'insults/train-utf8.csv')
test = pd.read_csv(DATA_DIR + 'insults/test-utf8.csv')

### Text data isn't useable in it's form. We need to vectorize text, does it
# make more sense to get counts? maybe... a count vectorizer?

# Search 'text count vectorizer' in google.

# Seems like there's a bunch of arguments, but tells me it makes a world count matrix.
# Matrix sounds like a training set, so DONE HERE!

# Some things I see it also does:

# remove stop words. Neat!
# ngram range: relatively useful, how many words to consider per feature.

########## TRANSFORM THE DATA: COUNT VECTORIZE, CLEANING
vectorizer = CountVectorizer(ngram_range=(1,1), lowercase=True)
X_train = vectorizer.fit_transform(train.Comment)
X_test = vectorizer.transform(test.Comment)



##### USE THE WORD COUNT MATRIX TO PREDICT INSULT/NOT INSULT (1/0)
### BUILD A TRAINING AND TEST SET
model = naive_bayes.MultinomialNB().fit(X_train, list(train.Insult))

###### TEST RESULTS
###### CROSS VALIDATE
####### DISPLAY RESULTS: AUC TO CHECK FOR ERROR
print cross_validation.cross_val_score(naive_bayes.MultinomialNB(), X_train, train.Insult, cv=10).mean()
fpr, tpr, thresholds = metrics.roc_curve(train.Insult, model.predict(X_train), pos_label=1)
print metrics.auc(fpr, tpr)

######  OUTPUT RESULTS
predictions = model.predict_proba(X_test)[:,1]

submission = pd.DataFrame({'id' : test.id, 'insult': predictions})

#submission.to_csv(DATA_DIR + 'insults/submission.csv', index=False)

submission.to_csv(os.getcwd()+'/submission.csv', index=False)


from sklearn.feature_extraction.text import TfidfVectorizer
vect = TfidfVectorizer(sublinear_tf=True, max_df=0.7)
X_train = vect.fit_transform(train.Comment)
X_test = vect.transform(test.Comment)
model = naive_bayes.MultinomialNB().fit(X_train, list(train.Insult))
print cross_validation.cross_val_score(naive_bayes.MultinomialNB(), X_train, train.Insult, cv=10).mean()
fpr, tpr, thresholds = metrics.roc_curve(train.Insult, model.predict(X_train), pos_label=1)
print metrics.auc(fpr, tpr)

'''
print train.describe()

print train.head(5)

from sklearn.feature_extraction.text import HashingVectorizer

vectorizer_train = CountVectorizer(min_df=0, token_pattern=r"\b\w+\b", ngram_range=(1,1))
vectorizer_train.fit(train['Comment'])

print len(vectorizer_train.get_feature_names())

#print vectorizer_train.get_feature_names()


'''
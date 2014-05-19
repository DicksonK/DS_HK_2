# ________   .__          __                            
# \______ \  |__|  ____  |  | __  ______  ____    ____  
#  |    |  \ |  |_/ ___\ |  |/ / /  ___/ /  _ \  /    \ 
#  |    `   \|  |\  \___ |    <  \___ \ (  <_> )|   |  \
# /_______  /|__| \___  >|__|_ \/____  > \____/ |___|  /
#         \/          \/      \/     \/              \/ 
# GA DataScience Lab-7
# Date: 2014-05-19

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

import re

# Set some Pandas options
pd.set_option('max_columns', 30)
pd.set_option('max_rows', 20)

# Store data in a consistent place
DATA_DIR = '../../../data/'

#url = 'http://www-958.ibm.com/software/analytics/manyeyes/datasets/af-er-beer-dataset/versions/1.txt'
beer = pd.read_csv(DATA_DIR + 'beer.txt', delimiter="\t")

#print beer.head()

#print beer.describe()

beer = beer.dropna()
def good(x):
    if x > 4.3:
        return 1
    else:
        return 0

beer['Good'] = beer['WR'].apply(good)

from sklearn import linear_model

logm = linear_model.LogisticRegression()

X = beer[ ['Reviews', 'ABV'] ].values
y = beer['Good'].values

logm.fit(X, y)

logm.predict(X)

print logm.score(X, y)

#print len(set(beer['Type']))

# If this is true, then there was a match!
re.search('Apple', 'Apple Computer') != None

def find_ale(x):
    if x.find('Ale') > -1:
        return 1
    else:
    	return 0

def find_stout(x):
    if x.find('Stout') > -1:
        return 1
    else:
    	return 0

def find_ipa(x):
    if x.find('IPA') > -1:
        return 1
    else:
    	return 0

def find_lager(x):
    if x.find('Lager') > -1:
        return 1
    else:
    	return 0

beer['Ale'] = beer['Type'].apply(find_ale)
beer['Stout'] = beer['Type'].apply(find_stout)
beer['IPA'] = beer['Type'].apply(find_ipa)
beer['Lager'] = beer['Type'].apply(find_lager)

#print beer.head()

X = beer[ ['Ale', 'Stout', 'IPA', 'Lager', 'Reviews', 'ABV'] ].values

y = beer['Good'].values

logm2 = linear_model.LogisticRegression(penalty='l1')

logm2.fit(X, y)

#print logm2.predict(X)

print logm2.score(X, y)




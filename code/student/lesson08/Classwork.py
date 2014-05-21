# ________   .__          __                            
# \______ \  |__|  ____  |  | __  ______  ____    ____  
#  |    |  \ |  |_/ ___\ |  |/ / /  ___/ /  _ \  /    \ 
#  |    `   \|  |\  \___ |    <  \___ \ (  <_> )|   |  \
# /_______  /|__| \___  >|__|_ \/____  > \____/ |___|  /
#         \/          \/      \/     \/              \/ 
# GA DataScience Classwork
# Date: 2014-05-21

import pandas as pd
from sklearn.datasets import load_iris
from matplotlib.colors import ListedColormap
from sklearn import neighbors, datasets, feature_selection

from sklearn.neighbors import DistanceMetric

import pandas as pd
import numpy as np

DATA_DIR = '../../../data/'

beer = pd.read_csv(DATA_DIR + 'beer.txt', delimiter="\t")
 
# Various variables we'll need to set intially.
n_neighbors = range(1, 51, 2)
#np.random.seed(1234)

# Load in the data and seperate the class labels and input data

def good(x):
  if x > 4.3:
    return 1
  else:
    return 0

print beer.head(2)

beer = beer.dropna()
beer['Good'] = beer['WR'].apply(good)

beer_types = ['Ale', 'Stout', 'IPA', 'Lager']

for t in beer_types:
	beer[t] = beer['Type'].str.contains(t) * 1

#select = ['Reviews', 'ABV', 'Ale', 'Stout', 'IPA', 'Lager']

select = ['Reviews', 'ABV']

dummies = pd.get_dummies(beer['Brewery'])

#X = beer[select].join(dummies.ix[:, 1:])

X = beer[select]

y = beer['Good']

#print X

# Create the training (and test) set using the rng in numpy
idx = np.random.uniform(0, 1, len(X)) <= 0.3
np.random.shuffle(idx)

X_train, X_test = X[idx], X[idx==False]
y_train, y_test = y[idx], y[idx==False]

# Loop through each neighbors value from 1 to 51 and append
# the scores
'''
scores = []
for n in n_neighbors:
    clf = neighbors.KNeighborsClassifier(n)
    
    # training data set create model
    clf.fit(X_train, y_train)

    #test the model with the testing data
    scores.append(clf.score(X_test, y_test))
'''
#print scores

# we found k = 11 is the best



for k in n_neighbors:
	scores = []
	for num_run in range(10):
	    np.random.shuffle(idx)
	    X_train, X_test = X[idx], X[idx == False]
	    y_train, y_test = y[idx], y[idx == False]
	    clf = neighbors.KNeighborsClassifier(k, weights='uniform')
	    clf.fit(X_train, y_train)
	    scores.append(clf.score(X_test, y_test))
	print np.mean(scores)

#print np.mean(scores)
'''
clf = neighbors.KNeighborsClassifier(21, weights='uniform')
clf.fit(X[:, 2:4], y)



df = pd.DataFrame(iris.data, columns=iris.feature_names)

df = df.ix[:,:4]
df.head()
print df.describe()

'''
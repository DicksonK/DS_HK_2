# ________   .__          __                            
# \______ \  |__|  ____  |  | __  ______  ____    ____  
#  |    |  \ |  |_/ ___\ |  |/ / /  ___/ /  _ \  /    \ 
#  |    `   \|  |\  \___ |    <  \___ \ (  <_> )|   |  \
# /_______  /|__| \___  >|__|_ \/____  > \____/ |___|  /
#         \/          \/      \/     \/              \/ 
# GA DataScience Lab-14
# Date: 2014-06-11

import sklearn
print sklearn.__version__

'''
from sklearn.ensemble import BaggingClassifier
from sklearn.tree import DecisionTreeClassifier
ensemble = BaggingClassifier(base_estimator=DecisionTreeClassifier(),
                             bootstrap=True,
                             bootstrap_features=False)
print ensemble.fit(X_train, y_train)
'''

from sklearn.cross_validation import cross_val_score
from sklearn.datasets import make_blobs
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.tree import DecisionTreeClassifier

X, y = make_blobs(n_samples=10000, n_features=10, centers=100,
     random_state=0)

## DecisionTreeClassifier
clf = DecisionTreeClassifier(max_depth=None, min_samples_split=1,
       random_state=0)
scores = cross_val_score(clf, X, y)
print scores.mean()

## RandomForestClassifier
clf = RandomForestClassifier(n_estimators=10, max_depth=None,
     min_samples_split=1, random_state=0)
scores = cross_val_score(clf, X, y)
print scores.mean()

## ExtraTreesClassifier
clf = ExtraTreesClassifier(n_estimators=10, max_depth=None,
    min_samples_split=1, random_state=0)
scores = cross_val_score(clf, X, y)

print scores

print scores.mean()
# > 0.999



import random
from pandas import read_csv
from sklearn.cross_validation import train_test_split
from sklearn.ensemble.forest import ExtraTreesClassifier
from sklearn import metrics
from sklearn import preprocessing
authorship = read_csv('http://people.stern.nyu.edu/jsimonof/AnalCatData/Data/Comma_separated/authorship.csv')

authors = list(set(authorship.Author.values))
le = preprocessing.LabelEncoder()
le.fit(authors)
authorship['Author_num'] = le.transform(authorship['Author'])

# Create a random variable (random forests work best with a random variable)
authorship['random'] = [random.random() for i in range(841)]

#What are some of the stop words we're looking at?
features = list(authorship.columns)
features
features.remove('Author')
features.remove('Author_num')
features.remove('BookID')

# create a test and training set
x_train, x_test, y_train, y_test = train_test_split(authorship[features], authorship.Author_num.values, test_size=0.4, random_state=123)


# Fit Model
etclf = ExtraTreesClassifier(n_estimators=20)
etclf.fit(x_train, y_train)

# Print Confusion Matrix
print metrics.confusion_matrix(etclf.predict(x_test), y_test)

#print authorship
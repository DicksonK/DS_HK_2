# ________   .__          __                            
# \______ \  |__|  ____  |  | __  ______  ____    ____  
#  |    |  \ |  |_/ ___\ |  |/ / /  ___/ /  _ \  /    \ 
#  |    `   \|  |\  \___ |    <  \___ \ (  <_> )|   |  \
# /_______  /|__| \___  >|__|_ \/____  > \____/ |___|  /
#         \/          \/      \/     \/              \/ 
# GA DataScience Classwork
# Date: 2014-05-28

from sklearn import datasets, metrics, tree, cross_validation
from matplotlib import pyplot as plt
iris = datasets.load_iris()

clf = tree.DecisionTreeClassifier()
y_pred = clf.fit(iris.data, iris.target).predict(iris.data)
print "Number of mislabeled points : %d" % (iris.target != y_pred).sum()
print "Absolutely ridiculously overfit score: %d" % (tree.DecisionTreeClassifier().fit(iris.data,
    iris.target).score(iris.data, iris.target))

metrics.confusion_matrix(iris.target, clf.predict(iris.data))

import random
import pylab as pl
import numpy as np
from sklearn import svm, datasets
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import auc

# import some data to play with
iris = datasets.load_iris()
X = iris.data
y = iris.target
X, y = X[y != 2], y[y != 2]  # Keep also 2 classes (0 and 1)
n_samples, n_features = X.shape
p = range(n_samples)  # Shuffle samples
random.seed(0)
random.shuffle(p)
X, y = X[p], y[p]
half = int(n_samples / 2)

# Add noisy features
np.random.seed(0)
X = np.c_[X, np.random.randn(n_samples, 200 * n_features)]

# Run classifier
classifier = svm.SVC(kernel='linear', probability=True, random_state=0)
probas_ = classifier.fit(X[:half], y[:half]).predict_proba(X[half:])

# Compute Precision-Recall and plot curve
precision, recall, thresholds = precision_recall_curve(y[half:], probas_[:, 1])
area = auc(recall, precision)
print("Area Under Curve: %0.2f" % area)

print metrics.classification_report(iris.target, clf.predict(iris.data))

x_train, x_test, y_train, y_test = cross_validation.train_test_split(iris.data,
    iris.target, test_size=.3)
clf.fit(x_train, y_train)

print metrics.confusion_matrix(y_train, clf.predict(x_train))
print metrics.classification_report(y_train, clf.predict(x_train))

print metrics.confusion_matrix(y_test, clf.predict(x_test))
print metrics.classification_report(y_test, clf.predict(x_test))

clf.set_params(min_samples_leaf=2)
clf.set_params(max_depth=5)
clf.fit(x_train, y_train)
print metrics.confusion_matrix(y_train, clf.predict(x_train))
print metrics.confusion_matrix(y_test, clf.predict(x_test))
print metrics.classification_report(y_test, clf.predict(x_test))
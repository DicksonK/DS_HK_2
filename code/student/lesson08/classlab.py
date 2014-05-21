# ________   .__          __                            
# \______ \  |__|  ____  |  | __  ______  ____    ____  
#  |    |  \ |  |_/ ___\ |  |/ / /  ___/ /  _ \  /    \ 
#  |    `   \|  |\  \___ |    <  \___ \ (  <_> )|   |  \
# /_______  /|__| \___  >|__|_ \/____  > \____/ |___|  /
#         \/          \/      \/     \/              \/ 
# GA DataScience Lab-8
# Date: 2014-05-21

import pandas as pd
from sklearn.datasets import load_iris
from matplotlib.colors import ListedColormap
from sklearn import neighbors, datasets, feature_selection

import pandas as pd
import numpy as np

# Don't show deprecation warnings
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning) 

iris = load_iris()
 
df = pd.DataFrame(iris.data, columns=iris.feature_names)
df['species'] = [iris.target_names[x] for x in iris.target]
df.head(10)

print 'Independent Variables: \n%s' % iris.feature_names

print 'Class Labels: \n%s' % iris.target_names

DATA_DIR = '../../../data/'

# Various variables we'll need to set intially.
n_neighbors = range(1, 51, 2)
np.random.seed(32)

# Load in the data and seperate the class labels and input data
iris = datasets.load_iris()
X = iris.data
y = iris.target

# Create the training (and test) set using the rng in numpy
n = int(len(y) * .7)
ind = np.hstack((np.ones(n, dtype=np.bool), np.zeros(len(y) - n, dtype=np.bool)))
np.random.shuffle(ind)
X_train, X_test = X[ind], X[ind == False]
y_train, y_test = y[ind], y[ind == False]

#print ind

# Or more concisely
idx = np.random.uniform(0, 1, len(X)) <= 0.3
X_train, X_test = X[idx], X[idx==False]
y_train, y_test = y[idx], y[idx==False]

# Loop through each neighbors value from 1 to 51 and append
# the scores
scores = []
for n in n_neighbors:
    clf = neighbors.KNeighborsClassifier(n)
    clf.fit(X_train, y_train)    
    scores.append(clf.score(X_test, y_test))

# Loop through each neighbors value from 1 to 51 and append
# the scores
scores = []
for n in n_neighbors:
    clf = neighbors.KNeighborsClassifier(n)
    clf.fit(X_train, y_train)    
    scores.append(clf.score(X_test, y_test))

print scores
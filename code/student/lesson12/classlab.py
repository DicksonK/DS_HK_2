# ________   .__          __                            
# \______ \  |__|  ____  |  | __  ______  ____    ____  
#  |    |  \ |  |_/ ___\ |  |/ / /  ___/ /  _ \  /    \ 
#  |    `   \|  |\  \___ |    <  \___ \ (  <_> )|   |  \
# /_______  /|__| \___  >|__|_ \/____  > \____/ |___|  /
#         \/          \/      \/     \/              \/ 
# GA DataScience Lab-12
# Date: 2014-06-06

import pandas as pd
import numpy as np

# Don't show deprecation warnings
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning) 

# Set some Pandas options
pd.set_option('max_columns', 30)
pd.set_option('max_rows', 20)

# Store data in a consistent place
DATA_DIR = '../../../data/'
DATA_DIR_2 = '../../../data/baseball/'

from sklearn import cluster
from numpy import random
from pandas import DataFrame, concat
from matplotlib import pyplot as plt

random.seed(1)

classone = DataFrame({
    'x' :random.random(20) + 1,
    'y' : random.random(20) + 1,
    'label' : ['r' for i in range(20)]
})
classtwo = DataFrame({
    'x' :random.random(20) + 1,
    'y' : random.random(20) + 3,
    'label' : ['g' for i in range(20)]
})
classthree = DataFrame({
    'x' :random.random(20) + 3,
    'y' : random.random(20) + 1,
    'label' : ['b' for i in range(20)]
})
classfour = DataFrame({
    'x' :random.random(20) + 3,
    'y' : random.random(20) + 3,
    'label' : ['purple' for i in range(20)]
})
data = concat([classone, classtwo, classthree, classfour])

cls = cluster.k_means(data[ ['x', 'y'] ].values, 4)

data['clusters'] = cls[1]

classfive = DataFrame({
    'x' : random.random(50) * 50 + 100,
    'y' : random.random(50) * 50 + 100,
    'label' : ['orange' for i in range(50)]
})

data = concat([data, classfive])

cls = cluster.k_means(data[ ['x', 'y'] ].values, 5)

data['clusters'] = cls[1]

## Iris Data Application
from sklearn import datasets
iris = datasets.load_iris()
cls = cluster.k_means(iris.data, 3)
cls

from sklearn.metrics import silhouette_score

print silhouette_score(iris.data, cls[1])

#================================================================
'''
Consider a strategy here for the baseball problem: Could Kmeans be used to determine groups
of players, which could then better predict salary?
Consider it's application to bad vs good used car purchases. What data could be used to to 
generate new car groups? What should those car groups represent?

normalize = lambda x: (x - x.mean())/x.std()

cdystonia_grouped.transform(normalize).head()
'''

b2011 = pd.read_csv(DATA_DIR_2 + 'baseball_training_2011.csv')

print b2011.columns

temp = b2011[['G', 'AB', 'R', 'H', 'X2B', 'X3B', 'HR', 'RBI', 'SB', 'CS', 'BB', 'SO', 'IBB', 'HBP', 'SH', 'SF', 'salary']].values

#print temp

max_value = 0
best_k = 0
curr_value = 0



for x in range(2, 100):

	cls_2 = cluster.k_means(temp, x)

	#print silhouette_score(temp, cls_2[len(cls_2)-2])

	curr_value = silhouette_score(temp, cls_2[1])

	if (curr_value > max_value):
		#print "Yes"
		max_value = curr_value
		best_k = x
	print x
	#print curr_value
	#print cls_2[1]


print '\nBest'
print max_value
print best_k

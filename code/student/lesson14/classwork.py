# ________   .__          __                            
# \______ \  |__|  ____  |  | __  ______  ____    ____  
#  |    |  \ |  |_/ ___\ |  |/ / /  ___/ /  _ \  /    \ 
#  |    `   \|  |\  \___ |    <  \___ \ (  <_> )|   |  \
# /_______  /|__| \___  >|__|_ \/____  > \____/ |___|  /
#         \/          \/      \/     \/              \/ 
# GA DataScience Lab-14 classwork
# Date: 2014-06-11

'''

With the authorship data, determine how changing the parameters in the random forest model changes the performance of the model.
Also with the authorship data, feel free to go back to the base random forest classifer included in sklearn, or see how using adaboost does on guess work.
Try timing adaboost in comparison to randomforests to see how performance changes.
Consider building your own bagging algorithm (or get crazy and see if you can write up a simple boosting one) on your own. While this is relatively efficient in python, R users tend to complain a lot about how slow ensemble methods are (from the base packages). Building a strong understanding of these approaches can really move you along in the world of machine learning!
How can ensemble methods be distributed across a cluster of servers? Can they be?

'''

import random
from pandas import read_csv
from sklearn.cross_validation import train_test_split
from sklearn.ensemble.forest import ExtraTreesClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier

from sklearn.cross_validation import cross_val_score

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

print len(features)

# create a test and training set
x_train, x_test, y_train, y_test = train_test_split(authorship[features], authorship.Author_num.values, test_size=0.4, random_state=123)


# Fit Model
etclf = ExtraTreesClassifier(n_estimators=20)
etclf.fit(x_train, y_train)

# Print Confusion Matrix
#print metrics.confusion_matrix(etclf.predict(x_test), y_test)

print "ExtraTreesClassifier"
print cross_val_score(etclf, x_train, y_train).mean()
print "\n"

## RandomForestClassifier

temp_score = 0
temp_est = 0
temp_dep = 0
'''
for num_est in range(68,80):
	print num_est
	for num_dep in range (50,200):
		clf = RandomForestClassifier(n_estimators=num_est, max_depth=num_dep,
		     min_samples_split=1, random_state=0)
		clf.fit(x_train, y_train)



		if cross_val_score(clf, x_train, y_train).mean() > temp_score:
			print cross_val_score(clf, x_train, y_train).mean()
			temp_est = num_est
			temp_dep = num_dep
			#print metrics.confusion_matrix(clf.predict(x_test), y_test)
			#print cross_val_score(clf, x_train, y_train).mean()


print temp_dep
print temp_est

#49
#69

clf = RandomForestClassifier(n_estimators=temp_dep, max_depth=temp_est,
     min_samples_split=1, random_state=0)
clf.fit(x_train, y_train)



cross_val_score(clf, x_train, y_train).mean()

'''
for num_est in range(1,200):
	clf = AdaBoostClassifier(DecisionTreeClassifier(), n_estimators=40)

	scores = cross_val_score(clf, x_train, y_train).mean()
	if scores > temp_score:
		temp_score = scores
		print temp_score


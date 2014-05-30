# ________   .__          __                            
# \______ \  |__|  ____  |  | __  ______  ____    ____  
#  |    |  \ |  |_/ ___\ |  |/ / /  ___/ /  _ \  /    \ 
#  |    `   \|  |\  \___ |    <  \___ \ (  <_> )|   |  \
# /_______  /|__| \___  >|__|_ \/____  > \____/ |___|  /
#         \/          \/      \/     \/              \/ 
# GA DataScience Baseball attemp 1
# Date: 2014-05-19

import pandas as pd
from sklearn import linear_model, metrics

DATA_DIR = '../../../../data/baseball/'

b2011 = pd.read_csv(DATA_DIR + 'baseball_training_2011.csv')
b2012 = pd.read_csv(DATA_DIR + 'baseball_test_2012.csv')

#print b2011['SF']

birth_con_list = ['D.R.', 'USA', 'Venezuela', 'CAN', 'Germany', 'Australia', 'P.R.']

for t in birth_con_list:
	b2011[t] = b2011['birthCountry'].str.contains(t) * 1
	b2012[t] = b2012['birthCountry'].str.contains(t) * 1

#print b2011.columns

b2011['HR*3'] = b2011['HR']*3
b2012['HR*3'] = b2012['HR']*3

#
train_X = b2011[['G', 'AB', 'R', 'H', 'X2B', 'X3B', 'HR', 'RBI', 'SB', 'CS', 'BB', 'SO', 'IBB', 'HBP', 'SH', 'SF']].values
train_y = b2011['salary'].values
#
test_X = b2012[['G', 'AB', 'R', 'H', 'X2B', 'X3B', 'HR', 'RBI', 'SB', 'CS', 'BB', 'SO', 'IBB', 'HBP', 'SH', 'SF']].values
b2012_csv = b2012[['playerID','yearID', 'salary']]
#

'''
R-Squared: 0.191236646455
MSE: 1.70429254467e+13
'''

lm = linear_model.Ridge()
lm.fit(train_X, train_y)
#
# Checking performance, roughly .19
print 'R-Squared:',lm.score(train_X, train_y)
# Checking MSE, roughly terrible
print 'MSE:',metrics.mean_squared_error(lm.predict(train_X), train_y)
#
print '\n'
# Outputting to a csv file
#print "Outputting submission file as 'submission.csv'"
#b2012_csv['predicted'] = lm.predict(test_X)
#b2012_csv.to_csv('submission.csv')



b2011['HR*3'] = b2011['HR']*3
b2012['HR*3'] = b2012['HR']*3
#
train_X = b2011[['HR*3','G', 'AB', 'R', 'H', 'X2B', 'X3B', 'HR', 'RBI', 'SB', 'CS', 'BB', 'SO', 'IBB', 'HBP', 'SH', 'SF']].values
train_y = b2011['salary'].values
#
test_X = b2012[['HR*3','G', 'AB', 'R', 'H', 'X2B', 'X3B', 'HR', 'RBI', 'SB', 'CS', 'BB', 'SO', 'IBB', 'HBP', 'SH', 'SF']].values
b2012_csv = b2012[['playerID','yearID', 'salary']]
#

regr_log = linear_model.LinearRegression()
regr_log.fit(train_X, train_y)
#
# Checking performance, roughly .19
print 'R-Squared:',regr_log.score(train_X, train_y)
# Checking MSE, roughly terrible
print 'MSE:',metrics.mean_squared_error(regr_log.predict(train_X), train_y)
#
print '\n'

#[('yearID', 0.0), ('R', 0.0), ('HR', 0.0), ('RBI', 0.0), ('BB', 0.0), ('IBB', 0.0),
#

b2011['R*3'] = b2011['R']*10
b2012['R*3'] = b2012['R']*10

b2011['HR*3'] = b2011['HR']*10
b2012['HR*3'] = b2012['HR']*10

train_X = b2011[['R*3', 'HR*3', 'RBI', 'BB', 'IBB']].values
train_y = b2011['salary'].values
#
test_X = b2012[['R*3', 'HR*3', 'RBI', 'BB', 'IBB']].values
b2012_csv = b2012[['playerID','yearID', 'salary']]
#

regr_log = linear_model.LinearRegression()
regr_log.fit(train_X, train_y)
#
# Checking performance, roughly .19
print 'R-Squared:',regr_log.score(train_X, train_y)
# Checking MSE, roughly terrible
print 'MSE:',metrics.mean_squared_error(regr_log.predict(train_X), train_y)
#
print '\n'

train_X = b2011[['G', 'AB', 'R', 'H', 'X2B', 'X3B', 'HR', 'RBI', 'SB', 'CS', 'BB', 'SO', 'IBB', 'HBP', 'SH', 'SF']].values
train_y = b2011['salary'].values
#
test_X = b2012[['G', 'AB', 'R', 'H', 'X2B', 'X3B', 'HR', 'RBI', 'SB', 'CS', 'BB', 'SO', 'IBB', 'HBP', 'SH', 'SF']].values
b2012_csv = b2012[['playerID','yearID', 'salary']]
#

'''
R-Squared: 0.191236646455
MSE: 1.70429254467e+13
'''

lm2 = linear_model.Ridge(alpha=0.9, copy_X=True, fit_intercept=True, max_iter=None,
      normalize=False, solver='auto', tol=0.001)
lm2.fit(train_X, train_y)
#
# Checking performance, roughly .19
print 'R-Squared:',lm2.score(train_X, train_y)
# Checking MSE, roughly terrible
print 'MSE:',metrics.mean_squared_error(lm2.predict(train_X), train_y)
#
print '\n'


#
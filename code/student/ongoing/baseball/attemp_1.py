# ________   .__          __                            
# \______ \  |__|  ____  |  | __  ______  ____    ____  
#  |    |  \ |  |_/ ___\ |  |/ / /  ___/ /  _ \  /    \ 
#  |    `   \|  |\  \___ |    <  \___ \ (  <_> )|   |  \
# /_______  /|__| \___  >|__|_ \/____  > \____/ |___|  /
#         \/          \/      \/     \/              \/ 
# GA DataScience Baseball attemp 1
# Date: 2014-05-19

import pandas as pd
from sklearn import linear_model, metrics, feature_selection

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


#============== 

train_X = b2011[['weight', 'height', 'HR']].values
train_y = b2011['salary'].values
#
test_X = b2012[['weight', 'height', 'HR']].values
b2012_csv = b2012[['playerID','yearID', 'salary']]


try_model = linear_model.Ridge(alpha=0.5)
try_model.fit(train_X, train_y)
#
#
# Checking performance, roughly .19
print 'R-Squared:',try_model.score(train_X, train_y)
# Checking MSE, roughly terrible
print 'MSE:',metrics.mean_squared_error(try_model.predict(train_X), train_y)
#
print '\n'

b2011_num = b2011._get_numeric_data()
b2011_num = b2011_num.dropna(axis=1)
salary = b2011_num['salary']
b2011_num = b2011_num.drop(['salary'],1)

fp_value = feature_selection.univariate_selection.f_regression(b2011_num, salary)
p_value = zip(b2011_num.columns.values,fp_value[1])
print sorted(p_value,key=lambda x: x[1])

'''
[('birthYear', 7.9835736658094558e-24), ('lahmanID', 1.4981569016438451e-19), ('RBI', 2.3423696853156987e-19), 
('HR', 2.1781665697039328e-18), ('HR*3', 2.1781665697043661e-18), ('BB', 1.6894807429368926e-16), ('IBB', 1.3518139523605894e-15), 
('GIDP', 3.3671114818920081e-15), ('R', 5.5242446285898393e-15), ('R*3', 5.524244628591959e-15), ('H', 3.8926959093146868e-14), 
('AB', 1.4713547901030828e-13), ('X2B', 2.0126968058913591e-13), ('SF', 1.5442454315422193e-12), ('SO', 4.5120325955358599e-09), 
('weight', 1.4742817482563304e-07), ('G', 2.7770222456756615e-07), ('G_batting', 2.7770222456756615e-07), ('G_old', 2.7770222456756615e-07), 
('HBP', 4.3295685791807793e-06), ('height', 0.024490536292798772), ('birthDay', 0.025050282411500129), ('SB', 0.056525713590704518), 
('USA', 0.15687171768959324), ('CS', 0.16214722686250782), ('X3B', 0.1758587098104967), ('birthMonth', 0.22107508528233263), 
('yearID', nan), ('stint', nan), ('CAN', 0.32990232740163583), ('D.R.', 0.36608449516973751), ('P.R.', 0.40215980142108221), 
('SH', 0.40730098572339368), ('Australia', 0.6039726888491268), ('Venezuela', 0.81469953950938012), ('Germany', 0.88649615124230063)]
'''

train_X = b2011[['Germany', 'Venezuela', 'Australia', 'SH', 'P.R.', 'D.R.', 'CAN', 'birthMonth', 'X3B', 'CS', 'USA', 'SB', 
	'birthDay', 'height', 'HBP']].values
train_y = b2011['salary'].values
#
test_X = b2012[['Germany', 'Venezuela', 'Australia', 'SH', 'P.R.', 'D.R.', 'CAN', 'birthMonth', 'X3B', 'CS', 'USA', 'SB', 
	'birthDay', 'height', 'HBP']].values
b2012_csv = b2012[['playerID','yearID', 'salary']]


try_model = linear_model.Ridge()
try_model.fit(train_X, train_y)
#
#
# Checking performance, roughly .19
print 'R-Squared:',try_model.score(train_X, train_y)
# Checking MSE, roughly terrible
print 'MSE:',metrics.mean_squared_error(try_model.predict(train_X), train_y)
#
print '\n'


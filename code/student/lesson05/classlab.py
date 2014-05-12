# ________   .__          __                            
# \______ \  |__|  ____  |  | __  ______  ____    ____  
#  |    |  \ |  |_/ ___\ |  |/ / /  ___/ /  _ \  /    \ 
#  |    `   \|  |\  \___ |    <  \___ \ (  <_> )|   |  \
# /_______  /|__| \___  >|__|_ \/____  > \____/ |___|  /
#         \/          \/      \/     \/              \/ 
# GA DataScience Lab-5
# Date: 2014-05-12


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from numpy import array, dot
from scipy.linalg import inv

from numpy import log
from sklearn import linear_model

prediction = np.array([1, 2, 3])
actual = np.array([1.12, 1.89, 3.02])

#print np.mean((prediction - actual) ** 2)

# Set some Pandas options
pd.set_option('max_columns', 30)
pd.set_option('max_rows', 20)

# Store data in a consistent place
DATA_DIR = '../../../data/'


X = array([ [1, 1], [1, 2], [1, 3], [1, 4] ])
y = array([ [1], [2], [3], [4] ])

#print X
#print X.T

#print dot(X.T, X)

n = inv(dot(X.T, X))

#print n

k = dot(X.T, y)

#print k

coef_ = dot(n, k)

#print coef_

def regression(input, response):
    return dot(inv(dot(input.T, input)), dot(input.T, response))

mammals = pd.read_csv(DATA_DIR + 'mammals.csv')
#print mammals.describe()

#print mammals.head()
'''
# figure(figsize=(20,8))
plt.scatter(mammals['body'], mammals['brain'])
plt.show()

# figure(figsize=(20,8))
plt.hist(mammals['body'], bins=range(0, 10000, 100))
plt.show()

# figure(figsize=(20,8))
plt.hist(mammals['brain'], bins=range(0, 10000, 100))
plt.show()
'''

# figure(figsize=(20,8))

mammals['log_body'] = log(mammals['body'])
mammals['log_brain'] = log(mammals['brain'])

plt.scatter(mammals['log_body'], mammals['log_brain'])



# Make the model object
regr = linear_model.LinearRegression()

# Fit the data
body = [[x] for x in mammals['body'].values]
brain = mammals['brain'].values

regr.fit(body, brain)

print "Body v.s. Brain"

# Display the coefficients:
print regr.coef_

# Display our SSE:
print np.mean((regr.predict(body) - brain) ** 2)

# Scoring our model (closer to 1 is better!)
print regr.score(body, brain)

#=======================================================================
# Class Work Now
#=======================================================================

#===== Q1

mammals_2 = pd.read_csv(DATA_DIR + 'mammals.csv')

mammals_2['log_body'] = log(mammals_2['body'])
mammals_2['log_brain'] = log(mammals_2['brain'])

# Make the model object
regr_log = linear_model.LinearRegression()

# Fit the data
log_body = [[x] for x in mammals_2['log_body'].values]
log_brain = mammals_2['log_brain'].values

regr_log.fit(log_body, log_brain)

print "\nLog Body v.s. Log Brain"

# Display the coefficients:
print regr_log.coef_

# Display our SSE:
print np.mean((regr_log.predict(log_body) - log_brain) ** 2)

# Scoring our model (closer to 1 is better!)
print regr_log.score(log_body, log_brain)

#===== Q2
'''
Using your aggregate data compiled from nytimes1-30.csv, write a python script that 
determines the best model predicting CTR based off of age and gender. Since gender is
not actually numeric (it is binary), investigate ways to vectorize this feature. Clue:
you may want two features now instead of one.
'''

nytimes = pd.read_csv(DATA_DIR + 'nyagg.csv')

# Make the model object
CTR_modle = linear_model.LinearRegression()

# Fit the data
X = nytimes[['Age', 'Gender']].values

CTR = nytimes['Ctr'].values

CTR_modle.fit(X, CTR)

print "\nAge & Gender"

print CTR_modle.coef_
print np.mean((CTR_modle.predict(X) - CTR) ** 2)
print CTR_modle.score(X, CTR)

#===== Q3a
'''
Compare this practice to making two separate models based on Gender, with Age as your one 
feature predicting CTR. How are your results different? Which results would you be more 
confident in presenting to your manager? Why's that?
'''
nytimes = pd.read_csv(DATA_DIR + 'nyagg.csv')

# Make the model object
CTR_modle = linear_model.LinearRegression()

# Fit the data
X = nytimes[['Gender']].values

CTR = nytimes['Ctr'].values

CTR_modle.fit(X, CTR)

print "\nGender"

print CTR_modle.coef_
print np.mean((CTR_modle.predict(X) - CTR) ** 2)
print CTR_modle.score(X, CTR)

#===== Q3b

nytimes = pd.read_csv(DATA_DIR + 'nyagg.csv').query('Age >= 20')

#print nytimes.head()

#nytimes = nytimes.query('Age > 0')

# Make the model object
CTR_modle = linear_model.LinearRegression()

# Fit the data\
X = nytimes[['Age']].values

CTR = nytimes['Ctr'].values

CTR_modle.fit(X, CTR)

print "\nAge"

print CTR_modle.coef_
print np.mean((CTR_modle.predict(X) - CTR) ** 2)
print CTR_modle.score(X, CTR)

#===== Q4
'''
IMO the gender doesn't really matter to the CTR prediction

'''
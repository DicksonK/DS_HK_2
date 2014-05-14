# ________   .__          __                            
# \______ \  |__|  ____  |  | __  ______  ____    ____  
#  |    |  \ |  |_/ ___\ |  |/ / /  ___/ /  _ \  /    \ 
#  |    `   \|  |\  \___ |    <  \___ \ (  <_> )|   |  \
# /_______  /|__| \___  >|__|_ \/____  > \____/ |___|  /
#         \/          \/      \/     \/              \/ 
# GA DataScience Lab-6
# Date: 2014-05-14

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Set some Pandas options
pd.set_option('max_columns', 30)
pd.set_option('max_rows', 20)

# Store data in a consistent place
DATA_DIR = '../../../data/'

from sklearn import linear_model

mammals = pd.read_csv(DATA_DIR + 'mammals.csv')

lm = linear_model.LinearRegression()
log_lm = linear_model.LinearRegression()

from numpy import arange,array,ones#,random,linalg
from pylab import plot,show
from scipy import stats

xi = arange(0,9)
A = array([ xi, ones(9)])
# linearly generated sequence
y = [19, 20, 20.5, 21.5, 22, 23, 23, 25.5, 24]
slope, intercept, r_value, p_value, std_err = stats.linregress(xi,y)

# H0
print 'response mean', np.mean(y)

# Standard Deviation of Y
print 'standard Deviation of Y', np.std(y)

# Coefficient of Determination
print 'r-squared value', r_value**2

# Is the statistic significant?
print 'p_value', p_value

print 'standard deviation of error terms', std_err

line = slope*xi+intercept
plot(xi,line,'r-',xi,y,'o')
#show()

mammals['body_squared'] = mammals['body']**2
body_squared = [ [x, y] for x,y in zip(mammals['body'].values, mammals['body_squared'].values)]

body = [ [x] for x in mammals['body'].values]
brain = mammals['brain'].values

ridge = linear_model.Ridge()
ridge.fit(body_squared, brain)

print ((ridge.coef_[1] * mammals['body'])**2) + ((ridge.coef_[0] * mammals['body'])) + ridge.intercept_

print ridge.score

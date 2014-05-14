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

from numpy import arange,array,ones#,random,linalg
from pylab import plot,show
from scipy import stats

# Set some Pandas options
pd.set_option('max_columns', 30)
pd.set_option('max_rows', 20)

# Store data in a consistent place
DATA_DIR = '../../../data/'

from sklearn import linear_model

mammals = pd.read_csv(DATA_DIR + 'mammals.csv')

car_2col = pd.read_csv(DATA_DIR + 'cars1920.csv')

car_set = pd.read_csv(DATA_DIR + 'cars93.csv')

print car_2col.head()

lm = linear_model.LinearRegression()
log_lm = linear_model.LinearRegression()

mammals['body_squared'] = mammals['body']**2
body_squared = [ [x, y] for x,y in zip(mammals['body'].values, mammals['body_squared'].values)]

car_2col['dist_sq'] = car_2col['dist']**2
dist_sq = [ [x, y] for x,y in zip(car_2col['dist'].values, car_2col['dist_sq'].values)]

body = [ [x] for x in mammals['body'].values]
brain = mammals['brain'].values

speed = [ [x] for x in car_2col['speed'].values]
dist = car_2col['dist'].values

ridge = linear_model.Ridge()
ridge.fit(body_squared, brain)

ridge2 = linear_model.Ridge()
ridge2.fit(dist_sq, speed)

print "\n"

print ridge.coef_

print ridge.coef_[0] 

print ridge.coef_[1] 

#print ((ridge.coef_[1] * mammals['body'])**2) + ((ridge.coef_[0] * mammals['body'])) + ridge.intercept_

print ridge2.coef_

print ridge2.coef_[0,0]

print ((ridge2.coef_[0,1] * car_2col['dist'])**2) + ((ridge2.coef_[0, 0] * car_2col['dist'])) + ridge2.intercept_

from sklearn import feature_selection

def f_regression_feature_selection(input, response):    
    feature_selection.univariate_selection.f_regression(input, response)    

#=========================================================
#Predicting City and Highway MPG.
#=========================================================

#MPG.city
car_set['cylinders_sq'] = car_set['cylinders']**2
cylinders_sq = [ [x, y] for x,y in zip(car_set['cylinders'].values, car_set['cylinders_sq'].values)]

mpg_city = [ [x] for x in car_set['dist'].values]
speed = car_2col['speed'].values

ridge2 = linear_model.Ridge()
ridge2.fit(speed_sq, dist)

print car_set.columns
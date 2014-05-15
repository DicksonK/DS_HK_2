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
from sklearn.feature_selection.univariate_selection import f_regression

# Set some Pandas options
pd.set_option('max_columns', 30)
pd.set_option('max_rows', 20)

# Store data in a consistent place
DATA_DIR = '../../../data/'

from sklearn import linear_model

mammals = pd.read_csv(DATA_DIR + 'mammals.csv')

car_2col = pd.read_csv(DATA_DIR + 'cars1920.csv')

car_set = pd.read_csv(DATA_DIR + 'cars93_edit.csv')

print car_2col.head()

lm = linear_model.LinearRegression()
log_lm = linear_model.LinearRegression()

mammals['body_squared'] = mammals['body']**2
body_squared = [ [x, y] for x,y in zip(mammals['body'].values, mammals['body_squared'].values)]

car_2col['speed_sq'] = car_2col['speed']**2
car_2col['speed_sq3'] = car_2col['speed']**3
car_2col['speed_sq4'] = car_2col['speed']**4
car_2col['speed_sq5'] = car_2col['speed']**5
car_2col['speed_sq6'] = car_2col['speed']**6
car_2col['speed_sq7'] = car_2col['speed']**7
car_2col['speed_sq8'] = car_2col['speed']**8
car_2col['speed_sq9'] = car_2col['speed']**9
car_2col['speed_sq10'] = car_2col['speed']**10

speed_sq = [ [x, y, z, z1, z2, z3, z4, z5, z6, z7] for x,y,z,z1,z2,z3,z4,z5,z6,z7 in zip(car_2col['speed'].values, car_2col['speed_sq'].values, car_2col['speed_sq3'].values, car_2col['speed_sq4'].values, car_2col['speed_sq5'].values, car_2col['speed_sq6'].values, car_2col['speed_sq7'].values, car_2col['speed_sq8'].values, car_2col['speed_sq9'].values, car_2col['speed_sq10'].values)]

body = [ [x] for x in mammals['body'].values]
brain = mammals['brain'].values

dist = [ [x] for x in car_2col['dist'].values]
speed = car_2col['speed'].values

ridge = linear_model.Ridge()
ridge.fit(body_squared, brain)

ridge2 = linear_model.Ridge()
ridge2.fit(speed_sq, dist)

print "\n"

#print ((ridge.coef_[1] * mammals['body'])**2) + ((ridge.coef_[0] * mammals['body'])) + ridge.intercept_

print ridge2.coef_

print ((ridge2.coef_[0,1] * car_2col['speed'])**2) + ((ridge2.coef_[0, 0] * car_2col['speed'])) + ridge2.intercept_

from sklearn import feature_selection  

print ridge2.score(speed_sq, dist)

#*** Square the input variable

def f_regression_feature_selection(input, response):    
    print feature_selection.univariate_selection.f_regression(input, response)  

#=========================================================
#Predicting City and Highway MPG.
#=========================================================

print "\n\n================================"
print "Predicting City and Highway MPG."
print "================================"
#print car_set.columns

#print car_set.head()

car_set_temp = car_set

print isinstance(int(car_set['Cylinders'][0]), int)

for col in car_set.columns:
	#print "Check --> : "  + str(col) +  " " + str(isinstance(car_set[col][0], (int, long)))
	if isinstance(car_set[col][0], (str)):
		#print col
		car_set_temp = car_set_temp.drop([col],1)

#print car_set_temp.describe

car_set_temp = car_set_temp.fillna(0)

#Need to filter out NaN

print car_set_temp.columns

f, p = f_regression(car_set_temp, car_set['MPG.city'])

print f
print p

print car_set_temp.columns.values.tolist()

#car_set_f = pd.dataframe({car_set_temp.columns.tolist()})

car_set_f = pd.DataFrame(car_set_temp.columns.values.tolist(), columns=['col_head'])

car_set_f['f'] = f

car_set_f['p'] = p

print car_set_f.sort(['f'], ascending=[0])

'''
#MPG.city
car_set['Cylinders_sq'] = car_set['Cylinders']**2

cylinders_sq = [ [x, y] for x,y in zip(car_set['Cylinders'].values, car_set['Cylinders_sq'].values)]

mpg_city = [ [x] for x in car_set['dist'].values]
speed = car_2col['speed'].values

ridge2 = linear_model.Ridge()
ridge2.fit(speed_sq, dist)

'''
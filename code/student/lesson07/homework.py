# ________   .__          __                            
# \______ \  |__|  ____  |  | __  ______  ____    ____  
#  |    |  \ |  |_/ ___\ |  |/ / /  ___/ /  _ \  /    \ 
#  |    `   \|  |\  \___ |    <  \___ \ (  <_> )|   |  \
# /_______  /|__| \___  >|__|_ \/____  > \____/ |___|  /
#         \/          \/      \/     \/              \/ 
# GA DataScience HW-7
# Date: 2014-05-19

import pandas as pd
import matplotlib.pyplot as plt
from numpy import log, exp, mean
from sklearn import linear_model, feature_selection

DATA_DIR = '../../../data/'


baseball = pd.read_csv(DATA_DIR + 'baseball_new.csv')

#print baseball

baseball_clean = baseball._get_numeric_data()
baseball_clean = baseball_clean.fillna(0)
salary = baseball_clean['salary']
baseball_clean = baseball_clean.drop(['salary'],1)

#print baseball_clean

fp_value = feature_selection.univariate_selection.f_regression(baseball_clean, salary)
p_value = zip(baseball_clean.columns.values,fp_value[1])

print p_value

print "\n"

print sorted(p_value,key=lambda x: x[1])
'''
Ploting result
[ ["HR", "RBI", 'R', "G", "SB", "salary", 'height', 'weight', 'yearID'] ]
HR vs salary

If we only look at 2011 data, seems all the data culster at the left corner

P-value:
< 0.0001
Equation:
HR = 5.64447e-07*Salary + 3.28933

Coefficients
Term	Value 	StdErr 	t-value 	p-value
Salary 	5.644e-07 	6.308e-08 	8.94864 < 0.0001
intercept 3.28933 	0.361801 	9.09155 < 0.0001


'''
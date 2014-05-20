# ________   .__          __                            
# \______ \  |__|  ____  |  | __  ______  ____    ____  
#  |    |  \ |  |_/ ___\ |  |/ / /  ___/ /  _ \  /    \ 
#  |    `   \|  |\  \___ |    <  \___ \ (  <_> )|   |  \
# /_______  /|__| \___  >|__|_ \/____  > \____/ |___|  /
#         \/          \/      \/     \/              \/ 
# GA DataScience Baseball attemp 1
# Date: 2014-05-19


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn import linear_model, feature_selection

import re

# Store data in a consistent place
DATA_DIR = '../../../../data/'

baseball = pd.read_csv(DATA_DIR + 'baseball_new.csv')

#baseball = baseball.dropna(axis=1)

baseball = baseball.fillna(0)

#print baseball.columns

#print baseball.columns

X = baseball[ ["HR", "RBI", 'R', "G", "SB", 'height', 'weight', 'yearID'] ].values

y = baseball[ ['salary'] ].values

#print "==="

print y

#print baseball.describe()

bb_logm = linear_model.LogisticRegression()

bb_logm.fit(X, y)

print bb_logm.score(X, y)

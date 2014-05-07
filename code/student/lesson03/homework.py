# ________   .__          __                            
# \______ \  |__|  ____  |  | __  ______  ____    ____  
#  |    |  \ |  |_/ ___\ |  |/ / /  ___/ /  _ \  /    \ 
#  |    `   \|  |\  \___ |    <  \___ \ (  <_> )|   |  \
# /_______  /|__| \___  >|__|_ \/____  > \____/ |___|  /
#         \/          \/      \/     \/              \/ 
# GA DataScience HW-3
# Date: 2014-05-05

import sys
import os
import timeit
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from matplotlib.transforms import Bbox
from matplotlib.path import Path

from sets import Set
from datetime import datetime
start = timeit.default_timer()

targetURL = 'http://stat.columbia.edu/~rachel/datasets/'

runtimeSavingMode = False

print "Mode: " + str(runtimeSavingMode)
print "Currently importing: Dataset #01 out of 31"

dataset = pd.read_csv(targetURL + 'nyt1.csv', sep=',')

if not runtimeSavingMode:
	for num in range(2,32):
		print "Currently importing: Dataset #" + str(num).zfill(2) + " out of 31"
		dataset = dataset.append(pd.read_csv(targetURL + 'nyt' + str(num) + '.csv', sep=','),ignore_index = True)

stop = timeit.default_timer()

agg_dataset = dataset[ ['Age', 'Gender', 'Signed_In', 'Clicks', 'Impressions'] ].groupby(['Age', 'Gender', 'Signed_In']).agg([np.sum])

agg_dataset['CTR'] = agg_dataset['Clicks']/agg_dataset['Impressions']

print stop - start
if not runtimeSavingMode:
	agg_dataset.to_csv('nytimes_aggregation.csv')

print dataset.describe()

print agg_dataset.describe()


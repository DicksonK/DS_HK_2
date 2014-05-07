# ________   .__          __                            
# \______ \  |__|  ____  |  | __  ______  ____    ____  
#  |    |  \ |  |_/ ___\ |  |/ / /  ___/ /  _ \  /    \ 
#  |    `   \|  |\  \___ |    <  \___ \ (  <_> )|   |  \
# /_______  /|__| \___  >|__|_ \/____  > \____/ |___|  /
#         \/          \/      \/     \/              \/ 
# GA DataScience HW-2 (using Panads)
# Date: 2014-05-03

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

# File header
# Age
# Gender
# Impressions
# Clicks
# Signed_In

start = timeit.default_timer()

pd.set_option('display.mpl_style', 'default') # Make the graphs a bit prettier

#Read the datafile and save that in to dataset where Age is greater than 0
dataset = pd.read_csv('nytimes.csv', sep=',').query('Age > 0')

#Group Age, Gender, Signed_in
agg_result = dataset.groupby(['Age', 'Gender', 'Signed_In'])

#Print the aggregate result with sum, mean, max
print agg_result.aggregate([np.sum, np.mean, np.max])

print "Done!"

stop = timeit.default_timer()

print "Runtime: ", stop - start 

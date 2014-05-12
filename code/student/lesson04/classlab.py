# ________   .__          __                            
# \______ \  |__|  ____  |  | __  ______  ____    ____  
#  |    |  \ |  |_/ ___\ |  |/ / /  ___/ /  _ \  /    \ 
#  |    `   \|  |\  \___ |    <  \___ \ (  <_> )|   |  \
# /_______  /|__| \___  >|__|_ \/____  > \____/ |___|  /
#         \/          \/      \/     \/              \/ 
# GA DataScience Lab-4
# Date: 2014-05-07

import pandas as pd
import numpy as np

data_dir = '../../../data/'

# Set some Pandas options
pd.set_option('max_columns', 30)
pd.set_option('max_rows', 20)

# Store data in a consistent place

'''
lg             League
G              Games
AB             At Bats
R              Runs
H              Hits
X2B             Doubles
X3B             Triples
HR             Homeruns
RBI            Runs Batted In
SB             Stolen Bases
CS             Caught stealing
BB             Base on Balls
SO             Strikeouts
IBB            Intentional walks
HBP            Hit by pitch
SH             Sacrifices
SF             Sacrifice flies
GIDP           Grounded into double plays
'''

stats = pd.read_csv(data_dir + 'baseball.csv', sep=',', index_col='id')

#print stats

slg = lambda x: (x['h']-x['X2b']-x['X3b']-x['hr'] + 2*x['X2b'] + 3*x['X3b'] + 4*x['hr'])/(x['ab']+1e-6)

stats['slg'] = stats.apply(slg, axis = 1)

stats['slg_diff'] = stats.slg - stats.slg.max()

print stats[:10]

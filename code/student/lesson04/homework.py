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
from functools import partial

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

dataset = pd.read_csv(data_dir + 'baseball.csv', sep=',', index_col='id')

#print dataset[:10]

def add_column(tag_dataset, col_name, col_formula):
	tag_dataset[col_name] = tag_dataset.apply(col_formula, axis = 1)
	return tag_dataset

dataset = add_column(dataset, 'BBR', partial(lambda x: (x['h']/(x['ab'] + 1e-6))))
dataset = add_column(dataset, 'BSR', partial(lambda x: (x['sb']/(x['r'] + 1e-6))))
dataset = add_column(dataset, 'SBR', partial(lambda x: (x['cs']/(x['r'] + 1e-6))))
dataset = add_column(dataset, 'CSR', partial(lambda x: (x['cs']/(x['r'] + 1e-6))))

'''
Get the % of hit rate
then offset the strikeouts rate
factor the number of Homeruns
times 
the stolen bases rate
offset by the caught stealrate
'''
dataset = add_column(dataset, 'BPF', partial(lambda x: (((x['BBR'] - x['BSR']) * x['hr']) * ((x['SBR'] - x['CSR'])))))

result = dataset[:1].sort('BPF', ascending = False)

print "The best player is: " + result['player'][0]







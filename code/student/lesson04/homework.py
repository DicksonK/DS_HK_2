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

dataset = pd.read_csv(data_dir + 'baseball.csv', sep=',', index_col='id')

#print dataset[:10]

slg = lambda x: (x['h']-x['X2b']-x['X3b']-x['hr'] + 2*x['X2b'] + 3*x['X3b'] + 4*x['hr'])/(x['ab']+1e-6)

best_bat_rate = lambda x: (x['h']/(x['ab'] + 1e-6))

best_so_rate = lambda x: (x['so']/(x['ab'] + 1e-6))

stolen_base_rate = lambda x: (x['sb']/(x['r'] + 1e-6))

caught_steal_rate = lambda x: (x['cs']/(x['r'] + 1e-6))

dataset['BBR'] = dataset.apply(best_bat_rate, axis = 1)

dataset['BSR'] = dataset.apply(best_so_rate, axis = 1)

dataset['SBR'] = dataset.apply(stolen_base_rate, axis = 1)

dataset['CSR'] = dataset.apply(caught_steal_rate, axis = 1)

'''
Get the % of hit rate
then offset the strikeouts rate
factor the number of Homeruns
times 
the stolen bases rate
offset by the caught stealrate
'''
best_play_formule = lambda x: (((x['BBR'] - x['BSR']) * x['hr']) * ((x['SBR'] - x['CSR'])))

dataset['BPF'] = dataset.apply(best_play_formule, axis = 1)


print max(dataset['BBR'])

print max(dataset['BPF'])

#print dataset.query('BBLSR >0.2')

#print dataset.sort('BPF', ascending = False)

result = dataset[:1].sort('BPF', ascending = False)

print "The best player is: " + result['player'][0]







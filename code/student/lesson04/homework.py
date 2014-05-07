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


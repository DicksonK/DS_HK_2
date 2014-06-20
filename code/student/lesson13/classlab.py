# ________   .__          __                            
# \______ \  |__|  ____  |  | __  ______  ____    ____  
#  |    |  \ |  |_/ ___\ |  |/ / /  ___/ /  _ \  /    \ 
#  |    `   \|  |\  \___ |    <  \___ \ (  <_> )|   |  \
# /_______  /|__| \___  >|__|_ \/____  > \____/ |___|  /
#         \/          \/      \/     \/              \/ 
# GA DataScience Lab-13
# Date: 2014-06-09

import pandas as pd
import numpy as np
import matplotlib
import matplotlib.pyplot as plt

# Set some Pandas options
pd.set_option('max_columns', 30)
pd.set_option('max_rows', 20)

# Store data in a consistent place
DATA_DIR = '../../../data/'

# Libraries and seed set
from pandas import DataFrame
from sklearn.decomposition import PCA
np.random.seed(500)

recorders   = DataFrame({'locations' : ('A', 'B', 'C', 'D'), 'X' : (0, 0, 1, 1), 'Y' : (0, 1, 1, 0)})
locations   = np.array([ [.3, .5], [.8, .2] ])
intensities = np.array([
        [np.sin(np.array(range(100)) * np.pi/10) + 1.2],
        [np.cos(np.array(range(100)) * np.pi/15) * .7 + .9]]).T
distances   = np.array([
    np.sqrt((locations[0] - recorders.X[i])**2 + (locations[1] - recorders.Y[i])**2) for i in range(4)]).T

data = np.dot(intensities, np.exp(-2*distances))
data_transposed = data.T

row_means = [np.mean(i) for i in data_transposed]
data_transposed_scaled = np.array([data_transposed[i][0] - row_means[i] for i in range(4)])

'''

Print

'''

print data_transposed_scaled

pca = PCA()
pca.fit(data_transposed_scaled)

variance = pca.explained_variance_ratio_
readable_variance = variance * (1/variance[0])

from sklearn.decomposition import PCA, KernelPCA
from sklearn.datasets import make_circles

np.random.seed(0)

X, y = make_circles(n_samples=400, factor=.3, noise=.05)

kpca = KernelPCA(kernel="rbf", fit_inverse_transform=True, gamma=10)
X_kpca = kpca.fit_transform(X)
X_back = kpca.inverse_transform(X_kpca)
pca = PCA()
X_pca = pca.fit_transform(X)

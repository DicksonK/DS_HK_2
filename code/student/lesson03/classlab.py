# ________   .__          __                            
# \______ \  |__|  ____  |  | __  ______  ____    ____  
#  |    |  \ |  |_/ ___\ |  |/ / /  ___/ /  _ \  /    \ 
#  |    `   \|  |\  \___ |    <  \___ \ (  <_> )|   |  \
# /_______  /|__| \___  >|__|_ \/____  > \____/ |___|  /
#         \/          \/      \/     \/              \/ 
# GA DataScience Class lab
# Date: 2014-05-05

from numpy import *

from numpy import array, dot
from numpy.linalg import inv

import timeit

X = array([[1, 1], [1, 2], [1, 3], [1, 4]])
y = array([[1], [2], [3], [4]])

n = inv(dot(X.T, X))
print n

k = dot(X.T, y)
print k

coef_ = dot(n, k)

print coef_

def regression(input, response):
	return dot(inv(dot(input.T, input)), dot(input.T, response))

print regression(X,y)

arrayOne = arange(15).reshape(3, 5)
print arrayOne

arrayTwo = arange(15).reshape(5, 3)
print arrayTwo

vector = array([10, 15, 20])
print vector

matrixOne = matrix('1 2 3; 4 5 6')
print matrixOne

matrixTwo = matrix('1 2; 3 4; 5 6')
print matrixTwo

a1 = array([ [1, 2], [3, 4] ])
a2 = array([ [1, 3], [2, 4] ])
m1 = matrix('1 2; 3 4')
m2 = matrix('1 3; 2 4')

print a1 * a2

print m1 * m2

print dot(a1, a2)

print dot(m1, m2)

print a1.T
start = timeit.default_timer()
print eye(5)
stop = timeit.default_timer()

print stop - start

#9.53674316406e-07
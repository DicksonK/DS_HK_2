# ________   .__          __                            
# \______ \  |__|  ____  |  | __  ______  ____    ____  
#  |    |  \ |  |_/ ___\ |  |/ / /  ___/ /  _ \  /    \ 
#  |    `   \|  |\  \___ |    <  \___ \ (  <_> )|   |  \
# /_______  /|__| \___  >|__|_ \/____  > \____/ |___|  /
#         \/          \/      \/     \/              \/ 
# GA DataScience Class lab
# Date: 2014-04-30

#Display purpose
import numpy as np

vector_1 = [1, 2, 3]

matrix_1 = [ [1, 3, 9, 2], 
           [2, 4, 6, 8] ]

matrix_2 = [ [2, 1], 
		     [3, 2], 
		     [6, 0],
		     [5, 4] ]

matrix_3 = [ [1, 2, 3], 
           [4, 5, 6], 
           [7, 8, 9] ]

"""
Question: vectorMatrix multiplication

Answer:

"""

def vectorMartix_mult(matrix, vector):
	result = [[0]*len(matrix) for x in range(1)]
	for row in range(len(matrix)):
		sum = 0
		for moving in range(len(matrix[row])):
			sum += matrix[row][moving] * vector[moving]
		result[0][row] = sum
	return result

print np.matrix(vectorMartix_mult(matrix_3, vector_1))

"""
Question: matrixMatrix multiplication

Answer: See comment
"""

def matrixMartix_mult(matrix1, matrix2):
	if len(matrix1[0]) == len(matrix2):
		result = [[0 for row in range(len(matrix1))] for col in range(len(matrix2[0]))]
		#Get size of the each row from the first matrix
		for col in range(len(matrix1)):
		#Get size of each each col from the second matrix
			for row in range(len(matrix2[0])):
		#Loop thu each set of value, add their multiple 
				for moving in range(len(matrix1[0])):
					result[col][row] += matrix1[col][moving] * matrix2[moving][row]
		return result
	else:
		return "Error"


print np.matrix(matrixMartix_mult(matrix_1, matrix_2))

"""
Question: 
Write a function that creates an identity matrix. An identity matrix is 
a matrix where value = 1 if the row and column index are the same, and 
0 otherwise. It should build any size identity matrix you want.

Answer:
Only run when matrix_size is greater than zero
Generate a mateix filled with zero
A nested loop loop thu the matrix and set the value to 1 when col == row
"""

def iMatrix(matrix_size):
    if matrix_size > 0:
        result = [[0]*matrix_size for x in range(matrix_size)]
        for row in range(matrix_size):
            for col in range(matrix_size):
                    result[row][col] = 1 if row == col else 0
        return result

print np.matrix(iMatrix(5))


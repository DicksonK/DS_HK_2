# ________   .__          __                            
# \______ \  |__|  ____  |  | __  ______  ____    ____  
#  |    |  \ |  |_/ ___\ |  |/ / /  ___/ /  _ \  /    \ 
#  |    `   \|  |\  \___ |    <  \___ \ (  <_> )|   |  \
# /_______  /|__| \___  >|__|_ \/____  > \____/ |___|  /
#         \/          \/      \/     \/              \/ 
# GA DataScience HW-2
# Date: 2014-04-30

import sys
import os
import timeit
import numpy as np
from sets import Set
from datetime import datetime

# File header
# Age
# Gender
# Impressions
# Clicks
# Signed_In

#"age", "gender", "signed_in", "avg_click", "avg_impressions", "max_click", "max_impressions"

# Start a counter and store the textfile in memory
counter = 0
imp_count = 0
total_age = 0.0
temp_int = 0
str_result = ""
agg_set = Set([])
agg_list = list([])
age_set = Set([])

temp_age = 0
temp_gender = 0
temp_impressions = 0
temp_clicks = 0
temp_signed_in = 0

targetFile = open('nytimes.csv', 'r')
lines = targetFile.readlines()
targetFile.close()

lines.pop(0)
fileName = "result.txt"

print timeit.default_timer()
start = timeit.default_timer()


# ====================================================
if (not os.path.exists(fileName)):
    file = open(fileName, "w")
    file.close()
#print os.path.exists(fileName)
file = open(fileName, "a")

file.write("|=====================================================|\n")
file.write("|@Run: " + str(datetime.now()) + "                     |\n")
file.write("|@Version: 1.0                                        |\n")
file.write("|@Author: Dickson Kwong                               |\n")
file.write("|=====================================================|\n\n")
file.write("|=*=*=*=*=*=*=*=*=*=*=*=Start=*=*=*=*=*=*=*=*=*=*=*=*=|\n")

# ====================================================
# Click through rate (avg clicks per impression)
# For each line, insert index 0,1,4 in a set.
# Get the total number of lines
for line in lines:
    agg_set.add(str(line.strip().split(',')[0]) + "," + str(line.strip().split(',')[1]) + "," + str(line.strip().split(',')[4]))

num_lines = len(lines)

# Cast the set to list so it can be access by index
agg_list = list(agg_set)



print len(agg_list)

#Create 2D array to store the sum(click) and the count(click)
matrix = np.zeros(len(agg_set)*9).reshape((len(agg_set), 9))

# Set the first three column to the matrix.
for row in range(len(agg_list)):
    #print repr(agg_list[row])
    matrix[row,0] = agg_list[row].strip().split(',')[0]
    matrix[row,1] = agg_list[row].strip().split(',')[1]
    matrix[row,2] = agg_list[row].strip().split(',')[2]
    #counter += 1

#print matrix

# Set the second column to the sum(click) and third to the count(click).
for line in lines:
    #counter = 0
    for n in range(len(agg_list)):
        temp_age = int(line.strip().split(',')[0])
        temp_gender = int(line.strip().split(',')[1])
        temp_impressions = int(line.strip().split(',')[2])
        temp_clicks = int(line.strip().split(',')[3])
        temp_signed_in = int(line.strip().split(',')[4])

        if temp_age == matrix[n,0] and temp_gender == matrix[n,1] and temp_impressions == matrix[n,2] :
            matrix[n,3] += temp_clicks
            matrix[n,4] += 1
            matrix[n,5] += temp_impressions
            matrix[n,6] += 1
            if temp_clicks > matrix[n,7]:
                matrix[n,7] = temp_clicks
            if temp_impressions > matrix[n,8]:
                matrix[n,8] = temp_impressions

matrix = sorted(matrix, key=lambda x:int(x[0]))
"""
print "Age: " + str(matrix[0][0])
print "Gender: " + str(matrix[0][1])
print "Signed In: " + str(matrix[0][2])
print "Sum of Click: " + str(matrix[0][3])
print "Click count: " + str(matrix[0][4])
print "Avg Click: " + str(matrix[0][3]/matrix[0][4])
print "Sum of Impression: " + str(matrix[0][5])
print "Impression count: " + str(matrix[0][6])
print "Avg Impression: " + str(matrix[0][5]/matrix[0][6])
print "Max age: " + str(matrix[0][7])
print "Max impression: " + str(matrix[0][8])
"""

for n in range(len(matrix)):
    str_result = str(int(matrix[n][0]))
    str_result += ","
    str_result += str(int(matrix[n][1]))
    str_result += ","
    str_result += str(int(matrix[n][2]))
    str_result += ","
    str_result += "{:.4f}".format((matrix[n][3])/(matrix[n][4]))
    str_result += ","
    str_result += "{:.4f}".format((matrix[n][5])/(matrix[n][6]))
    str_result += ","
    str_result += str(int(matrix[n][7]))
    str_result += ","
    str_result += str(int(matrix[n][8]))
    file.write(str_result + "\n")
    str_result = ""

file.write("\n")

stop = timeit.default_timer()


file.write(stop - start + "\n")

file.write("|=*=*=*=*=*=*=*=*=*=*=*=*End*=*=*=*=*=*=*=*=*=*=*=*=*=|\n")
file.close
print "Done!"

#stop = timeit.default_timer()

print "Runtime: ", stop - start 


# ________   .__          __                            
# \______ \  |__|  ____  |  | __  ______  ____    ____  
#  |    |  \ |  |_/ ___\ |  |/ / /  ___/ /  _ \  /    \ 
#  |    `   \|  |\  \___ |    <  \___ \ (  <_> )|   |  \
# /_______  /|__| \___  >|__|_ \/____  > \____/ |___|  /
#         \/          \/      \/     \/              \/ 
# GA DataScience HW-1
# Date: 2014-04-28

#!/usr/bin/python
# Import required libraries
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

# Start a counter and store the textfile in memory
counter = 0
imp_count = 0
total_age = 0.0
temp_int = 0
str_result = ""
imp_set = Set([])
imp_list = list([])
age_set = Set([])
lines = sys.stdin.readlines()
lines.pop(0)
fileName = "result.txt"


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
# The total impression in the file
# For each line, find the sum of index 2 in the list.
for line in lines:
  imp_count = imp_count + int(line.strip().split(',')[2])

str_result = "Total impression: " + str(imp_count)

# Print out the total impression
#print
#print str_result
file.write(" 0.   The total impression in the file\n")
file.write("   " + str_result + "\n")
str_result = ""

# ====================================================
# The average age in the file
# For each line, find the sum of index 0 in the list.
for line in lines:
  total_age = total_age + int(line.strip().split(',')[0])
  
# Get total number of line
num_lines = sum(1 for line in lines)

str_result = "Average age: " + "{:.2f}".format(total_age/num_lines)

# Print out the total impression
#print str_result
file.write("\n i.   The average age in the file\n")
file.write("   " + str_result + "\n")
str_result = ""

# ====================================================
# Click through rate (avg clicks per impression)
# For each line, insert index 2 in a set.
for line in lines:
  imp_set.add(int(line.strip().split(',')[2]))

# Cast the set to list so it can be access by index
imp_list = list(imp_set)

#Create 2D array to store the sum(click) and the count(click)
matrix = np.zeros(len(imp_set)*3).reshape((len(imp_set), 3))

# Set the first column to the impression.
for n in imp_set:
  matrix[counter,0] = imp_list[counter]
  counter += 1

# Set the second column to the sum(click) and third to the count(click).
for line in lines:
  counter = 0
  for n in imp_list:
    if int(line.strip().split(',')[2]) == imp_list[counter]:
      matrix[counter,1] += int(line.strip().split(',')[3])
      matrix[counter,2] += 1
    counter += 1

#Print out the result
file.write("\n ii.  Click through rate (avg clicks per impression)\n")

counter = 0
for n in imp_list:
  str_result = "Impression " + str("%2d" % imp_list[counter])
  str_result += " has average clicks "
  str_result += "{:.4f}".format(matrix[counter,1]/matrix[counter,2])
  #print  str_result
  file.write("   " + str_result + "\n")
  str_result = ""
  counter += 1

# ====================================================
# The oldest person in the file
# For each line, insert index 0 in a set.
for line in lines:
  age_set.add(int(line.strip().split(',')[0]))
str_result = "The oldest person in the file is: " + str(max(age_set))
#print str_result
file.write("\n iii. The oldest person in the file\n")
file.write("   " + str_result + "\n")
str_result = ""

file.write("|=*=*=*=*=*=*=*=*=*=*=*=*End*=*=*=*=*=*=*=*=*=*=*=*=|\n\n\n")
file.close
print "Done!"


stop = timeit.default_timer()

print "Runtime: ", stop - start 
#test111788
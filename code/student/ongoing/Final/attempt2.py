# ________   .__          __                            
# \______ \  |__|  ____  |  | __  ______  ____    ____  
#  |    |  \ |  |_/ ___\ |  |/ / /  ___/ /  _ \  /    \ 
#  |    `   \|  |\  \___ |    <  \___ \ (  <_> )|   |  \
# /_______  /|__| \___  >|__|_ \/____  > \____/ |___|  /
#         \/          \/      \/     \/              \/ 
# Kaggle - Allstate
# Date: 2014-05-12

import numpy as np
import pandas as pd
import csv as csv
import itertools
import time
import math as m
from sklearn import linear_model
from sklearn.ensemble import RandomForestClassifier
from sklearn.cross_validation import cross_val_score

#=====================================
# HELPIER FUNCTION
#=====================================
def read_a_csv(file_name, isDes=False):
	try:
		df = pd.read_csv(file_name)
		if isDes:
			print "Imported a data set from [", file_name, "] :", len(df), "rows", len(df.columns), "columns"
		return df
	except IOError:
		print "Oops!  Failed to load..."


def fillna_int_w_mode(df, col):
	if len(df[col][ df[col].isnull() ]) > 0:
		df[col][ df[col].isnull() ] = df[col].dropna().mode().values
		#print len(df[col][ df[col].isnull() ]) 
	return df

def dropna_string(df, col, isDes = False):
	if isDes:
		print "Dropped", len(df[col][ df[col].isnull() ]) , "row(s) for [", col ,"]."
	df = df[pd.notnull(df[col])]

	return df

def df_get_unique(df, col, isPrint = False):

	result_list = dropna_string(df,col)[col].unique()

	return result_list

def add_string_dummy(df, col, isPrint = False):
	counter = 0
	df = dropna_string(df, col, isPrint)
	for t in df_get_unique(df, col):
		if (isinstance(t, basestring)):
			df[col + "_" + t] = train_set[col].str.contains(t) * 1
			counter = counter + 1
	if isPrint:
		print "Added ", counter, "column(s) for [", col ,"]."
	return df

def drop_some_col(df, col_list, isPrint = False):
	df = df.drop(col_list, axis=1) 
	if isPrint:
		print "Dropped ", len(col_list), "column(s)."
	return df

def print_section_line(count):
	result = "#"
	if count < 20:
		count = 20
	for n in range(count+1):
		result = result + "="
	return result + "#"

def print_section_space(count):
	result = ""
	for pos in range(count):
			result = result + " "
	return result

def print_section_break(header):
	size = 40

	result = print_section_line(size) + "\n"
	result = result + "#"
	result = result + print_section_space((size-len(header))/2)
	if len(header)%2 == 1:
		result = result + " "
	result = result + header
	result = result + print_section_space((size-len(header))/2)
	result = result +  " #\n" 
	result = result + print_section_line(size) + "\n"
	return result



if False:

	export_set = read_a_csv('myfirstforest_predict.csv')

	run_list = ['A','B','C','D','E','F','G']

	export_set["predict"] = ""

	for n in range(len(export_set['A'])):
		temp = ""

		for i in run_list:
			temp = temp + str(export_set[i][n])

		export_set["predict"][n] = temp

	#print len(export_set['A'])

#	print export_set['A'][0]

#	for i in run_list:

		#export_set["predict"].concat(export_set["predict"], )
#		export_set["predict"] = export_set["predict"].map(str) + str(export_set[i])

#	export_set["predict"] = pd.concat(export_set['A'].to_frame(),export_set['B'].to_frame())

	print export_set["predict"]

	exp_cols = [col for col in export_set.columns if col not in [u'customer_ID', u'predict']]

	drop_some_col(export_set, exp_cols).to_csv("myfirstforest_predict_clean.csv", header=True, index=False)

if True:


	#=====================================
	# Pre-work
	#=====================================

	print print_section_break("Pre-work")

	t0 = time.clock()

	train_set = read_a_csv('train.csv', True)
	test_set = read_a_csv('test_v2.csv', True)
	test_set_result = test_set

	#Car_age has no null
	train_set = fillna_int_w_mode(train_set, "car_age")
	test_set = fillna_int_w_mode(test_set, "car_age")

	train_set = dropna_string(train_set, "car_value", True)
	test_set = dropna_string(test_set, "car_value", True)

	train_set = add_string_dummy(train_set, "car_value", True)
	test_set = add_string_dummy(test_set, "car_value", True)

	train_set = add_string_dummy(train_set, "state", True)
	test_set = add_string_dummy(test_set, "state", True)

	drop_col_list = ['customer_ID', 'shopping_pt', 'record_type', 'day', 'time', 'car_age', 'age_youngest', 'age_oldest', 'C_previous', 'duration_previous', 'cost',  'state', 'location',  'homeowner', 'car_value', 'risk_factor']
	drop_col_list_2 = ['shopping_pt', 'record_type', 'day', 'time', 'car_age', 'age_youngest', 'age_oldest', 'C_previous', 'duration_previous', 'cost',  'state', 'location',  'homeowner', 'car_value', 'risk_factor']

	#drop_col_list_3 = ['shopping_pt', 'record_type', 'day', 'time', 'car_age', 'age_youngest', 'age_oldest', 'C_previous', 'duration_previous', 'cost',  'state', 'location',  'homeowner', 'car_value', 'risk_factor']

	train_set = drop_some_col(train_set, drop_col_list_2, True) 
	test_set_result = drop_some_col(test_set, drop_col_list_2, True) 

	test_set = drop_some_col(test_set, drop_col_list_2, True) 


	for c in train_set.columns:
		train_set = fillna_int_w_mode(train_set, c)
		test_set = fillna_int_w_mode(test_set, c)
		test_set_result = fillna_int_w_mode(test_set_result, c)



	print time.clock(), "s"

#=====================================
# Training Model
#=====================================

#print train_set.columns

"""
Index([u'group_size', u'married_couple', u'A', u'B', u'C', u'D', u'E', u'F', u'G', u'car_value_g', u'car_value_e', 
	u'car_value_c', u'car_value_d', u'car_value_f', u'car_value_h', u'car_value_i', u'car_value_b', u'car_value_a', 
	u'state_IN', u'state_NY', u'state_PA', u'state_WV', u'state_MO', u'state_OH', u'state_OK', u'state_FL', u'state_OR', 
	u'state_WA', u'state_KS', u'state_NV', u'state_ID', u'state_CO', u'state_CT', u'state_AL', u'state_AR', u'state_NM', 
	u'state_MS', u'state_MD', u'state_RI', u'state_UT', u'state_ME', u'state_TN', u'state_WI', u'state_MT', u'state_KY', 
	u'state_WY', u'state_NE', u'state_ND', u'state_DE', u'state_GA', u'state_NH', u'state_IA', u'state_DC', u'state_SD'], 
	dtype='object')
"""


 
if True:
	print print_section_break("Training Model")

	forest = RandomForestClassifier(n_estimators=20)

	run_list = ['A','B','C','D','E','F','G']

	test_set_result = drop_some_col(test_set_result, run_list)

	#test_set.to_csv("checkTest.csv", header=True, index=False)

	#predictions_file = open("myfirstforest.csv", "wb")
	#open_file_object = csv.writer(predictions_file)

	#print len(train_set)
	#663718

	#print len(test_set)
	#198117

	#org is 198856

	import pandas as pd
	from sklearn import ensemble



#  loc_train = "kaggle_forest\\train.csv"
#  loc_test = "kaggle_forest\\test.csv"
#  loc_submission = "kaggle_forest\\kaggle.forest.submission.csv"
 
	df_train = train_set
	df_test = test_set
 
	feature_cols = [col for col in df_train.columns if col in [u'group_size', u'married_couple', u'A', u'B', u'C', u'D', u'E', u'F', u'G', u'car_value_g', u'car_value_e', 
	u'car_value_c', u'car_value_d', u'car_value_f', u'car_value_h', u'car_value_i', u'car_value_b', u'car_value_a', 
	u'state_IN', u'state_NY', u'state_PA', u'state_WV', u'state_MO', u'state_OH', u'state_OK', u'state_FL', u'state_OR', 
	u'state_WA', u'state_KS', u'state_NV', u'state_ID', u'state_CO', u'state_CT', u'state_AL', u'state_AR', u'state_NM', 
	u'state_MS', u'state_MD', u'st sysate_RI', u'state_UT', u'state_ME', u'state_TN', u'state_WI', u'state_MT', u'state_KY', 
	u'state_WY', u'state_NE', u'state_ND', u'state_DE', u'state_GA', u'state_NH', u'state_IA', u'state_DC', u'state_SD']]
 
	for n in run_list:

		X_train = df_train[feature_cols]
		X_test = df_test[feature_cols]
		y_train = df_train[n]
		y_test = df_test[n]

		test_ids = df_test['customer_ID']

		clf = ensemble.RandomForestClassifier(n_estimators = 200, n_jobs = -1)

		clf.fit(X_train, y_train)
		test_set_result[n] = clf.predict(X_test)
		score = cross_val_score(clf, X_test, y_test)
		print score.mean()
"""
	test_set_result.to_csv("myfirstforest_predict2.csv", header=True, index=False)

    #Simple K-Fold cross validation. 5 folds.
    cv = cross_validation.KFold(len(df_train), k=5, indices=False)

    #iterate through the training and test cross validation segments and
    #run the classifier on each one, aggregating the results into a list
    results = []
    for traincv, testcv in cv:
        probas = cfr.fit(df_train[traincv], target[traincv]).predict_proba(train[testcv])
        results.append( logloss.llfun(target[testcv], [x[1] for x in probas]) )

    #print out the mean of the cross-validated results
    print "Results: " + str( np.array(results).mean() )

"""

		#print clf.predict(X_test)#.to_csv("myfirstforest_predict.csv", header=True, index=False)
		#print test_set_result["A"]
"""
	with open("myfirstforest_other.csv", "wb") as outfile:
		outfile.write("customer_ID,A\n")
    	for e, val in enumerate(list()):
      		outfile.write("%s,%s\n"%(test_ids[e],val))
"""


if False:

	for n in run_list:
		forest = forest.fit( drop_some_col(train_set, run_list), train_set[n] )
		output = forest.predict(drop_some_col(test_set, run_list), test_set[n])
		test_set_result[n] = output
		#print output
		#open_file_object.writerow(output)
		scores = cross_val_score(forest, drop_some_col(test_set, run_list), test_set[n])
		print scores.mean(), "\t\t\t" ,time.clock(), "s"

	#predictions_file.close()


	


#!/usr/bin/python

import sys
import pickle
import numpy as np
sys.path.append("../tools/")

from feature_format import featureFormat, targetFeatureSplit
from tester import dump_classifier_and_data

### Task 1: Select what features you'll use.
### features_list is a list of strings, each of which is a feature name.
### The first feature must be "poi".
#features_list = ['poi','salary'] # You will need to use more features

finance_features = ["salary", "deferral_payments", "total_payments", 
"loan_advances", "bonus", "restricted_stock", "restricted_stock_deferred", 
"deferred_income", "total_stock_value", "expenses", "exercised_stock_options", 
"long_term_incentive", "director_fees", "other"]

email_features = ["from_this_person_to_poi", "from_poi_to_this_person", "from_messages", 
"to_messages", "shared_receipt_with_poi"]

poilabel = ["poi"]

features_list = poilabel + finance_features + email_features 

### Load the dictionary containing the dataset
with open("final_project_dataset.pkl", "r") as data_file:
    data_dict = pickle.load(data_file)

#summary of the data
#number of data points
print "total number of data points: ", len(data_dict)

#number of people broken up as poi or non poi

poi = 0
notpoi = 0

for i in data_dict:
	if data_dict[i]["poi"] == True:
		poi += 1
	elif data_dict[i]["poi"] == False:
		notpoi +=1

print "total number of people in dataset: ", poi + notpoi
print "total number of poi: ", poi 
print "total number of not poi: ", notpoi

total_features = data_dict[data_dict.keys()[0]].keys()
print("There are %i features for each person in the dataset, and %i features are used" 
	%(len(total_features), len(features_list)))

#how many of the features have "NaN"
nan_values = {}

for features in total_features:
	nan_values[features] = 0

for person in data_dict:
	for feature in data_dict[person]:
		if data_dict[person][feature] == "NaN":
			nan_values[feature] += 1


print "the number of NaN values for features in our dataset: "

for f in nan_values:
	print f, " ", nan_values[f]



### Task 2: Remove outliers
#code to show that 'TOTAL' was an outlier
#for employee in data_dict:
#	if(data_dict[employee]["salary"] != "NaN") and (data_dict[employee]["bonus"] != "NaN"):
#		if float(data_dict[employee]["salary"]) > 1000000 and float(data_dict[employee]["bonus"]) > 5000000:
#			print employee



data_dict.pop('TOTAL', 0)
data_dict.pop('THE TRAVEL AGENCY IN THE PARK', 0) 
data_dict.pop('LOCKHART EUGENE E', 0)


#remove all NaN values before proceeding
for key in data_dict:
	for k,v in data_dict[key].iteritems():
		if(v=="NaN"):
			data_dict[key][k].replace(v,"")


### Task 3: Create new feature(s)
#new feature here is similar to what we did in the quiz. I am creating a ratio of to messages and from messages
def checknan(nanval):
	if str(nanval).lower == 'nan':
		return 0
	else:
		return nanval

def compute_fraction(to_messages, from_messages):

	fraction = 0
	if to_messages == "NaN" or to_messages == 0 or from_messages == "NaN" or from_messages == 0:
		return fraction
	else:
		fraction = checknan(float(to_messages)/float(from_messages))

	return fraction	

### Store to my_dataset for easy export below.
my_dataset = data_dict

#we then take the function we wrote above and extract the necessary information compute the fraction and that back into the dataset
for person in my_dataset:

	from_poi_to_this_person = my_dataset[person]["from_poi_to_this_person"]
	to_messages = my_dataset[person]["to_messages"]
	if from_poi_to_this_person != "NaN" and to_messages != "NaN":
		ratio_frompoi = compute_fraction(from_poi_to_this_person, to_messages)
		my_dataset[person]["ratio_frompoi"] = ratio_frompoi
	else:	
		my_dataset[person]["ratio_frompoi"] = 0

	from_this_person_to_poi = my_dataset[person]["from_this_person_to_poi"]
	from_messages = my_dataset[person]["from_messages"]
	if from_this_person_to_poi != "NaN" and from_messages != "NaN":
		ratio_topoi = compute_fraction(from_this_person_to_poi, from_messages)
		my_dataset[person]["ratio_topoi"] = ratio_topoi
	else:
		my_dataset[person]["ratio_topoi"] = 0

newfeaturelist = features_list + ["ratio_frompoi", "ratio_topoi"]



### Extract features and labels from dataset for local testing

#print('all %d features :: %r' % (len(my_dataset), my_dataset))
#print('all %d features :: %r' % (len(newfeaturelist), newfeaturelist))

data = featureFormat(my_dataset, newfeaturelist, sort_keys = True)
labels, features = targetFeatureSplit(data)


from sklearn.feature_selection import SelectKBest

selection = SelectKBest(k=10)
selection.fit(features, labels)
scores = selection.scores_


#zip features and scores into a table
featurescore = zip(newfeaturelist[1:], scores)

#order them and then print them out
ordered_featurescore = sorted(featurescore, key = lambda x: x[1], reverse = True)
for feature, score in ordered_featurescore:
	print feature, score


from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
features = scaler.fit_transform(features)

### Task 4: Try a varity of classifiers
### Please name your classifier clf for easy export below.
### Note that if you want to do PCA or other multi-stage operations,
### you'll need to use Pipelines. For more info:
### http://scikit-learn.org/stable/modules/pipeline.html 

# Provided to give you a starting point. Try a variety of classifiers.
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score
from sklearn.model_selection import train_test_split, StratifiedShuffleSplit
from sklearn.grid_search import GridSearchCV
from sklearn import linear_model 

features_train, features_test, labels_train, labels_test = train_test_split(
	features, labels, test_size=0.3, random_state=42)


#clf = GaussianNB()
#clf.fit(features_train, labels_train)
#pred = clf.predict(features_test)

#nb_accuracy = accuracy_score(pred, labels_test)
#precision = precision_score(pred, labels_test)
#recall = recall_score(pred, labels_test)

# clf = DecisionTreeClassifier()
# clf.fit(features_train, labels_train)
# pred = clf.predict(features_test)

# tree_accuracy = accuracy_score(pred, labels_test)
# precision = precision_score(pred, labels_test)
# recall = recall_score(pred, labels_test)

# print "\n DTC accuracy: ", tree_accuracy

# print "\n DTC precision: ", precision

# print "\n DTC recall: ", recall

# clf = KNeighborsClassifier(n_neighbors = 3, p = 1, weights = "uniform", leaf_size = 1, metric = "minkowski")
# clf.fit(features_train,labels_train)


# params = {"n_neighbors": [1,2,3,4,5,6], 
# "weights": ["uniform","distance"], 
# "algorithm": ["auto", "ball_tree", "kd_tree", "brute"], "p":[1,2], "metric" : ["minkowski","euclidean"], 
# "leaf_size" : [1,5,10,15,20,25,30]}
 

# search = GridSearchCV(clf, params, cv = sss, scoring = "recall")
# search.fit(features_train,labels_train)


# best_params = search.best_estimator_.get_params()
# for param_name in params.keys():
#  	print("%s = %r, " % (param_name, best_params[param_name]))


# pred = clf.predict(features_test)

# KN_acc = accuracy_score(pred, labels_test)
# precision = precision_score(pred, labels_test)
# recall = recall_score(pred, labels_test)

# print "\n K neighbors accuracy: ", KN_acc

# print "\n KNN precision: ", precision

# print "\n KNN recall: ", recall


clf = 









### Task 5: Tune your classifier to achieve better than .3 precision and recall 
### using our testing script. Check the tester.py script in the final project
### folder for details on the evaluation method, especially the test_classifier
### function. Because of the small size of the dataset, the script uses
### stratified shuffle split cross validation. For more info: 
### http://scikit-learn.org/stable/modules/generated/sklearn.cross_validation.StratifiedShuffleSplit.html


### Task 6: Dump your classifier, dataset, and features_list so anyone can
### check your results. You do not need to change anything below, but make sure
### that the version of poi_id.py that you submit can be run on its own and
### generates the necessary .pkl files for validating your results.

dump_classifier_and_data(clf, my_dataset, features_list)
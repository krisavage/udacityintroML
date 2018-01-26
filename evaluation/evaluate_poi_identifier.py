#!/usr/bin/python


"""
    Starter code for the evaluation mini-project.
    Start by copying your trained/tested POI identifier from
    that which you built in the validation mini-project.

    This is the second step toward building your POI identifier!

    Start by loading/formatting the data...
"""

import pickle
import sys
import numpy as np
sys.path.append("../tools/")
from feature_format import featureFormat, targetFeatureSplit

data_dict = pickle.load(open("../final_project/final_project_dataset.pkl", "r") )

### add more features to features_list!
features_list = ["poi", "salary"]

data = featureFormat(data_dict, features_list)
labels, features = targetFeatureSplit(data)



### your code goes here 
from sklearn import cross_validation
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn import tree
features_train, features_test, labels_train, labels_test = cross_validation.train_test_split(features, labels, test_size = 0.3, random_state = 42)


count  = 0 
for i in labels_test:
	if i == 1.0:
		count += 1

print "count of poi: ", count
print "num people in test set: ", len(labels_test)

#accuracy of a biased identifier

pred = [0.] * 29
accuracy = accuracy_score(pred, labels_test)

print accuracy

#number of true positives
clf = tree.DecisionTreeClassifier()
clf = clf.fit(features_train,labels_train)
pred = clf.predict(features_test)

cpp = [1 for j in zip(labels_test, pred) if j[0] == j[1] and j[1] == 1]
numcpp = np.sum(cpp)
print "number of correct positive predictions: ", numcpp


precision = precision_score(pred, labels_test)
recall = recall_score(pred, labels_test)

print "precision score: ", precision
print "recall score: ", recall

#precision and recall of test set


predictions = [0, 1, 1, 0, 0, 0, 1, 0, 1, 0, 0, 1, 0, 0, 1, 1, 0, 1, 0, 1] 
true_labels = [0, 0, 0, 0, 0, 0, 1, 0, 1, 1, 0, 1, 0, 1, 1, 1, 0, 1, 0, 0]


precision2 = precision_score(true_labels,predictions)
recall2 = recall_score(true_labels, true_labels)

print "p2: ", precision2
print "r2: ", recall2







#!/usr/bin/python

""" 
    Starter code for exploring the Enron dataset (emails + finances);
    loads up the dataset (pickled dict of dicts).

    The dataset has the form:
    enron_data["LASTNAME FIRSTNAME MIDDLEINITIAL"] = { features_dict }

    {features_dict} is a dictionary of features associated with that person.
    You should explore features_dict as part of the mini-project,
    but here's an example to get you started:

    enron_data["SKILLING JEFFREY K"]["bonus"] = 5600000
    
"""

import pickle

enron_data = pickle.load(open("../final_project/final_project_dataset.pkl", "r"))

# no of employees in the data set
count = 0
salarycount = 0
emailcount = 0
for employee in enron_data:
	count+=1

print count

# no of features in the dataset
count1 = 0
for employee in enron_data:
	for features in enron_data[employee].iteritems():
		print features
		count1+=1
	break
		
print count1


# no of POIs in the data
count2 = 0
for employee in enron_data:
	if enron_data[employee]["poi"] == 1:
		count2+=1

print count2

#total value of the stock belonging to james prentice
print enron_data["PRENTICE JAMES"]["total_stock_value"]


# no of email messages from wesley colwell to persons of interest
print enron_data["COLWELL WESLEY"]["from_this_person_to_poi"]


# value of the stock options owned by jefferey k skilling
print enron_data["SKILLING JEFFREY K"]["exercised_stock_options"]

print enron_data["LAY KENNETH L"]["total_payments"]




for k, v in enron_data.iteritems():
	for m, n in v.iteritems():
		if m == 'salary' and n != 'NaN':
			salarycount +=1
		if m == 'email_address' and n != 'NaN':
			emailcount +=1



print "salary count", salarycount, "email count ", emailcount




#print len (enron_data)
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
poi_num = 0
persons = ["SKILLING JEFFREY K","LAY KENNETH L","FASTOW ANDREW S"]
total=0
whichone =""
quantified_salary_persons=0
quantified_emails=0
NaN_total_payments=0
poi_NaN_payments = 0
for p in enron_data:
   if enron_data[p]['poi'] == 1:
	   poi_num +=1
	   if enron_data[p]["total_payments"] == "NaN":
	      poi_NaN_payments += 1
   if enron_data[p]['salary'] != "NaN":
	    quantified_salary_persons +=1
   if enron_data[p]['email_address'] != "NaN":
	    quantified_emails += 1
   if enron_data[p]["total_payments"] == "NaN":
      NaN_total_payments += 1
	  
	    
for person in persons:
    if total<enron_data[person]["total_payments"]:
	   total=enron_data[person]["total_payments"]
	   whichone = person
	   

NaN_percentage= (NaN_total_payments/float(len(enron_data)))

poi_NaN_percentage = (poi_NaN_payments/float(poi_num))


print "How many data points(people) are in the dataset?\n+ %r" %len(enron_data)
print "For each person how many features are available? \n+ %r" %len(enron_data['SKILLING JEFFREY K'])
print "How many POIs are there in E+F dataset? \n+ %r" % poi_num
print "How many POIs were there in total? \n+ %r" % 35   ## look at the no of POIs in the poi_names.txt file in the final_project folder
print "What is the total value of stocks belonging to James Prentice? \n+ %r" % enron_data['PRENTICE JAMES']['total_stock_value']
print "How many email messages do we have from Wesley Colwell to persons of interest? \n+ %r" % enron_data['COLWELL WESLEY']['from_this_person_to_poi']
print "Whats the value of stock options exercised by Jeffrey Skilling? \n+ %r" % enron_data['SKILLING JEFFREY K']['exercised_stock_options']
print "of these three individuals(Lay,Skilling and Fastow) who took home the most money(largest value of total payments feature)? \n+ %r %r" %(total,whichone)
print "How many folks in the dataset have quantified salary?How about their email addresses? \n+ %r %r" %(quantified_salary_persons,quantified_emails)
print "What percentage of people in the dataset have NaN as their total_payments and how many are they? \n+ %r %r" %(NaN_percentage,NaN_total_payments)
print "What percentage of POIs have NaN as their total_payments and how many are they? \n+ %r %r" %(poi_NaN_percentage,poi_NaN_payments)
print "if a machine learning algorithm were to use total_payments as a feature,would you expect it to associate a NaN value with POIs or non-POIs?\n+ %r" % "Non-POIs"  
print "If you added 10 more data points which were all POIs and put NaN for the total payments for these folks.What would be the new number of people of the dataset? What would be the new number of folks with NaN for total payments? \n+ %r %r" % (156 , 31)
print "What is the new number of POIs in the dataset? What is the new number of POIs with NaN for total payments? \n+ %r %r" %(28,10)

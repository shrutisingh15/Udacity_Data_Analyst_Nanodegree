#!/usr/bin/python

import sys
import pickle
import pprint
sys.path.append("../tools/")

from feature_format import featureFormat, targetFeatureSplit
from tester import dump_classifier_and_data, test_classifier
from sklearn.grid_search import GridSearchCV
from sklearn.cross_validation import StratifiedShuffleSplit


### Task 1: Select what features you'll use.
### features_list is a list of strings, each of which is a feature name.
### The first feature must be "poi".
### features_list = ['poi','salary'] # You will need to use more features


### Load the dictionary containing the dataset
with open("final_project_dataset.pkl", "r") as data_file:
    data_dict = pickle.load(data_file)

### Task 2: Remove outliers
### Task 3: Create new feature(s)
### Store to my_dataset for easy export below.
## Get the names of all employees or keys of the data dictionary.
for data in data_dict:
    pprint.pprint(data)

## Lets take a glimpse of the data. How does the data look like ???
for data in data_dict:
    pprint.pprint(data_dict[data])

## Looks like an outlier is found as "TOTAL" which does not make any sense.Lets double check it.

pprint.pprint(data_dict['TOTAL'])

## Also there is another datapoint "THE TRAVEL AGENCY IN THE PARK" which consists of mainly misiing values or NaNs.Lets check that too.
pprint.pprint(data_dict['THE TRAVEL AGENCY IN THE PARK'])

## Calculate the total length of data dictionary before popping any data.
print "Length of data dictionary before removing any records :", len(data_dict)  ## 146 records	as expected

## Removing the records for "TOTAL" and "THE TRAVEL AGENCY IN THE PARK" datapoints.
data_dict.pop('TOTAL',0)  
data_dict.pop('THE TRAVEL AGENCY IN THE PARK',0)	

## Calculate the total length of data dictionary after popping the data in the above lines of code
print "Length of data dictionary after removing outliers :", len(data_dict)  ## 144 records 


## Function to calculate percentage of missing values in a feature
def NaN_Percentage(feature):
    count = 0
    for data in data_dict:
	    if (data_dict[data][feature] == "NaN"):
		    count += 1
    NaN_Percent = float(count)/float(len(data_dict))
    NaN_Percent = NaN_Percent * 100
    NaN_Percent = round(NaN_Percent,2)
    return NaN_Percent	
	
	
## Going through the percentage of missing values in the features I care about.
print "Lets care to check the percentage of missing values in the features..."	
print "Percentage of missing values in salary feature :", NaN_Percentage('salary')	
print "Percentage of missing values in bonus :", NaN_Percentage('bonus')
print "Percentage of missing values in exercised stock options :", NaN_Percentage('exercised_stock_options')
print "Percentage of missing values in long term incentives :", NaN_Percentage('long_term_incentive')
print "Percentage of missing values in emails from employees i.e from_messages :", NaN_Percentage('from_messages')
print "Percentage of missing values in emails to employees i.e to_messages :", NaN_Percentage('to_messages')
print "Percentage of missing values in emails from an employee to a poi employee :", NaN_Percentage('from_this_person_to_poi')
print "Percentage of missing values in emails to an employee from a poi employee :", NaN_Percentage('from_poi_to_this_person')
print "Percentage of missing values in total stock value :", NaN_Percentage('total_stock_value')

###print  'Features Selected: {}'.format(len(features_list)-1)


## Function to find records where total_stock_value, bonus, and exercised_stock_options are missing or nan.

def NaN_Records(data_dict):
    nan_records =[]
    for data in data_dict:
	    if ((data_dict[data]['total_stock_value'] == 'NaN') and (data_dict[data]['bonus'] == 'NaN') and (data_dict[data]['exercised_stock_options'] == 'NaN')):
		    nan_records.append(data)
    return nan_records	
	
nan_records= NaN_Records(data_dict)
print "printing nan records..."
for i in range(len(nan_records)):
    print nan_records[i]
    print data_dict[nan_records[i]]

## From the above results,it looks like the values for all the items are nan for LOCKHART EUGENE E person. so, it would be appropriate to remove this record from the data dictionary.So, lets confirm that first and then remove the datapoint.

print "Checking to confirm missing values in LOCKHART data point... "
print data_dict['LOCKHART EUGENE E'] ## And, yes all the values are missing.So, pop it.

print "Yes all the values are missing for LOCKHART EUGENE E. Lets pop it..."

data_dict.pop('LOCKHART EUGENE E',0)

## Length of data dictionary now will be ...
print "Length of data dictionary now will be :", len(data_dict)	

# Lets, write a function to calculate the percentage ratio of poi messages to/from  a person.
def poi_message_ratio(poi_messages,total_messages):
    if poi_messages == "NaN":
	   ratio = 0
    elif total_messages == "NaN":
	   ratio = 0
    else:
       ratio = float(poi_messages)/float(total_messages)	
	
    return ratio	
           
## Lets add the message ratio to the dataset and see if there is any connection to the person being a POI.	
for data in data_dict:
    messages_from_poi_ratio = poi_message_ratio(data_dict[data]['from_poi_to_this_person'], data_dict[data]['to_messages'])
    data_dict[data]['messages_from_poi_ratio'] = messages_from_poi_ratio
	
    messages_to_poi_ratio = poi_message_ratio(data_dict[data]['from_this_person_to_poi'],data_dict[data]['from_messages'])
    data_dict[data]['messages_to_poi_ratio'] = messages_to_poi_ratio
	

##for data in data_dict:
    ##if data_dict[data]['poi']:
       ##print data_dict[data]['messages_to_poi_ratio']
	   

## Looks like there is no connection ....
## Hence, storing the final data_dict into my_dataset
my_dataset = data_dict

## features_list = ['total_stock_value','exercised_stock_options','bonus','deferred_income','restricted_stock','total_payments','salary',
##                 'messages_from_poi_ratio','messages_to_poi_ratio']

## features_list = ['total_stock_value','exercised_stock_options','bonus','deferred_income','restricted_stock','total_payments','salary',
##                 'messages_from_poi_ratio','messages_to_poi_ratio','long_term_incentive','expenses']

## features_list = ['total_stock_value','exercised_stock_options','bonus','deferred_income','restricted_stock','total_payments','salary',
##                 'messages_from_poi_ratio','messages_to_poi_ratio','long_term_incentive','expenses','other','shared_receipt_with_poi']

## features_list = ['total_stock_value','exercised_stock_options','bonus','deferred_income','restricted_stock','total_payments','salary',
##                 'messages_from_poi_ratio','messages_to_poi_ratio','long_term_incentive','expenses','other','shared_receipt_with_poi',
##                 'loan_advances','director_fees','deferral_payments','restricted_stock_deferred','to_messages','from_messages']

## features_list = ['poi','bonus','total_stock_value','exercised_stock_options','deferred_income','restricted_stock','total_payments','salary']

## features_list = ['poi','total_stock_value','exercised_stock_options','bonus','total_payments','salary','restricted_stock']

## features_list = ['poi','total_stock_value','exercised_stock_options','bonus','total_payments','salary']

## features_list = ['poi','total_stock_value','exercised_stock_options','bonus','total_payments']

## features_list = ['poi','total_stock_value','exercised_stock_options','bonus','salary','restricted_stock']

## features_list = ['poi','total_stock_value','exercised_stock_options','bonus','salary']

features_list = ['poi','total_stock_value','exercised_stock_options','bonus']

## features_list = ['poi','bonus','total_stock_value','salary']

## features_list = ['poi','bonus','total_stock_value']

## features_list = ['poi','bonus']

## features_list =['poi','total_stock_value']

### Extract features and labels from dataset for local testing
data = featureFormat(my_dataset, features_list, sort_keys = True)
labels, features = targetFeatureSplit(data)

### Task 4: Try a varity of classifiers
### Please name your classifier clf for easy export below.
### Note that if you want to do PCA or other multi-stage operations,
### you'll need to use Pipelines. For more info:
### http://scikit-learn.org/stable/modules/pipeline.html

# Provided to give you a starting point. Try a variety of classifiers.

## Classifier 1: Fitting a naive bayes classifier

'''
from sklearn.naive_bayes import GaussianNB
clf = GaussianNB()
test_classifier(clf, my_dataset, features_list)
'''


#####################################################################################################


## Classifier 2: Fitting a decision tree classifier
'''
from sklearn.tree import DecisionTreeClassifier

clf = DecisionTreeClassifier()
test_classifier(clf, my_dataset, features_list)
##params = {'criterion':['gini','entropy'],'max_features':['sqrt','log2','auto']}
##clf = GridSearchCV(DecisionTreeClassifier(),params)
##print "Best parameter options are :",clf.best_estimator_
##print features_list
print "Importances of features are :", clf.feature_importances_
'''


## Classifier 3: Fitting a RandomForestClassifier
'''
from sklearn.ensemble import RandomForestClassifier
clf = RandomForestClassifier()
test_classifier(clf,my_dataset,features_list)
##params = {'criterion':['gini','entropy'], 'n_estimators':[5,10,15], 'max_features' :['auto','sqrt','log2']}
##clf = GridSearchCV(RandomForestClassifier(),params)
print "Importance of the features are :", clf.feature_importances_
'''



## Classifier 4: Fitting an ExtraTreesClassifier
## Final algorithm choosen..

print "Finally using ExtraTreesClassifier as my POI identifier..."
print "Features Selected are ...", features_list
from sklearn.ensemble import ExtraTreesClassifier
params = {'n_estimators':[1,5,10],'criterion':['gini','entropy'],'max_features':['sqrt','log2',None]}
clf = ExtraTreesClassifier()
##test_classifier(clf,my_dataset,features_list)
## print "Features used :", features_list


## Tried fitting a KNN classification model too..
'''
from sklearn.neighbors import KNeighborsClassifier
params = {'algorithm':['auto','ball_tree','kd_tree','brute'],
           'weights' :['uniform','distance'],
		   'n_neighbors':[1,5,10]}
clf = KNeighborsClassifier()
##clf = KNeighborsClassifier(algorithm='auto',leaf_size=30,metric='minkowski',weights='uniform',n_neighbors=5,p=2)
test_classifier(clf, my_dataset, features_list) 
###print clf.feature_importances_

'''




### Task 5: Tune your classifier to achieve better than .3 precision and recall 
### using our testing script. Check the tester.py script in the final project
### folder for details on the evaluation method, especially the test_classifier
### function. Because of the small size of the dataset, the script uses
### stratified shuffle split cross validation. For more info: 
### http://scikit-learn.org/stable/modules/generated/sklearn.cross_validation.StratifiedShuffleSplit.html

# Example starting point. Try investigating other evaluation techniques!
##from sklearn.cross_validation import train_test_split
##features_train, features_test, labels_train, labels_test = \
   ## train_test_split(features, labels, test_size=0.3, random_state=42)
	

cross_validation=StratifiedShuffleSplit(labels,1000,test_size=0.1,random_state=42)
gridsearch=GridSearchCV(clf,params,cv=cross_validation,scoring='recall')
gridsearch.fit(features,labels)

print "Best score is :", gridsearch.best_score_
print "Best parameters are :", gridsearch.best_params_
print "Best estimators are :", gridsearch.best_estimator_
clf = gridsearch.best_estimator_

test_classifier(clf,my_dataset,features_list)
print "Importances of features are :", clf.feature_importances_

### Task 6: Dump your classifier, dataset, and features_list so anyone can
### check your results. You do not need to change anything below, but make sure
### that the version of poi_id.py that you submit can be run on its own and
### generates the necessary .pkl files for validating your results.


dump_classifier_and_data(clf, my_dataset, features_list)

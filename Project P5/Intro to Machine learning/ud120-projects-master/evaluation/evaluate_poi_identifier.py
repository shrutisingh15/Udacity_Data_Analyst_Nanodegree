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
sys.path.append("../tools/")
from feature_format import featureFormat, targetFeatureSplit

data_dict = pickle.load(open("../final_project/final_project_dataset.pkl", "r") )

### add more features to features_list!
features_list = ["poi", "salary"]

data = featureFormat(data_dict, features_list)
labels, features = targetFeatureSplit(data)

### your code goes here 

from sklearn import tree
from sklearn.metrics import accuracy_score
from sklearn import cross_validation
features_train, features_test, labels_train, labels_test = cross_validation.train_test_split(features,labels,random_state=42,test_size=0.3)
### Testing and training after test train split.
clf = tree.DecisionTreeClassifier()
clf.fit(features_train,labels_train)
pred = clf.predict(features_test)
accuracy = accuracy_score(pred,labels_test)
print "Accuracy after test train split:", accuracy

'''
How many POIs are predicted for the test set for your POI identifier?

(Note that we said test set! We are not looking for the number of POIs in the whole dataset.)
'''
print "POIs predicted for the test set : ", sum(pred)


'''
How many people total are in your test set?
'''

print "Total no. of people in test set:", len(pred)

'''
If your identifier predicted 0. (not POI) for everyone in the test set, what would its accuracy be?
'''
print "accuracy in case of non poi :", pred.tolist().count(0) / float(len(pred))

'''
Look at the predictions of your overfit model and compare them to the true test labels. Do you get any true positives? (In this case, we define a true positive as a case where both the actual label and the predicted label are 1)
'''

true_positives = 0
for p in range(len(pred)):
    if (pred[p] == labels_test[p]) and (labels_test == 1):
	    true_positives += 1
		
print "No. of true positives:", true_positives

'''
As you may now see, having imbalanced classes like we have in the Enron dataset (many more non-POIs than POIs) introduces some special challenges, namely that you can just guess the more common class label for every point, not a very insightful strategy, and still get pretty good accuracy!

Precision and recall can help illuminate your performance better. Use the precision_score and recall_score available in sklearn.metrics to compute those quantities.

Whats the precision?
'''

from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
print "precision score is:", precision_score(pred,labels_test)

'''
Whats the recall? 

(Note: you may see a message like UserWarning: The precision and recall are equal to zero for some labels. Just like the message says, there can be problems in computing other metrics (like the F1 score) when precision and/or recall are zero, and it wants to warn you when that happens.) 

Obviously this isnt a very optimized machine learning strategy (we haven’t tried any algorithms besides the decision tree, or tuned any parameters, or done any feature selection), and now seeing the precision and recall should make that much more apparent than the accuracy did.

'''
print "Recall score is:", recall_score(pred,labels_test)
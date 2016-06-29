#!/usr/bin/python

import pickle
import numpy
numpy.random.seed(42)


### The words (features) and authors (labels), already largely processed.
### These files should have been created from the previous (Lesson 10)
### mini-project.
words_file = "../text_learning/your_word_data.pkl" 
authors_file = "../text_learning/your_email_authors.pkl"
word_data = pickle.load( open(words_file, "r"))
authors = pickle.load( open(authors_file, "r") )



### test_size is the percentage of events assigned to the test set (the
### remainder go into training)
### feature matrices changed to dense representations for compatibility with
### classifier functions in versions 0.15.2 and earlier
from sklearn import cross_validation
features_train, features_test, labels_train, labels_test = cross_validation.train_test_split(word_data, authors, test_size=0.1, random_state=42)

from sklearn.feature_extraction.text import TfidfVectorizer
vectorizer = TfidfVectorizer(sublinear_tf=True, max_df=0.5,
                             stop_words='english')
features_train = vectorizer.fit_transform(features_train)
features_test  = vectorizer.transform(features_test).toarray()


### a classic way to overfit is to use a small number
### of data points and a large number of features;
### train on only 150 events to put ourselves in this regime
features_train = features_train[:150].toarray()
labels_train   = labels_train[:150]



### your code goes here

'''
A classic way to overfit an algorithm is by using lots of features and not a lot of training data. You can find the starter code in feature_selection/find_signature.py. Get a decision tree up and training on the training data, and print out the accuracy. How many training points are there, according to the starter code?
'''
from sklearn import tree
from sklearn.metrics import accuracy_score

clf = tree.DecisionTreeClassifier()
clf.fit(features_train,labels_train)
pred=clf.predict(features_test)
accuracy = accuracy_score(pred,labels_test)
print "Accuracy of testing data set with 150 training points:", accuracy

'''
Take your (overfit) decision tree and use the featureimportances attribute to get a list of the relative importance of all the features being used. We suggest iterating through this list (its long, since this is text data) and only printing out the feature importance if its above some threshold (say, 0.2--remember, if all words were equally important, each one would give an importance of far less than 0.01). Whats the importance of the most important feature? What is the number of this feature?
'''

feature_importance = clf.feature_importances_

counter = 0
for imp in feature_importance:
    if (imp > 0.2):
	   print "importance :", imp , "index of the feature:", counter
	   break   
    counter += 1
	
'''
In order to figure out what words are causing the problem, you need to go back to the TfIdf and use the feature numbers that you obtained in the previous part of the mini-project to get the associated words. You can return a list of all the words in the TfIdf by calling get_feature_names() on it; pull out the word thats causing most of the discrimination of the decision tree. What is it? Does it make sense as a word thats uniquely tied to either Chris Germany or Sara Shackleton, a signature of sorts?
'''


feature_name = vectorizer.get_feature_names()[counter]
print "Feature name for feature number", counter , "is :", feature_name 		
			
'''
This word seems like an outlier in a certain sense, so lets remove it and refit. Go back to text_learning/vectorize_text.py, and remove this word from the emails using the same method you used to remove sara, chris, etc. Rerun vectorize_text.py, and once that finishes, rerun find_signature.py. Any other outliers pop up? What word is it? Seem like a signature-type word? (Define an outlier as a feature with importance >0.2, as before).
'''

## use the following line of code to solve this question in vectorize.py
## signature_words = ["sara", "shackleton", "chris", "germani","sshacklensf"]

'''
Update vectorize_test.py one more time, and rerun. Then run find_signature.py again. Any other important features (importance>0.2) arise? How many? Do any of them look like signature words, or are they more email content words, that look like they legitimately come from the text of the messages?
'''

## use the following line of code to solve this question in vectorize.py
## signature_words = ["sara", "shackleton", "chris", "germani","sshacklensf","cgermannsf"]
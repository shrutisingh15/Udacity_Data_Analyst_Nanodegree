#!/usr/bin/python

import os
import pickle
import re
import sys

sys.path.append( "../tools/" )
from parse_out_email_text import parseOutText
from sklearn.feature_extraction.text import TfidfVectorizer

"""
    Starter code to process the emails from Sara and Chris to extract
    the features and get the documents ready for classification.

    The list of all the emails from Sara are in the from_sara list
    likewise for emails from Chris (from_chris)

    The actual documents are in the Enron email dataset, which
    you downloaded/unpacked in Part 0 of the first mini-project. If you have
    not obtained the Enron email corpus, run startup.py in the tools folder.

    The data is stored in lists and packed away in pickle files at the end.
"""


from_sara  = open("from_sara.txt", "r")
from_chris = open("from_chris.txt", "r")

from_data = []
word_data = []

### temp_counter is a way to speed up the development--there are
### thousands of emails from Sara and Chris, so running over all of them
### can take a long time
### temp_counter helps you only look at the first 200 emails in the list so you
### can iterate your modifications quicker
temp_counter = 0

'''
In vectorize_text.py, you will iterate through all the emails from Chris and from Sara. For each email, feed the opened email to parseOutText() and return the stemmed text string. Then do two things:

1.remove signature words (sara, shackleton, chris, germani--bonus points if you can figure out why its germani and not germany)
2.append the updated text string to word_data -- if the email is from Sara, append 0 (zero) to from_data, or append a 1 if Chris wrote the email.
Once this step is complete, you should have two lists: one contains the stemmed text of each email, and the second should contain the labels that encode (via a 0 or 1) who the author of that email is.
'''
for name, from_person in [("sara", from_sara), ("chris", from_chris)]:
    for path in from_person:
        # only look at first 200 emails when developing
        # once everything is working, remove this line to run over full dataset
        # temp_counter += 1
        ##if temp_counter < 200: #remove comment if you need 200
        path = os.path.join('..', path[:-1])
        # print '[\033[91m OK\033[0m ]' + path
        email = open(path, "r")

        # use parseOutText to extract the text from the opened email
        text = parseOutText(email)
        # use str.replace() to remove any instances of the words
        # ["sara", "shackleton", "chris", "germani"]
        signature_words = ["sara", "shackleton", "chris", "germani","sshacklensf","cgermannsf"]
        for sword in signature_words:
            text = text.replace(sword, "") 
        # append the text to word_data
        word_data.append(text)
        # append a 0 to from_data if email is from Sara, and 1 if email is from Chris
        from_data.append(0) if (name == "sara") else from_data.append(1)
        email.close()

		
'''
In the box below, put the string that you get for word_data[152].
'''		
print "emails processed"
print "word_data[152]:", word_data[152]
from_sara.close()
from_chris.close()

pickle.dump( word_data, open("your_word_data.pkl", "w") )
pickle.dump( from_data, open("your_email_authors.pkl", "w") )



'''
Transform the word_data into a tf-idf matrix using the sklearn TfIdf transformation. Remove english stopwords.

You can access the mapping between words and feature numbers using get_feature_names(), which returns a list of all the words in the vocabulary. How many different words are there?
'''

### in Part 4, do TfIdf vectorization here
tfidfvectorizer = TfidfVectorizer(stop_words="english",lowercase=True)
tfidfvectorizer.fit_transform(word_data)
vocabulary=tfidfvectorizer.get_feature_names()
print "How many different words are in vocabulary?", len(vocabulary)

'''
What is word number 34597 in your TfIdf?

(Just to be clear--if the question were what is word number 100, we would be looking for the word corresponding to vocab_list[100]. Zero-indexed arrays are so confusing to talk about sometimes.)
'''
print "What is word number 34597 in your TfIdf?", vocabulary[34597]

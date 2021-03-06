#!/usr/bin/python 

""" 
    Skeleton code for k-means clustering mini-project.
"""




import pickle
import numpy
import matplotlib.pyplot as plt
import sys
sys.path.append("../tools/")
from feature_format import featureFormat, targetFeatureSplit




def Draw(pred, features, poi, mark_poi=False, name="image.png", f1_name="feature 1", f2_name="feature 2"):
    """ some plotting code designed to help you visualize your clusters """

    ### plot each cluster with a different color--add more colors for
    ### drawing more than five clusters
    colors = ["b", "c", "k", "m", "g"]
    for ii, pp in enumerate(pred):
        plt.scatter(features[ii][0], features[ii][1], color = colors[pred[ii]])

    ### if you like, place red stars over points that are POIs (just for funsies)
    if mark_poi:
        for ii, pp in enumerate(pred):
            if poi[ii]:
                plt.scatter(features[ii][0], features[ii][1], color="r", marker="*")
    plt.xlabel(f1_name)
    plt.ylabel(f2_name)
    plt.savefig(name)
    plt.show()



### load in the dict of dicts containing all the data on each person in the dataset
data_dict = pickle.load( open("../final_project/final_project_dataset.pkl", "r") )
### there's an outlier--remove it! 
data_dict.pop("TOTAL", 0)


### the input features we want to use 
### can be any key in the person-level dictionary (salary, director_fees, etc.) 
feature_1 = "salary"
feature_2 = "exercised_stock_options"
feature_3 = "total_payments"
poi  = "poi"
features_list = [poi, feature_1, feature_2]
data = featureFormat(data_dict, features_list )
poi, finance_features = targetFeatureSplit( data )


### in the "clustering with 3 features" part of the mini-project,
### you'll want to change this line to 
### for f1, f2, _ in finance_features:
### (as it's currently written, the line below assumes 2 features)
for f1, f2 in finance_features:
    plt.scatter( f1, f2 )
plt.show()

### cluster here; create predictions of the cluster labels
### for the data and store them to a list called pred

'''
Deploy k-means clustering on the financial_features data, with 2 clusters specified as a parameter.
Store your cluster predictions to a list called pred, so that the Draw() command at the bottom of the script works properly. 
In the scatterplot that pops up, are the clusters what you expected?
'''

from sklearn.cluster import KMeans

features_list = ["poi",feature_1,feature_2,feature_3]
ndata=featureFormat(data_dict,features_list)
poi,finance_features = targetFeatureSplit(ndata)
clf = KMeans(n_clusters=3)
pred=clf.fit_predict(finance_features)
Draw(pred,finance_features,poi,mark_poi=False,name="Cluster_3_features.png",f1_name=feature_1,f2_name=feature_2)
### rename the "name" parameter when you change the number of features
### so that the figure gets saved to a different file
try:
    Draw(pred, finance_features, poi, mark_poi=False, name="clusters.pdf", f1_name=feature_1, f2_name=feature_2)
except NameError:
    print "no predictions object named pred found, no clusters to plot"
	
	
'''
In the next lesson, we will talk about feature scaling. It is a type of feature preprocessing that you should perform before some classification and regression tasks. Here is a sneak preview that should call your attention to the general outline of what feature scaling does.

What are the maximum and minimum values taken by the exercised_stock_options feature used in this example?

(NB: if you look at finance_features, there are some NaN values that have been cleaned away and replaced with zeroes--so while those might look like the minima, it is  a bit deceptive because they are more like points for which we don't have information, and just have to put in a number. So for this question, go back to data_dict and look for the maximum and minimum numbers that show up there, ignoring all the NaN entries.)

'''

stock_options=[]
for d in data_dict:
    if data_dict[d]["exercised_stock_options"] == "NaN":
	   pass
    else:
       stock_options.append(data_dict[d]["exercised_stock_options"])
	   
min_stock = min(stock_options)
max_stock = max(stock_options)

print "min stock :", min_stock
print "max_stock :", max_stock
		

'''
What are the maximum and minimum values taken by salary?

(NB: same caveat as in the last quiz. If you look at finance_features, there are some NaN values that have been cleaned away and replaced with zeroes--so while those might look like the minima, it is a bit deceptive because they are more like points for which we don not have information, and just have to put in a number. So for this question, go back to data_dict and look for the maximum and minimum numbers that show up there, ignoring all the NaN entries.)
'''
	
salary=[]
for d in data_dict:
    if data_dict[d]["salary"] == "NaN":
       pass
    else:
       salary.append(data_dict[d]["salary"])

min_salary=min(salary)
max_salary=max(salary)	   

print "minimum salary :", min_salary
print "maximum salary :", max_salary


'''
Apply feature scaling to your k-means clustering code from the last lesson, on the salary and exercised_stock_options features (use only these two features). What would be the rescaled value of a salary feature that had an original value of $200,000, and an exercised_stock_options feature of $1 million? (Be sure to represent these numbers as floats, not integers!)
'''

rescaled_salary= float(200000 - min_salary)/(max_salary-min_salary)

rescaled_stock= float(1000000 - min_stock)/(max_stock-min_stock)

print "Rescaled salary for 200000 : ", rescaled_salary
print "Rescaled stock for 1 million : ", rescaled_stock

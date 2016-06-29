#!/usr/bin/python


def outlierCleaner(predictions, ages, net_worths):
    """
        Clean away the 10% of points that have the largest
        residual errors (difference between the prediction
        and the actual net worth).

        Return a list of tuples named cleaned_data where 
        each tuple is of the form (age, net_worth, error).
    """
    
    cleaned_data = []

    ### your code goes here
    errors = [(n-p)**2 for n, p in zip(net_worths,predictions)]
    full_data = zip(ages,net_worths,errors)
	## Sorting the data
    full_data=sorted(full_data,key=lambda k: k[2])
	## taking the first 90% of the data
    cleaned_data=full_data[:(len(full_data)*9)/10]
    return cleaned_data


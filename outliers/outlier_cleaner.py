#!/usr/bin/python


def outlierCleaner(predictions, ages, net_worths):
    """
        Clean away the 10% of points that have the largest
        residual errors (difference between the prediction
        and the actual net worth).

        Return a list of tuples named cleaned_data where 
        each tuple is of the form (age, net_worth, error).
    """
    for pred, age,networth in zip(predictions,ages,net_worths):
        cleaned_data.append(age[0],networth[0], pred[0] - networth[0])

    cleaned_data.sort(key=lambda tup: tup[2])
    ### your code goes here

    
    return cleaned_data[:81]


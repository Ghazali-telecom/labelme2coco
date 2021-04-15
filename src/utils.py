import numpy as np 

def random_number(n):
    """ Given a number n return one from the interval [0,n]
    :param: n int
    :return : random number from [0,n]"""
    
    return np.random.randint(n)

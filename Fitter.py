# -*- coding: utf-8 -*-
"""
Created on Tue Feb 03 16:22:51 2015

@author: Jacob
"""

def fitter(x,y, n = 1):
    """This function will perform a linear least squares fit on the given values

    Input:
    x = The independent value.
    y = The dependent value
    n = An optional parameter for the degree of the polynomial being fit

    Output:
    b = The coefficients of the fit, [constant, slope]
    fit = The dependent values estimated based on the fitted coefficients.
    fit_err = The error in the fitted values.
    """

    # Composing the array of x values in the form [1 x x^2...x^n] where each value in that 
    #   list implies a column (first a column of ones, then x, etc.)
    X = np.array([x**i for i in range(n+1)])
    

    # Calculating the coefficients of the polynomial with the linear least squares solution
    b = np.dot(np.linalg.inv(np.dot(X,X.T)), np.dot(y,X.T))


    # Calculating the fitted line and estimating the error in the fit
    fit = np.dot(b,X)
    res = (y-fit)**2
    fit_err =  np.sqrt(res.sum())/len(y) # error = L2 Norm


    return b, fit, fit_err
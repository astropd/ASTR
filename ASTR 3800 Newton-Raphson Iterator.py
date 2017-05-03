# -*- coding: utf-8 -*-
# <nbformat>3.0</nbformat>

# <codecell>

import numpy as np
def NR_iter(f, fprime, guess, tol=1e-6):
    """This function will perform the Newton-Raphson method
    
    f = function
    fprime = its derivative
    guess = initial solution
    tol = how closely the solution must converge
    """
    
    x_n = guess #setting the initial value
    x_n1 = x_n - f(x_n)/fprime(x_n)
    
    #iterate to solution
    while np.abs(f(x_n1)) >= tol:
        x_n = x_n1 
        x_n1 = x_n - f(x_n)/fprime(x_n)
    return x_n1
        

# <codecell>

#working through some examples
g = lambda x: 3*(np.exp(x)-1)-x*np.exp(x)
gprime = lambda x: 2*np.exp(x)-x*np.exp(x)
value = 3. 


NR_iter(g, gprime, value)

# <codecell>



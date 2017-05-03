# -*- coding: utf-8 -*-
"""
Created on Tue Feb 17 10:45:11 2015

@author: Jacob
"""

import numpy as np
import matplotlib.pyplot as pl

#1
#label constants for first part
sig = 1.0*(np.sqrt(2*np.pi))
sigg = 2.0*1.0**2.0
X = 2.0

#label constants for second part
sig1 = 0.3*(np.sqrt(2.0*np.pi))
sigg1 = 2.0*0.3**2.0
X1 = 3.0

x = np.linspace(-5, 5)
x1 = x
Gauss = sig**-1.0 * np.exp(-(x-X)**2.0 /sigg)
Gauss1= sig1**-1.0 * np.exp(-(x1-X1)**2.0 /sigg1)

pl.plot(x, Gauss, label='Sig=1, X=2')
pl.plot(x1, Gauss1, label='Sig=0.3, X=3')
pl.title('Plot of Two Gaussian')
pl.legend(loc=2)
pl.show()

#2
2*np.sqrt(2*np.log(2))
#so you can see that the result is 2.35*sig


#3
print 0.68*1000
print 0.05*1000
print 0.005*1000
print 0.68*0.5*1000

#4
sigma = 15.0*(np.sqrt(2*np.pi))
siggma = 2.0*15.0**2
X2 = 100
#%%
x2 = np.linspace(0,200)
Gauss3 = sigma**-1 * np.exp(-(x2-X2)**2/siggma)
pl.plot(x2,Gauss3)
pl.title('Gaussian of IQ')

def Gauss(x, mu = 100, sig = 15):
    '''GAUSSIAN'''
    co = 1/(sig*np.sqrt(2*np.pi))
    return co * exp(-(x-mu)**2/(2*sig**2))

#a neat trick Chloe showed me, finding where a given function is zero
from sympy.solvers import solve
from sympy import Symbol, exp, integrate

x = Symbol('x')
def solving(function):
    '''returns when the given function is zero if the function only has one indp. variable'''
    answer = solve(function,x)
    return answer
US = 318.9E6 #people

GUS = lambda x: US*Gauss(x) - 1 #-1 because then the function will cross zero at the smartest person and we can use solve!
print solving(GUS(x))

EARTH = 10E9 #people 
GEARTH = lambda x: EARTH*Gauss(x) - 1 #same thing...
print solving(GEARTH(x))


#Below creates a new cell in my editor (helps keep the plots from overlapping)
#%%

#continued blackbody, def(Star) copied from HW3, slightly modified

def HistStar(T, r, dist):
    '''
    Use Star to calculate the blackbody radiation when 
    input is (temp(K), radius(m), distance(pc))
    
    
    returns a plot of plancks function for given star
    pl.show() command required after calling Star
    '''
    
    h = 6.62606957e-34 #Js
    c = 299792458.0 #m/s
    k = 1.3806488e-23 # J/K
    lam = np.linspace(1e-8, 1e-6,1000) #wavelengths 0-1000
    d = dist*3.08567758e16 #convert pc to meters
    solangle = 4.0*np.pi
    distarea = 4.0*np.pi*d**2
    
    Balmost = 2.0*h*c**2/(lam**5)*(1.0/(np.exp(h*c/(lam*k*T))-1.0)) #J*s^-1*sr^-1*m^-2*^-1
    B1 = Balmost *solangle /distarea #J*s^-1*m^-1*m^-2
    peak = B1.max() #finds the peak 
    B = [x/peak for x in B1] #normalizes the function
    
    
    
    pl.hist(B) #plots histogram of normalized with bin size 10
    pl.title('Flux of Star')
    pl.ylabel('flux')
    pl.xlabel('wavelength')
    

HistStar(6000, 7E8, 10) #suns measurements at 10pc
pl.show()

#%%

def LStar(T, r, dist):
    '''
    Use Star to calculate the blackbody radiation when 
    input is (temp(K), radius(m), distance(pc))
    
    
    returns the sum of Planck's function, multiplied by pi to retrieve a rough luminosity
    '''
    
    h = 6.62606957e-34 #Js
    c = 299792458.0 #m/s
    k = 1.3806488e-23 # J/K
    lam = np.linspace(1e-8, 1e-6,10000) #wavelengths 0-1000
    d = dist*3.08567758e16 #convert pc to meters
    solangle = 4.0*np.pi
    distarea = 4.0*np.pi*d**2
    
    Balmost = 2.0*h*c**2/(lam**5)*(1.0/(np.exp(h*c/(lam*k*T))-1.0)) #J*s^-1*sr^-1*m^-2*^-1
    B1 = Balmost *solangle /distarea #J*s^-1*m^-1*m^-2
    return sum(B1)*np.pi
LStar(6000, 7E8, 5E-6) #T, radius and distance of Sun


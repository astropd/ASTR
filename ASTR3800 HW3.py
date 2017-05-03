# -*- coding: utf-8 -*-
"""
Created on Tue Feb 10 08:16:11 2015

@author: Jacob
"""

import numpy as np
import matplotlib.pyplot as pl
#4.2
g = [9.9,9.6,9.5,9.7,9.8]
a = np.mean(g) #takes the average of g
b = np.std(g) #finds the standard deviation of g
print a,b


#4.6
geig = [10,13,8,15,8,13,14,13,19,8,3,13,7,8,6,8,11,12,8,7]
aa = np.mean(geig)
bb = np.std(geig)
cc = np.sqrt(aa) #uses the "best value" (mean) 
print aa, bb, cc
#turns out that bb (standard deviation) and cc (square root)
#are close to eachother

#4.9
pend = [1.6,1.7,1.8]
ab = np.mean(pend)
ba = np.std(pend)
print ab, ba
#if the student were to take another measurement,
#it would have a 68% to be within the right value

#4.13
t = [8.16,8.14,8.12,8.16,8.18,8.10,8.18,8.18,8.18,8.24,8.16,8.14,8.17,8.18, 8.21,8.12,8.12,8.17,8.06,8.10,8.12,8.10,8.14,8.09,8.16,8.16,8.21,8.14,8.16,8.13]
l = np.mean(t)
ll = np.std(t)
print l, ll

#4.19
stdmean = aa/np.sqrt(len(geig)) #bb is the std of geig
print stdmean
add = np.sum(geig)
ans = add / len(geig)
ansstd = np.sqrt(ans)
print ans, aa, ansstd, bb
#so the two results are similar, using he SDOM gets a more
#precise answer

#4.28
length = np.array([51.2,59.7,68.2,79.7, 88.3])
peri = np.array([1.448, 1.566, 1.669,1.804, 1.896])
grav = 4*np.pi**2*length/peri**2
gcm = np.mean(grav)
gcmstd = np.std(grav)/np.sqrt(len(grav))
print gcm, gcmstd, 979.6/gcm
#the only way that the answer he got would fall into place with
#the true answer would be if the length had a 1.5% syst error to it

#correcting his mistake...
lcorrect = length-1
grav1 = 4*np.pi**2*lcorrect/peri**2
gcmnew = np.mean(grav1)
stdgrav = np.std(grav1)/np.sqrt(len(grav1))
print gcmnew, stdgrav

def Star(T, r, dist):
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
    
    
    
    pl.plot(lam,B)
    pl.title('Flux of Star')
    pl.ylabel('flux')
    pl.xlabel('wavelength')
    
    
Star(5778, 695800, 4.84813681e-6 )#Sun
Star(9602, 1960000000, 7.62) #Vega
Star(3400, 883*695800, 170) #Antares   
pl.legend('SVA')
pl.show()
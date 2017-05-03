# -*- coding: utf-8 -*-
"""
Created on Wed Apr 1 11:40:56 2015

@author: Jacob
"""

import numpy as np
import matplotlib.pyplot as pl
from mpl_toolkits.mplot3d import Axes3D

#two datasets which were contained within one catalogue are split into two lists because the data is inconsistant across the two sets
data = np.genfromtxt('C:\Users\Jacob\Documents\College\ASTR\ASTR 3800\PleaseReadMe.tsv', comments = '#', delimiter = ';')
data1 = np.genfromtxt('C:\Users\Jacob\Documents\College\ASTR\ASTR 3800\Pulsars1.txt', comments = '#', delimiter = ';')

#it is worth noting that this catologue no longer exists on Vzier for some odd reason...

#separate the data from the .tsv set
drad = data[:,0]  #Radial distance, in units of beam radii, most likley useless data
Fd = data[:,1]  #Flux density at 1400MHz, [mJy]
Frms = data[:,2] #rms uncertainty on Fd, [mJy]
bm = data[:,3]  #Pulse width [ms]
P = data[:,4] #period, [s]
DM = data[:,6] #dispersion measure,[pc/cm3]
DMrms = data[:,7]  #rms uncertainty on DM, [pc/cm3]
DEdt = data[:,8] #log of rotational energy, [10-7W]
D = data[:,9] #distance derived from DM and Taylor & Cordes, [mJy]
Lum = data[:,10] #radio luminosity at 1400Mhz, [kpc2]

#separate the data from the .txt set
drad1 = data1[:,0] #Radial distance, in units of beam radii, most likley useless data
Fd1 = data1[:,1]  #Flux density at 1400MHz, [mJy]
Frms1 = data1[:,2] #rms uncertainty on Fd1, [mJy]
bm1 = data1[:,3] #Pulse width [ms]
DM1 = data1[:,4] #dispersion measure,[pc/cm3]
DMrms = data1[:,5] #rms uncertainty on DM, [pc/cm3]
DMC = data1[:,6] #Catalogued dispersion measure,[pc/cm3]
FdC = data1[:,7] #Catalogued Flux density at 1400mHz [mJy]

#%%
#first let's see what the 3 relationships look like together, it's AWESOME.
#define 3D parameter space for some variables within my shitload of data
#do some linear fitting and then 3d "binning" to associate directions with how clumped together they are
fig = pl.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(Lum, P, DEdt, 'ro')
fig.savefig('3D model!') #sweet this looks like things I can do fits with!

#%%
#ensuring that fitting all 3 may be reasonable
pl.plot(P, DEdt, 'k.') 
pl.xlabel('Period [s] ')
pl.ylabel('Spin-Down Rate [10-7W]')
pl.title('Period vs. Energy Loss')

#%%
pl.plot(P, Lum, 'k.')
pl.xlabel('Period [s] ')
pl.ylabel('Luminosity at 1400MHz [kpc**2]')
pl.title('Period vs. Luminosity')

#%%
pl.plot(DEdt, Lum, 'k.')
pl.xlabel('Spin-Down Rate [10-7W] ')
pl.ylabel('Luminosity at 1400MHz [kpc**2]')
pl.title('Energy Loss vs. Luminosity')

#%%
#form: y = bx + a (which probably won't be a very accurate fit, but let's start there)
#start with the sum of the squares (found online via:http://mathworld.wolfram.com/LeastSquaresFitting.html )
sxx = np.sum(P**2) - len(P)*np.mean(P)**2
syy = np.sum(DEdt**2) - len(DEdt)*np.mean(DEdt)**2
sxy = np.sum(P*DEdt) - len(DEdt)*np.mean(P)*np.mean(DEdt)

#now we need covariance for our constants
sigx = sxx/len(P)
sigy = syy/len(DEdt)
cov = sxy/len(P)

b = sxy/sxx
a = np.mean(DEdt)-b*np.mean(P)
corr = sxy**2/(sxx*syy)
print corr

#same thing, but through Taylor's book (they both yield the same results)
A = (np.sum(P**2)*np.sum(DEdt)-np.sum(P)*np.sum(P*DEdt))/(len(P)*np.sum(P**2)-np.sum(P)**2)
B = (len(P)*np.sum(P*DEdt)-np.sum(P)*np.sum(DEdt))/(len(P)*np.sum(P**2)-np.sum(P)**2)

print a, b, A, B

lPDE = B*P + A

pl.plot(P, lPDE, 'r', label = 'Linear Best Fit')
pl.plot(P, DEdt, 'k.')
pl.xlabel('Period [s]')
pl.ylabel('Log of Rotational Energy Loss [10-7W]')
pl.title('Linear Fit of Period vs. Rotational Energy Loss')
pl.legend()
#stellar it kind of sucks. Let's do this again but with a power-law fit. 
#We'll also include Period vs. Luminosity and Luminosity vs. DEdt for this part. Should be ridiculous.

#%%
#Period vs Luminosity
A1 = (np.sum(P**2)*np.sum(Lum)-np.sum(P)*np.sum(P*Lum))/(len(P)*np.sum(P**2)-np.sum(P)**2)
B1 = (len(P)*np.sum(P*Lum)-np.sum(P)*np.sum(Lum))/(len(P)*np.sum(P**2)-np.sum(P)**2)

lPLUM = B1*P+A1
pl.plot(P, lPLUM, 'r', label = 'linear Best Fit')
pl.plot(P, Lum, 'k.')
pl.xlabel('Period [s]')
pl.ylabel('Luminosity at 1400 MHz [kpc**2]')
pl.title('Linear Fit of Period vs. Luminosity')
pl.legend()

#%%
#DEdt vs Lum
A2 = (np.sum(DEdt**2)*np.sum(Lum)-np.sum(DEdt)*np.sum(DEdt*Lum))/(len(P)*np.sum(DEdt**2)-np.sum(DEdt)**2)
B2 = (len(DEdt)*np.sum(DEdt*Lum)-np.sum(DEdt)*np.sum(Lum))/(len(DEdt)*np.sum(DEdt**2)-np.sum(DEdt)**2)

lDEdtLum = B2*P+A2
pl.plot(DEdt, lDEdtLum, 'r', label = 'linear Best Fit')
pl.plot(DEdt, Lum, 'k.')
pl.xlabel('Log of Rotational Energy Loss [10-7W]')
pl.ylabel('Luminosity at 1400 MHz [kpc**2]')
pl.title('Linear Fit of Rotational Energy Loss vs. Luminosity')
pl.legend()
#this one doesn't work...

#%%
#using form y = A*x**B (from:http://mathworld.wolfram.com/LeastSquaresFittingPowerLaw.html)
bexp = len(P) * np.sum(np.log(P)*DEdt) - np.sum(np.log(P))*np.sum(DEdt) / (len(P)*np.sum(np.log(P)**2)-np.sum(np.log(P))**2)
aexp = np.sum(DEdt)- bexp * np.sum(np.log(P)) / len(P)
Aexp = np.exp(aexp)
print aexp, bexp, Aexp
#%%
#so Aexp is zero, this method will no longer work! maybe try normalizing the quantities
Pn = P/P.max()
DEdtn = DEdt/DEdt.max()
bexpn = len(P) * np.sum(np.log(Pn)*DEdtn) - np.sum(np.log(Pn))*np.sum(DEdtn) / (len(P)*np.sum(np.log(Pn)**2)-np.sum(np.log(Pn))**2)
aexpn = np.sum(DEdtn)- bexpn * np.sum(np.log(Pn)) / len(P)
Aexpn = np.exp(aexpn)
print aexpn, bexpn, Aexpn 
#same problem. 

#%%
#Next up, let's figure out how to do a second degree fit
poly = np.polyfit(P, DEdt, 2) #finds coefficients for the appropriate power in ascending oder
fit = poly[0]*P**2 +poly[1]*P +poly[2] #the line ax**2+bx+c
pl.plot(P, fit, 'r', label='2nd Degree Fit')
pl.plot(P, DEdt, 'k.')
pl.xlabel('Period [s]')
pl.ylabel('Rotational Energy Loss [10-7W]')
pl.title('Quadratic Fit of Period vs. Rotational Energy Loss')
pl.legend()
#WHAT THE FUCK IS THIS...my "quadratic line" is a goddamned spider web.


#%%
#Okay so after some googling, there's a matrix method for higher powers which we did in tutorial
#fitter taken from tutorial notes
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
    b = np.dot(np.linalg.inv(np.dot(X,X.T)),np.dot(y,X.T))


    # Calculating the fitted line and estimating the error in the fit
    fit = np.dot(b,X)
    res = (y-fit)**2
    fit_err =  np.sqrt(res.sum())/len(y) # error = L2 Norm


    return b, fit, fit_err
#%%
fit2 = fitter(P, DEdt, 3)
L2 = fit2[0][0] +fit2[0][1]*P + fit2[0][2]*P**2 +fit2[0][3]*P**3

pl.plot(P, fit2[1], 'r', label='2nd Degree Fit')
pl.plot(P, DEdt, 'k.')
pl.xlabel('Period [s]')
pl.ylabel('Rotational Energy Loss [10-7W]')
pl.title('Quadratic Fit of Period vs. Rotational Energy Loss')
pl.legend()
#Damn, it's still a spiderweb. 
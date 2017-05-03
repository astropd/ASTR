# -*- coding: utf-8 -*-
"""
Created on Sun Apr 12 11:42:39 2015

@author: Jacob
"""

import numpy as np
import matplotlib.pyplot as pl
from scipy.stats import chi2

angst, count = np.genfromtxt("C:\Users\Jacob\Documents\Python Scripts\poissspec.txt", skip_header=1).T
wave = angst*1e-10 #list given as angstroms, so coversion

a = 0.0143878 #hc/k, m*K

BB = lambda w,T: 1/(w**5)/(np.exp(a/(w*T))-1) #BlackBody
FF = lambda w,T: 1/w**2/np.exp(a/(w*T)) #Free-free

norm = lambda x: x/x.max() #normalizing function
nBB = lambda w,T: norm(BB(w,T)) #normalized BB
nFF = lambda w,T: norm(FF(w,T)) #normalized FF
nC = norm(count) #normalized experimental data

chiX = lambda obs, exp: np.sum((exp-obs)**2/exp) #chi-squared
#%%

pl.plot(wave, nC) #plot the data
pl.plot(wave, nFF(wave, 1.5E7)) #after multiple guesses of a temperature 3e7 was achieved to "fit"

Tguess = np.logspace(6, 8, 100) #recommended log scale from Baylee
ChiBB = np.zeros(100)
ChiFF = np.zeros(100)

#fill ChiBB with the fit, note: +1 is to avoid divide by zeros
for i in np.arange(100):
    ChiBB[i] = chiX(nC+1,nBB(wave, Tguess[i])+1) 
#similarly with ChiFF
for i in np.arange(100):
    ChiFF[i] = chiX(nC+1, nFF(wave, Tguess[i])+1)


minBB = np.where(ChiBB == ChiBB.min())
minFF = np.where(ChiFF == ChiFF.min())

#%%
#Now we actually do the fit, for BB! 
pl.plot(wave, nC) #plot the data
pl.plot(wave, nBB(wave, Tguess[minBB[0]])) #after multiple guesses of a temperature 1.5e7 was achieved to "fit"

#%%
#Now we actually do the fit, for FF! 
pl.plot(wave, nC) #plot the data
pl.plot(wave, nFF(wave, Tguess[minFF[0]])) #after multiple guesses of a temperature 1.5e7 was achieved to "fit"

#%%
tBB = np.linspace(3.5E6,7E6,100) #generate values centered around the minimum temp
sBB = np.empty((100,100)) #empty array to be filled with Chai-Square values
IBB = np.linspace(50,350, 100) #Blank array to be filled in a for loop

tFF = np.linspace(8E6, 2E7, 100) #similarly for the FF
IFF = np.linspace(50,250,100)
sFF = np.empty((100,100))

#will eventually fill sBB/sFF with the chi values
def fitBB(I,T):
    z = []
    
    for i in T:
        K = chiX(I*nBB(wave,i), count+1)
        z.append(K)
    z = np.array(z)
    return z
    

def fitFF(I,T):
    
    h = []
    
    for i in T:
        J = chiX(I*nFF(wave,i), count+1)
        h.append(J)
    h = np.array(h)
    return h
#filling arrays!
for i, j in enumerate(IBB):
    sBB[i] = fitBB(j,tBB)
for i, j in enumerate(IFF):
    sFF[i] = fitFF(j,tFF)
    
#%%        

#making the general contour plot for FF      
pl.imshow(np.log10(sFF))
pl.contour(sFF/100., levels=[1,4,9])
pl.ylabel('Intensity')
pl.xlabel('Temp')
pl.colorbar()
pl.show()


#%%
#same thing but for BB
pl.imshow(np.log10(sBB))
pl.contour(sBB/100., levels=[1,4,9], colors='r')
pl.ylabel('Intensity')
pl.xlabel('Temp')
pl.colorbar()
pl.show()
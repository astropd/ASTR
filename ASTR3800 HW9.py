# -*- coding: utf-8 -*-
"""
Created on Thu Apr 23 16:20:51 2015

@author: Jacob
"""
import numpy as np
import matplotlib.pyplot as pl
import math 
#11.5 
decay = np. array([0,1,2,3,4,5,6,7,8,9])
obs = np.array([5,19,23,21,14,12,3,2,1,0])/100.

pl.bar(decay, obs)
pl.xlabel('# Decays')
pl.ylabel('Frac. of Results')
pl.title('Fraction per # of Decay')

#%%
#Poisson!
P = lambda v: np.exp(-3)*3**v/math.factorial(v)

prob = np.empty(len(decay))
for i in np.arange(len(decay)):
    prob[i] = P(decay[i])

pl.bar(decay, obs)
pl.plot(prob,color = 'r', label = 'Poisson Prob Fnc')
pl.xlabel('# Decays')
pl.ylabel('Frac. of Results')
pl.title('Fraction per # of Decay')

#%%

def CHIX(obs,exp):
    return ((obs-exp)**2/exp).sum()
print CHIX(obs*100, prob*100), "DoF: 10-4 = 6", CHIX(obs*100, prob*100)/6

# -*- coding: utf-8 -*-
"""
Pulsar Project
Created on Fri Mar 27 08:44:19 2015

@author: Jacob
"""

import numpy as np
import matplotlib.pyplot as pl


# Note: data has been edited to only include full datasets, some information was invalid
dataN= np.genfromtxt('C:\Users\Jacob\Downloads\Pulsar.tsv', comments = '#', delimiter=';')
per = dataN[0] #ms
D1 = dataN[2] #kpc
Derr = dataN[3] #^error
Lum1 = dataN[4] #erg/s
Lumerr = dataN[5] #^error
RFlux1 = dataN[6] #10e-26 W/(m**2*Hz)
RFlux2 = dataN[7] #10e-26 W/(m**2*Hz)
alpha1 = dataN[8] #N/A
alphaerr = dataN[9] #^error
phFlux1 = dataN[10] #ph/(cm**2*s)
phFluxerr = dataN[11] #^error

repl = np.where(RFlux1 == 0)
print repl

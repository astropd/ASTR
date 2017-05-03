# -*- coding: utf-8 -*-
"""
Created on Thu Mar 12 14:03:05 2015

@author: Jacob
"""

import numpy as np
import matplotlib.pyplot as pl

info = np.genfromtxt('C:\Users\Jacob\Documents\College\ASTR\ASTR 3800\correl.txt')
x = np.array(info[:,0])
y = np.array(info[:,1])
p = np.polyfit(x,y,1)

pl.plot(x,y, 'k.', label='Points')
pl.plot(x, p[0]*x+p[1], 'r-', label='LoBF')
pl.xlabel('x')
pl.ylabel('y')
pl.title('Data + LoBF')
pl.legend(loc='best')
pl.show()
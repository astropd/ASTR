# -*- coding: utf-8 -*-
"""
Created on Tue Mar 03 11:29:33 2015

@author: Jacob
"""


#note: A_WHOLE_LOT_OF_DATA is rays.txt, genfromtxt wouldn't read it in with that name...
import numpy as np
import matplotlib.pyplot as pl
data = np.genfromtxt('C:\Users\Jacob\Documents\College\ASTR\ASTR 3800\A_WHOLE_LOT_OF_DATA.txt').T
x = np.array(data[0])
y = np.array(data[1])
z = np.array(data[2])
qx = np.array(data[3])
qy = np.array(data[4])
qz = np.array(data[5])
x_av = np.mean(x)
y_av = np.mean(y)
cats = x-x_av
dogs = y-y_av


pl.plot(cats, dogs, 'k.')

l = np.linspace(0,26)
xn = x[:25]
yn = y[:25]
zn = z[:25]
dx = qx[:25]
dy = qy[:25]
dz = qz[:25]

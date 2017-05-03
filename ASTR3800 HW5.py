# -*- coding: utf-8 -*-
"""
Created on Tue Mar 03 11:00:05 2015
 
@author: Jacob
"""

import numpy as np

def LStar(T, r, dist):
    h = 6.62606957e-34 #Js
    c = 299792458.0 #m/s
    k = 1.3806488e-23 # J/K
    #lam = np.linspace(1e-8, 1e-6,1000) #wavelengths 0-1000
    distarea = dist*3.08567758e16 #convert pc to meters  
    solangle = 4.0*np.pi
    
    Balmost = 2.0*h*c**2/(x**5)*(1.0/(np.exp(h*c/(x*k*T))-1.0)) #J*s^-1*sr^-1*m^-2*^-1
    B1 = Balmost *solangle /distarea #J*s^-1*m^-1*m^-2
    Lum = np.quad()
    
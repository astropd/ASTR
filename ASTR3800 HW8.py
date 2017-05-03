# -*- coding: utf-8 -*-
"""
Created on Thu Mar 19 12:08:47 2015

@author: Jacob
"""

import numpy as np
import matplotlib.pyplot as pl
from astropy.io import fits as pf


image = np.fromfile('C:\Users\Jacob\Downloads\GALAXY.dat', dtype=np.int16)
#GALAXY.dat is the same data set, my computer hates your naming of things for some reason...
#print image.shape #shows that this is actually a 1D array... so we need edit it
imreal = np.reshape(image, (1000,1000))
imgreal = np.reshape(image, (1000,1000))

#plot original image
pl.imshow(imgreal)
pl.title('Original M33')

#plot new color scheme
#pl.imshow(imreal)
#pl.set_cmap('hot')
#pl.title('M33 with cmap=hot')
#pl.colorbar()

#contour things
x = np.arange(0, len(imgreal))
y = np.arange(0,len(imgreal))

#pl.contour(x,y,imgreal)

#intensity
imsum = imgreal.sum(0)
#pl.plot(imsum)
#pl.title('Sum of Intesity')


#peak
most = np.where(imgreal==imgreal.max())
print most, imgreal[281,53] #which is not anywhere near the center, so that's neat

mostx = np.where(imgreal.sum(0)==imgreal.sum(0).max())
mosty = np.where(imgreal.sum(1)==imgreal.sum(1).max())
print mostx, mosty, imgreal[530,511] #much more reasonable answer

#average
cent = imgreal[530-50:530+50, 511-50:511+50]
avg = cent.mean()
print avg

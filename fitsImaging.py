# -*- coding: utf-8 -*-
"""
Created on Thu Feb 16 15:54:40 2017

@author: Jacob
"""

import numpy as np

#making nicer plots
import matplotlib.pyplot as plt

from astropy.utils.data import download_file
from astropy.io import fits

#getting a photo - Horse Head Neb.
im_file = download_file('http://data.astropy.org/tutorials/FITS-images/HorseHead.fits', cache=True)

hdu_list = fits.open(im_file)
hdu_list.info() #seeing what's in it
#%%
#^^ new cell - in Spyder3
im_data = hdu_list[0].data #stores data in 2D array
hdu_list.close()

#%%
plt.imshow(im_data, cmap='gray')
plt.colorbar() #plotted in grayscale, other options avail

#%%
#time to flatten
NBINS = 1000
histogram = plt.hist(im_data.flat, NBINS)

#%%

from matplotlib.colors import LogNorm
plt.imshow(im_data, cmap='gray', norm=LogNorm())
cbar = plt.colorbar(ticks=[5e3, 1e4, 2e4]) #chosen from hist above
cbar.ax.set_yticklabels(['5,000','10,000','20,000'])


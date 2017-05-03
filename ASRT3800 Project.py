# -*- coding: utf-8 -*-
# <nbformat>3.0</nbformat>

# <codecell>

import numpy as np
import matplotlib.pyplot as pl
from mpl_toolkits.mplot3d import Axes3D


# Note: data has been edited to only include full datasets, some information was invalid
dataN= np.genfromtxt('C:\Users\Jacob\Documents\Python Scripts\Pulsar.tsv', comments = '#', delimiter=';')
per1 = dataN[:,0] #ms
Spindown = dataN[:,1] #erg/s
D1 = dataN[:,2] #kpc
Derr = dataN[:,3] #^error
Lum1 = dataN[:,4] #erg/s
Lumerr = dataN[:,5] #^stat error
Lumserr = dataN[:,6] #^sys error
RFlux = dataN[:,7] #10e-26 W/(m**2*Hz)
RFlux2 = dataN[:,8] #10e-26 W/(m**2*Hz)
alpha1 = dataN[:,9] #N/A
alphaerr = dataN[:,10] #^error
EFlux1 = dataN[:,11] #mW/(m**2)
EFluxerr = dataN[:,12] #^error


EFlux = EFlux1*1E3 #mW to W
D = D1*3.08567758E19 #kpc to m
Lum = Lum1*1E-7 #erg to J
per = per1*1E3 #ms to s
fre = 1/(per1*1E3) #ms to 1/s

k = 1.3806488E-23 #m^2* kg* s^-2* K^-1
h = 6.626E-34 #m^2*kg*s^-1
c = 3E8 #m*s^-1
# <codecell>

#Values in dataset had sometimes used the upper limit if no data was provided
#values in the two lists have many zeros (RFlux1) or few values (RFlux2) so this will compact the two into one list

repl = np.where(RFlux == 0) #finds locations of zero values
val = RFlux2[repl] #finds values of the zero locations in other array

#combine!
for i in xrange(len(val)):
    RFlux[repl[0][i]] = val[i]

print np.where(RFlux==0)
#prove that there aren't anymore zero values in the array (bitchin')

# <codecell>

#Weighted Values!


small = np.where(per1 < 100) #find locations for spin rates of <100ms (locations will be used throughout all splitting)
BIG = np.where(per1 > 100 )   #equivalently for >100ms

#DISTANCES
Ds = D1[small]           #find values within the distance list
dserr = Derr[small]      #find values within the error list
DB = D1[BIG]             #find values within the distance list
DBerr = Derr[BIG]        #find values within the error list

#LUMINOSITY
Ls = Lum1[small] #see above for what's going on...
Lserr = Lumerr[small]
LB = Lum1[BIG]
LBerr = Lumerr[BIG]

#ALPHA 
als = alpha1[small]
alerrs = alphaerr[small]
alB = alpha1[BIG]
alerrB = alphaerr[BIG]


#ENERGY FLUX
EFs = EFlux[small]
EFerrs = EFluxerr[small]
EFB = EFlux[BIG]
EFerrB = EFluxerr[BIG]


#using the dividing line of big and small, now take the weighted values of each 
DistS = sum(Ds* (1/(dserr**2)))/sum(1/(dserr**2)) #weighted avg of small D
DistB = sum(DB * (1/(DBerr**2)))/sum(1/(DBerr**2)) #weighted avg of big D (chuckle..."the D"...)
LumS = sum(Ls * (1/(Lserr**2)))/sum(1/(Lserr**2)) #"----" small Lum
LumB = sum(LB * (1/(LBerr**2)))/sum(1/(LBerr**2)) #"----" big lum
ES = sum(EFs * (1/(EFerrs**2)))/sum(1/(EFerrs**2)) #"----" small energy flux
EB = sum(EFB * (1/(EFerrB**2)))/sum(1/(EFerrB**2)) #'-----' big energy flux

# <codecell>

#brightness temperature requires an inverse log:

def inv_logit(p):
    return np.exp(p) / (1 + np.exp(p))
#finding the brightness temperature
TB = h*fre/k*inv_logit(1+(2*h*(fre)**3/RFlux*c**2))
print np.where(TB == 0)



pl.plot(TB, 'k.')

# <codecell>
fig = pl.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(Lum1, per1, Spindown, 'ro')
fig.savefig('3D model!')

# <codecell>
nfit = fitter(per1, Spindown, 2)
pl.plot(per1, nfit[1], 'r', label= 'Linearl Fit')
pl.plot(per1, Spindown, 'k.')

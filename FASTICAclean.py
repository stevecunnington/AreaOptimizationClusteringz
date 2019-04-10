'''
Module for using FASTICA to clean foregrounds
'''

import numpy as np
from sklearn.decomposition import FastICA, PCA

def FASTICAclean(Input, skymask, N_IC=4):
    '''
    Takes input in data cube form but collapsed to HEALpix 1D array maps for
    each frequency bin with dimensions [nz, npix] where nz is number of z bins.
    skymask should be healpy boolean array with size npix, used to cut sky to survey
    footprint. N_IC is number of independent components for FASTICA to try and find
    '''
    Input = np.swapaxes(Input,0,1) #Put in [npix, nz] form which is req'd for FASTICA
    maskedInput = Input[skymask] #only include pixels within sky-footprint
    ica = FastICA(n_components=N_IC, whiten=True)
    S_ = ica.fit_transform(maskedInput) # Reconstruct signals
    A_ = ica.mixing_ # Get estimated mixing matrix
    Recon_FG = np.dot(S_, A_.T) + ica.mean_ #Reconstruct foreground
    Residual = maskedInput - Recon_FG #Residual of fastICA is HI plus any Noise
    CleanFullSky = np.zeros( np.shape(Input) ) #rebuild full sky array
    CleanFullSky.fill(np.nan) #Start with array on NaNs
    CleanFullSky[skymask] = Residual
    CleanFullSky = np.swapaxes(CleanFullSky,0,1) #return to [nz, npix] form
    return CleanFullSky

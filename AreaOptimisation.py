import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import healpy as hp
import scipy
from scipy import signal
from scipy.signal import lfilter
from scipy import integrate
v_21cm = 1420.405751#MHz
d_max = 13.5 #Single dish baseline in metres i.e dish diameter for single-dish IM
c = 3e8 #speed of light

numberofzbins = 24
zmin = 0
zmax = 0.48
deltaz = (zmax-zmin)/numberofzbins
zbins = np.linspace(zmin,zmax,numberofzbins+1)
zbincentres = zbins+(zbins[1]-zbins[0])/2
zbincentres = zbincentres[:len(zbincentres)-1] #remove last value since this is outside of bins
vmin = v_21cm*1e6 / (1+zbincentres[-1]) #Minimum frequency in Hz for highest redshift bin
beamsize = np.degrees( 1.22*c / (vmin*d_max) ) #Maximum beamsize in degrees for IM survey
print('Beamsize = %s deg'%beamsize)
nside = 128
pixsize = hp.nside2resol(nside) #in radians
pixarea = np.degrees(pixsize)**2 #approximate pix area in sq.degrees
lpix = int( np.pi / pixsize ) + 1 #maximum scale of l to probe

n_g_orig = np.load('HealpyMaps/n_g_nside%s-GAEA.npy'%nside)
dT_HI_orig = np.load('HealpyMaps/dT_HI_nside%s-GAEA.npy'%nside)
skymask = np.load('HealpyMaps/skymask_nside%s-GAEA.npy'%nside) #used for excluding area of sky not covered by MICE

#Smooth IM maps to emulate beam effects. Use constant beamsize from maximum redshift bin
#    since constant smoothing is needed for foreground removal and mitigates effect of polarization leakage:
for i in range(numberofzbins):
    dT_HI_orig[i][np.logical_not(skymask)] = hp.UNSEEN
    dT_HI_orig[i] = hp.smoothing(dT_HI_orig[i], fwhm=np.radians(beamsize),verbose=False,lmax=4*nside)
skymask = hp.reorder(skymask,inp='RING',out='NESTED') #put into nested so can select nested areas

def b_HI(z):
    return 0.67 + 0.18*z + 0.05*z**2 #From SKA red book or Alkistis' paper: https://arxiv.org/pdf/1709.07316.pdf
def b_g(z):
    return 1 + 0.84*z #From LSST Science Book

def CorrelationFunction(C_l,lmin,lmax):
    '''
    Weighted correlation function following Menard with power law weighting
    '''
    l = np.arange(1,len(C_l)+1)
    lmask = (l>lmin) & (l<lmax)
    gamma = 1
    W = l ** gamma
    return scipy.integrate.simps( W[lmask] * C_l[lmask] )

def dNdzEstimator():
    '''
    Cross-correlates each IM with n_g and returns estimator statistic at each
    redshift slice to build predicted redshift distribution
    '''
    n_g = np.copy(n_g_orig)
    n_g[np.logical_not(AreaMask)] = np.nan #exclude pixels outside sky coverage
    delta_g = (n_g - np.nanmean(n_g)) / np.nanmean(n_g)
    wgHwHH=[]; wgH=[]; wHH=[]
    for i in range(numberofzbins):
        dT_HI[i][np.logical_not(AreaMask)] = hp.UNSEEN ## exclude pixels outside sky coverage
        delta_g[np.logical_not(AreaMask)] = hp.UNSEEN #    (set to zero so not to ruin healpy Cl)
        #Set scales to probe for correlation function measurements:
        lmax = int( np.pi / np.radians(beamsize) )
        lmin = 50
        if lmin>lmax: lmin = int(lmax*2)
        Cl_HH = hp.anafast(dT_HI[i],lmax=lpix) #Auto power spec
        wHH.append( CorrelationFunction(Cl_HH,lmin,lmax) )
        Cl_gH = hp.anafast(dT_HI[i],delta_g,lmax=lpix)
        if i==-1: #use for power spec viewing
            l = np.arange(1,lpix+2)
            plt.figure(figsize=(10,8))
            plt.plot(l,Cl_gH,label='$C_{gH}$', color='orange')
            plt.plot(l,Cl_HH,label='$C_{HH}$', color='blue')
            plt.plot([lmin,lmin],[np.min(Cl_gH),np.max(Cl_HH)],color='grey',linestyle='--')
            plt.plot([lmax,lmax],[np.min(Cl_gH),np.max(Cl_HH)],color='grey',linestyle='--')
            plt.xlabel('$\ell$',fontsize=16)
            plt.ylabel('$C_\ell$',fontsize=16)
            plt.xscale('log')
            plt.yscale('log')
            plt.legend(fontsize=16)
            plt.show()
            #exit()
        wgH.append( CorrelationFunction(Cl_gH,lmin,lmax) )
        wgHwHH.append( wgH[i] / wHH[i] )
    wgHwHH = np.array(wgHwHH)
    return 1/deltaz * wgHwHH * bHbg * Tbar

def ReceiverNoise(zmin,zmax,Area,beamsize):
    '''
    Returns noise map as a function of redshift bin edges and area surveyed in sq.deg
    '''
    vmax = v_21cm / (1+zmin) #MHz
    vmin = v_21cm / (1+zmax) #MHz
    deltav = vmax - vmin
    v = vmin + deltav/2 #In MHz for T_sys calculations
    deltav = deltav * 1e6 #convert to Hz for power spec
    #T_sys related values from Santos et al. pg 16: https://arxiv.org/pdf/1501.03989.pdf
    T_sky = 60 * (300/v)**2.55 #frequncies in MHz
    T_inst = 20 #in Kelvin for SKA1-MID lowest redshift range
    T_rcvr = 0.1*T_sky + T_inst
    T_sys = (T_rcvr + T_sky) * 1e3 #convert to mK for consistancy with rest of sims
    f_sky = Area/41253 #~41253sq.deg in whole sky - so this gives fraction
    N_dish = 64 #MeerKAT has 64 dishes
    t_obs = 10000 *60*60 #10,000 hours - converted to secs - SKA1
    Omega_beam = 1.133 * np.radians(beamsize)**2
    sigma_N = T_sys * np.sqrt( 4*np.pi * f_sky / (deltav * t_obs * N_dish * Omega_beam) )
    dT_noise = np.random.normal( 0, sigma_N, hp.nside2npix(nside) )
    return dT_noise

def b_HI(z):
    return 0.67 + 0.18*z + 0.05*z**2 #Alkistis' paper: https://arxiv.org/pdf/1709.07316.pdf
def b_g(z):
    return 1 + 0.84*z #Obtained from LSST Science Document

Tbar = 0.0559 + 0.2324*zbincentres - 0.024*zbincentres**2 #Model for Tbar from SKA RedBook
bHbg = b_HI(zbincentres)/b_g(zbincentres)

#Loop over different survey areas to compare constraints on dNdz prediction
dNdz_est = []
Area = [1000, 5000, 10000] #sq deg

for i in range(len(Area)):
    print('Survey Area %s'%(i+1),'of',str(len(Area)))
    dT_HI = np.copy(dT_HI_orig)
    numberofpix = int(Area[i] / pixarea)
    AreaMask = np.zeros(hp.nside2npix(nside))
    maskedindices = np.where(skymask)[0]
    AreaMask[maskedindices[:numberofpix]] = 1
    AreaMask = np.ma.make_mask(AreaMask)
    AreaMask = hp.reorder(AreaMask,inp='NESTED',out='RING')
    #Add Gaussian Noise (different for each sky area):
    for j in range(numberofzbins):
        dT_noise =  ReceiverNoise(zbins[j],zbins[j+1],Area[i],beamsize)
        dT_HI[j] = dT_HI[j] + dT_noise
    dNdz_est.append( dNdzEstimator() )

dNdz_true = np.load('HealpyMaps/dNdz_true-GAEA.npy') #GAEA true optical reddshift distribution

plt.plot(zbincentres, dNdz_true, linestyle='--', color='black', label='True-$z$')
for i in range(len(Area)):
    plt.plot(zbincentres, dNdz_est[i]/np.sum(dNdz_est[i] * deltaz),label='%s deg$^2$'%Area[i])
plt.xlabel('Redshift')
plt.ylabel('d$N$/d$z$')
plt.legend()
plt.show()

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import healpy as hp
import scipy
from scipy import signal
from scipy.signal import lfilter
from scipy import integrate
v_21cm = 1420.405751#MHz
d_max = 13.5 #MeerKAT single dish baseline i.e dish diameter - in metres
c = 3e8 #speed of light

numberofzbins = 24
zmin = 0.2
zmax = 1.4
deltaz = (zmax-zmin)/numberofzbins
zbins = np.linspace(zmin,zmax,numberofzbins+1)
zbincentres = zbins+(zbins[1]-zbins[0])/2
zbincentres = zbincentres[:len(zbincentres)-1] #remove last value since this is outside of bins
vmin = v_21cm*1e6 / (1+zbincentres[-1]) #Minimum frequency in Hz for highest redshift bin
beamsize = np.degrees( 1.22*c / (vmin*d_max) ) #Maximum beamsize in degrees for IM survey

nside = 128
pixsize = hp.nside2resol(nside) #in radians
pixarea = np.degrees(pixsize)**2 #approximate pix area in sq.degrees
print(pixarea)
lpix = int( np.pi / pixsize ) + 1 #maximum scale of l to probe

n_g_orig = np.load('HealpyMaps/n_g-MICE.npy')
dT_HI_orig = np.load('HealpyMaps/dT_HI-MICE.npy')
skymask = np.load('HealpyMaps/skymask-MICE.npy') #used for excluding area of sky not covered by MICE

print('here')

n_g = np.copy(n_g_orig)
n_g[np.logical_not(skymask)] = np.nan #exclude pixels outside sky coverage
delta_g = (n_g - np.nanmean(n_g)) / np.nanmean(n_g)
delta_g[np.logical_not(skymask)] = hp.UNSEEN #exclude pixels outside sky coverage
dT_HI_orig[12][np.logical_not(skymask)] = hp.UNSEEN

Cl_HH_orig = hp.anafast(dT_HI_orig[12],lmax=lpix) #Auto power spec
Cl_gH_orig = hp.anafast(dT_HI_orig[12],delta_g,lmax=lpix)


#Smooth IM maps to emulate beam effects. Use constant beamsize from maximum redshift bin
#    since constant smoothing is needed for foreground removal and mitigates effect of polarization leakage:
for i in range(numberofzbins):
    dT_HI_orig[i][np.logical_not(skymask)] = hp.UNSEEN
    dT_HI_orig[i] = hp.smoothing(dT_HI_orig[i], fwhm=np.radians(beamsize),verbose=False,lmax=4*nside)
skymask = hp.reorder(skymask,inp='RING',out='NESTED') #put into nested so can select nested areas


Cl_HH = hp.anafast(dT_HI_orig[12],lmax=lpix) #Auto power spec
Cl_gH = hp.anafast(dT_HI_orig[12],delta_g,lmax=lpix)
l = np.arange(1,lpix+2)
plt.figure(figsize=(10,8))
plt.plot(l,Cl_gH_orig,label='$C_{gH}$', color='orange')
plt.plot(l,Cl_gH,label='$C_{gH}$', color='orange',linestyle='--')
plt.plot(l,Cl_HH_orig,label='$C_{HH}$', color='blue')
plt.plot(l,Cl_HH,label='$C_{HH}$', color='blue',linestyle='--')
plt.xlabel('$\ell$',fontsize=16)
plt.ylabel('$C_\ell$',fontsize=16)
plt.xscale('log')
plt.yscale('log')
plt.legend(fontsize=16)
plt.show()
exit()

def b_HI(z):
    return 0.67 + 0.18*z + 0.05*z**2 #Obtained from Alkistis see emails - in Alkistis' paper: https://arxiv.org/pdf/1709.07316.pdf
def b_g(z):
    return 1 + 0.84*z #Obtained from D.Alonso clustering-z paper section B 2nd Para

def CorrelationFunction(C_l,lmin,lmax):
    '''
    Weighted correlation function following Menard
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
        print(i)
        dT_HI[i][np.logical_not(AreaMask)] = hp.UNSEEN ## exclude pixels outside sky coverage
        delta_g[np.logical_not(AreaMask)] = hp.UNSEEN #    (set to zero so not to ruin healpy Cl)

        if i==8:
            hp.mollview(dT_HI[i],xsize=2000)
            hp.mollview(delta_g,xsize=2000)
            plt.show()

        #Set scales to probe for correlation function measurements:
        #lcen = int( np.pi / np.radians(beamsize) )
        #lmax = lcen*2
        lmax = 150
        lmin = 50
        #lmin = 2*lcen - 20
        #lmax = 2*lcen + 20
        #lmax = 2*int( np.pi / np.radians(beamsize) )
        #lmin = int(lmax/2)
        Cl_HH = hp.anafast(dT_HI[i],lmax=lpix) #Auto power spec
        wHH.append( CorrelationFunction(Cl_HH,lmin,lmax) )
        Cl_gH = hp.anafast(dT_HI[i],delta_g,lmax=lpix)
        #Do a scipy noise reduction on power spectrum
        n = 50  # the larger n is, the smoother curve will be
        b = [1.0 / n] * n
        a = 1
        Cl_gH_noise = lfilter(b,a,np.abs(Cl_gH)) #noise reduced Cl (of absoulute values)
        #'''
        if i==12:
            l = np.arange(1,lpix+2)
            plt.figure(figsize=(10,8))
            plt.plot(l,Cl_gH,label='$C_{gH}$', color='orange')
            plt.plot(l,Cl_HH,label='$C_{HH}$', color='blue')
            #plt.plot(l,Cl_gH_noise, color='red')
            plt.plot([lmin,lmin],[np.min(Cl_gH),np.max(Cl_HH)],color='grey',linestyle='--')
            plt.plot([lmax,lmax],[np.min(Cl_gH),np.max(Cl_HH)],color='grey',linestyle='--')
            plt.xlabel('$\ell$',fontsize=16)
            plt.ylabel('$C_\ell$',fontsize=16)
            plt.xscale('log')
            plt.yscale('log')
            plt.legend(fontsize=16)
            plt.show()
            #exit()
        #'''

        #wgH.append( CorrelationFunction(Cl_gH_noise,lmin,lmax) )
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


#Loop over different
dNdz_est = []
#Area = [100, 1000, 5000, 10000] #sq deg
Area = [1000, 2000, 6000] #sq deg
#Area = [6000]
for i in range(len(Area)):
    dT_HI = np.copy(dT_HI_orig)
    numberofpix = int(Area[i] / pixarea)
    print(numberofpix)
    AreaMask = np.zeros(hp.nside2npix(nside))

    maskedindices = np.where(skymask)[0]
    AreaMask[maskedindices[:numberofpix]] = 1
    AreaMask = np.ma.make_mask(AreaMask)
    AreaMask = hp.reorder(AreaMask,inp='NESTED',out='RING')
    '''
    dT_HI[12][np.logical_not(AreaMask)] = hp.UNSEEN ## exclude pixels outside sky coverage
    hp.mollview(dT_HI[12],xsize=2000)
    plt.show()
    exit()
    '''
    #Add Gaussian Noise (different for each sky area):
    for j in range(numberofzbins):
        dT_noise =  ReceiverNoise(zbins[j],zbins[j+1],Area[i],beamsize)
        dT_HI[j] = dT_HI[j] + dT_noise
    dNdz_est.append( dNdzEstimator() )

dNdz_true = np.load('HealpyMaps/dNdz_true-MICE.npy') #MICE

plt.plot(zbincentres, dNdz_true, linestyle='--', color='black')
for i in range(len(Area)):
    print(dNdz_est[i])
    plt.plot(zbincentres, dNdz_est[i]/np.sum(dNdz_est[i] * deltaz),label='%s deg$^2$'%Area[i])
plt.legend()
plt.show()

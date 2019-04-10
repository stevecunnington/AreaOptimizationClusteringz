'''
Module for using GMCA to clean foregrounds
'''
import sys
import os
os.chdir('../FGTestsHIClusteringz')
PYGMCALAB_PATH = "pyGMCALab_v1"
sys.path.insert(1,PYGMCALAB_PATH)
from  pyGMCA.bss.amca import BSS_Utils as bu
from  pyGMCA.bss.amca import pyAMCA as pam
os.chdir('../AreaOptimizationClusteringz')

def GMCAclean(Input, N_IC=8):
    t0 = time.time()
    GMCAoutput = pam.AMCA(Input, N_IC,mints=0,nmax=500,AMCA=0,UseP=1)
    t1 = time.time()
    print('GMCAoutput:',str(t1-t0))
    np.save('GMCAoutput',GMCAoutput)
    #GMCAoutput = np.load('GMCAoutput.npy')
    #GMCAoutput = np.load('GMCAoutputNOSYNC.npy')
    S,A = GMCAoutput
    #print(np.shape(GMCAoutput))
    Recon_FG = np.dot(S.T, A.T) #Reconstruct foreground
    Recon_FG = np.swapaxes(Recon_FG,0,1)
    Residual = Input - Recon_FG #Residual of fastICA is HI plus any Noise
    return Residual

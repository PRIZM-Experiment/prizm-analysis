import numpy as np
# import prizmatoid as pzt # missing file/module
import data as da
import util # replaces import read_vna_csv, which is a non-existing module
from scipy import interpolate
from scipy import signal
from scipy import ndimage
import healpy as hp
from pygdsm import GlobalSkyModel16


#ANALYSIS FUNCTIONS

def sky_model(GSM, A, minutes, poly, low, high):
    bin1 = round(1440/minutes)
    bins = np.where(A.any(axis=1))[0].shape[0]
    Gavg = np.nansum(GSM[:,:], axis=0)/bins
    Tdif = GSM - Gavg
    Pavg = np.nansum(A[:,:], axis=0)/bins
    Pdiff = A - Pavg
    pdifsq = Pdiff*Pdiff
    B = np.nansum(pdifsq[:,:], axis=0)
    ptdif = Pdiff*Tdif
    C = np.nansum(ptdif[:,:], axis=0)
    k = C/B
    obs = k*Pavg
    v = []
    for j in range(low, high+2, 2):
        v.append(j)
    v=np.array(v)
    fit = np.polyfit(np.log10(v/100), np.log10(np.abs(obs)), poly)    
    power = np.zeros(int((high-low)/2)+1)
    i = 0
    while i <= poly:
        power = power + fit[poly-i]*(np.log10(v/100))**i
        i += 1
    y = 10**(power)
    return y, obs, k, fit
            



def find_efficiency1(ant_s11, freq, xsmooth, delimiter=','):
    """ Finds the antenna transmission efficiency from the antenna s11 only. 
    
    
    Parameters
    -----------
    ant_s11: filepath to antenna s11 (VNA) file
    xsmooth: critical frequency of lowpass filter

    """
    
    #read the sll files
    s11, time = util.read_vna_data(ant_s11, delimiter=delimiter)
    s11freqs = s11[:,0]
    
    #convert the s11 from dB to linear 
    s11_eff =[0 for x in range(len(s11[:,0]))] 
    for i in range(len(s11[:,0])):
        s11_eff[i]= 10**(s11[i,1]/20)
   
    #interpolate and smooth efficiency (to match frequencies used in data)
    sos = signal.butter(1, xsmooth, btype='lp', output='sos') # low pass filter
    lin=interpolate.interp1d(s11freqs, signal.sosfiltfilt(sos, s11_eff), kind='slinear',fill_value="extrapolate")
    # xnew= np.linspace(0,250000000, num = 4096, endpoint= True)
    xnew = np.arange(30,201) * 1e6
    eff=1-((lin(xnew))**2)
    
    return eff, time


def find_efficiency2(ant_s11, frontend_s11, freq, xsmooth=0.1, delimiter=','):
    """ Finds the antenna transmission efficiency from the antenna and front end (LNA) s11 data. """
    
    #read the sll files
    s11, s11_time =util.read_vna_data(ant_s11, delimiter=delimiter)
    j6, j6_time =util.read_vna_data(frontend_s11, delimiter=delimiter)
    s11freqs = s11[:,0]

    # phase shift due to coaxial cable from antenna to receiver
    s11[:,2] -= coax_phase_shift(s11[:,0])

    # convert s11 to reflection coefficient (magnitude & phase)
    ant_eff = 10**(s11[:,1]/20) 
    ant_ang = s11[:,2]*(np.pi/180) 
    lna_eff = 10**(j6[:,1]/20)
    lna_ang = j6[:,2]*(np.pi/180) 

    # convert mag & phase to real & imaginary
    ant_real = ant_eff*np.cos(ant_ang)
    ant_imag= ant_eff*np.sin(ant_ang)
    lna_real = lna_eff*np.cos(lna_ang)
    lna_imag= lna_eff*np.sin(lna_ang)

    # calculate combined transmission efficiency
    ant_lna_real = ant_real * lna_real - ant_imag * lna_imag
    ant_lna_im = ant_real * lna_imag + ant_imag * lna_real
    eff =  (1 - ant_eff**2) * (1 - lna_eff**2) / ((1 - ant_lna_real)**2 + ant_lna_im**2)

    # low pass filter transmission efficiency
    sos = signal.butter(1, xsmooth, btype='lp', output='sos')
    lin=interpolate.interp1d(s11freqs, signal.sosfiltfilt(sos, eff), kind='slinear',fill_value="extrapolate")
    # interpolate to data frequencies
    xnew = np.arange(30,201) * 1e6
    eff = lin(xnew)
    
    return eff, s11_time, j6_time


def find_calib_eff(load, frontend_s11, xsmooth=0.1, delimiter=','):
    """ Finds the transmission efficiency from the calibrator load and front end s11 data. """
    
    #read the front end sll files
    j6, j6_time =util.read_vna_data(frontend_s11, delimiter=delimiter)
    s11freqs = j6[:,0]

    # calibration load reflection coefficient (assumes real)
    Z_0 = 50 # characteristic impedance
    ant_real = (load - Z_0) / (load + Z_0)
    ant_imag= 0
    ant_eff = ant_real

    # convert s11 to reflection coefficient (magnitude & phase)
    lna_eff = 10**(j6[:,1]/20)
    lna_ang = j6[:,2]*(np.pi/180) 
    # convert mag & phase to real & imaginary
    lna_real = lna_eff*np.cos(lna_ang)
    lna_imag= lna_eff*np.sin(lna_ang)

    # calculate combined transmission efficiency
    ant_lna_real = ant_real * lna_real - ant_imag * lna_imag
    ant_lna_im = ant_real * lna_imag + ant_imag * lna_real
    eff =  (1 - ant_eff**2) * (1 - lna_eff**2) / ((1 - ant_lna_real)**2 + ant_lna_im**2)

    # low pass filter and interpolate to data freq
    eff = filter_interpolate(s11freqs, eff, xsmooth)
    
    return eff, j6_time


def filter_interpolate(s11freqs, eff, xsmooth):
    # low pass filter transmission efficiency
    sos = signal.butter(1, xsmooth, btype='lp', output='sos')
    lin=interpolate.interp1d(s11freqs, signal.sosfiltfilt(sos, eff), kind='slinear',fill_value="extrapolate")
    # interpolate to data frequencies
    xnew = np.arange(30,201) * 1e6
    eff = lin(xnew)
    return eff


def coax_phase_shift(nu):
    cable_len = 0.2 # m
    prop_vel = 0.84 * 2.99792458e8 # s/m
    beta = 2 * np.pi / prop_vel
    phi = 2 * nu * beta * cable_len 
    return phi * 180/np.pi


def s11file2Z(ant_s11, xsmooth=0.1, delimiter=',', coax=True):
    """ Finds the antenna efficiency from the antenna and front end s11 data. """
    
    #read the sll files
    s11, s11_time =util.read_vna_data(ant_s11, delimiter=delimiter)
    s11freqs = s11[:,0]
    
    if coax:
        s11[:,2] -= coax_phase_shift(s11[:,0])
    
    # antenna impedance
    Z_s11_real, Z_s11_imag = s11_to_Z(s11)
    sos = signal.butter(1, xsmooth, btype='lp', output='sos') # low pass filter
    lin=interpolate.interp1d(s11freqs, signal.sosfiltfilt(sos, Z_s11_real), kind='slinear',fill_value="extrapolate")
    xnew = np.arange(30,201) * 1e6
    # xnew = np.linspace(0,250,4096)*1e6
    Z_s11_real = lin(xnew)
    
    return Z_s11_real, Z_s11_imag, s11_time


def s11_to_Z(s11):
    """ Converts s11 data to impedance. """
    s11freqs = s11[:,0]
    
    #convert the angles to radians and amplitudes from dB to linear
    s11_ang = [0 for x in range((len(s11[:,0])))] 
    s11_eff =[0 for x in range((len(s11[:,0])))] 
    for i in range(len(s11[:,0])):
        s11_ang[i] = s11[i,2]*(np.pi/180)
        s11_eff[i]= 10**(s11[i,1]/20)
    
    #find the real and imaginary parts of the s11 data
    s11_real = [0 for x in range((len(s11[:,0])))] 
    s11_imag = [0 for x in range((len(s11[:,0])))] 
    for i in range(len(s11[:,0])):
        s11_real[i]= s11_eff[i]*np.cos(s11_ang[i])
        s11_imag[i]= s11_eff[i]*np.sin(s11_ang[i])
        
    #convert the reflection coefficients (s11) to impedences (Z)    
    Z_s11_real = [0 for x in range((len(s11[:,0])))] 
    Z_s11_imag = [0 for x in range((len(s11[:,0])))] 
    for i in range(len(s11[:,0])):
        Z_s11_imag[i] =(((50*s11_imag[i])*(1-s11_real[i]))-((50+(50*s11_real[i]))*(0-s11_imag[i])))/(((1-s11_real[i])**2)+((0-s11_imag[i])**2))
        Z_s11_real[i] =(((50+(50*s11_real[i]))*(1-s11_real[i])) + ((50*s11_imag[i])*(0-s11_imag[i])))/(((1-s11_real[i])**2)+((0-s11_imag[i])**2))
    
    return Z_s11_real, Z_s11_imag


def Z_to_efficiency(Z_s11_real, Z_s11_imag, Z_j6_real, Z_j6_imag, s11freqs, xsmooth):
    
   #find the top and bottom of the total reflection co-efficent (gamma) after combining the antenna and front end 
    N = len(s11freqs)
    topreal = [0 for x in range(N)]
    botreal = [0 for x in range(N)]
    topimag = [0 for x in range(N)]
    botimag = [0 for x in range(N)]
    for i in range(len(s11freqs)):
        topreal[i] = Z_s11_real[i] - Z_j6_real[i]
        botreal[i] = Z_s11_real[i] + Z_j6_real[i]
        topimag[i] = Z_s11_imag[i] - Z_j6_imag[i]
        botimag[i] = Z_s11_imag[i] + Z_j6_imag[i]
        
    #find the real and imaginary parts of gamma   
    gammareal = [0 for x in range(N)]
    gammaimag = [0 for x in range(N)]
    for i in range(N):
        gammareal[i] = ((topreal[i]*botreal[i])+(topimag[i]*botimag[i]))/((botreal[i]**2)+(botimag[i]**2))
        gammaimag[i] = ((topimag[i]*botreal[i])-(topreal[i]*botimag[i]))/((botreal[i]**2)+(botimag[i]**2))
    
    mag = [0 for x in range(N)] 
    for i in range(N):
        mag[i] = np.sqrt((gammareal[i])**2 + (gammaimag[i])**2)
   
    #interpolate and smooth gamma to match frequencies used in data
    sos = signal.butter(1, xsmooth, btype='lp', output='sos')
    lingamma=interpolate.interp1d(s11freqs, signal.sosfiltfilt(sos, mag), kind='slinear',fill_value="extrapolate")
    # xnew= np.linspace(0,250000000, num = 4096, endpoint= True)
    xnew=np.arange(30,201) * 1e6
    eff=1-((lingamma(xnew))**2)
    
    return eff
    

def find_short(shorton, prizm_data, newlist_antend, antenna, polarization):

    somelist1 = shorton
    newlist_shortend = []

    for i in range(len(somelist1)-1):
        if somelist1[i+1] != somelist1[i] + 1 :
            newlist_shortend.append(somelist1[i])
    
    x=len(somelist1)-1        
    newlist_shortend.append(somelist1[x])        
    
    newlist_shortstart = []

    for i in range(1, len(somelist1)):
        if somelist1[i-1] != somelist1[i] - 1:
            newlist_shortstart.append(somelist1[i])
         
    newlist_shortstart.insert(0,somelist1[0])
    
    print(len(newlist_shortstart))

    short_lengths = list(np.array(newlist_shortend) - np.array(newlist_shortstart))
    short = []
    for i in range(0, len(newlist_shortend)):
        sum = np.zeros(4096)
        for j in range(0, short_lengths[i]+1):
            sum = sum + prizm_data[antenna][polarization][newlist_shortstart[i]+j]
        short.append(sum/(short_lengths[i]+1))
     
       
    dif=[]
    for i in range(0, len(newlist_shortend)-1):
        dif.append(newlist_shortend[i+1] - newlist_shortend[i])
    base = np.average(dif)
  
    for i in range(0, len(dif)):
        if dif[i] > base + base/2:
            short.insert(i+1, pzt.interpolate_short(prizm_data, antenna, polarization)[i])
 
   
    if newlist_shortend[-1] < newlist_antend[-1]:
        short.append(pzt.interpolate_short(prizm_data, antenna, polarization)[-1])
    
    
    return short
    

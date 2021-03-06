import numpy as np
import prizmatoid as pzt
import read_vna_csv as cs
from scipy.interpolate import interp1d
from scipy import signal
from scipy import ndimage
import healpy as hp
from pygdsm import GlobalSkyModel2016


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




def GSM_calibration_data(minperbin, flow, fhigh, antenna, pol):

    
    """Variables: minperbin (2,3,5,10 or 20), flow (greater than 50), fhigh (less than 150), antenna (100 or 70), pol ('EW' or 'NS')"""

    data_arrays=[]
    gsms = []
    if antenna == 100 and pol == 'EW':
        days = 25 #needs to be changed depending on how many good chunks of data there are
        gsm_arrays = []
        with np.load('2018_Good_Data/GSM_temps_100MHz_EW.npz') as temps:
            temp_name = 'T_'+ str(minperbin)
            T = temps[temp_name]
            Tgsm = align_GSMdata(minperbin, T)
        
        for i in range(1, days+1):
            with np.load('2018_Good_Data/100MHz_EW_2018.npz') as data:
                data_name = 'power_'+str(i)
                power = data[data_name]
            with np.load('2018_Good_Data/100MHz_EW_2018_times.npz') as time:
                time_name = 'time_'+str(i)
                time = time[time_name]
            A = freq_binning(flow, fhigh, power)
            day_filtered = rfi_remove(A, 200, 20) 
            freq_bins = int(((fhigh-flow)/2)+1)
            globals()['day%s' % i] = antenna_binning(minperbin, day_filtered,  time, freq_bins)
            globals()['Tgsm%s' % i] = GSM_add_zeros(Tgsm, globals()['day%s' % i], minperbin)
            data_arrays.append(globals()['day%s' % i])
            gsms.append(globals()['Tgsm%s' % i])
    
        alldata =  np.stack(data_arrays)
        with open('100MHz_EW_antenna_allarrays' +'.npy', 'wb') as f:
            np.save(f, alldata)

        tmp = []
        alldataavg = np.zeros((alldata.shape[1], alldata.shape[2]))

        for j in range(0, alldata.shape[1]):
            for k in range(0, alldata.shape[2]):
                for i in range(0, alldata.shape[0]):
                    if alldata[i,j,k] != 0:
                        tmp.append(alldata[i,j,k])
                alldataavg[j,k] = np.nanmean(tmp)
                tmp = []
        with open('100MHz_EW_antenna_average' +'.npy', 'wb') as f:
            np.save(f, alldataavg)
    
        allgsm = np.stack(gsms)

        temp = []
        allgsmavg = np.zeros((allgsm.shape[1], allgsm.shape[2]))

        for j in range(0, allgsm.shape[1]):
            for k in range(0, allgsm.shape[2]):
                for i in range(0, allgsm.shape[0]):
                    if allgsm[i,j,k] != 0:
                        temp.append(allgsm[i,j,k])
                allgsmavg[j,k] = np.nanmean(temp)
                temp = []
        start = int((flow-50)/2)
        end = 51-int((150-fhigh)/2)
        allgsmavg = allgsmavg[:, start:end]
        with open('100MHz_EW_GSM_average' +'.npy', 'wb') as f:
            np.save(f, allgsmavg)
            
    
    
    
    
    
  







#FUNCTIONS FOR DATA RFI FLAGGING, BINNING AND SORTING

def rfi_remove(data, threshold, median_filter_window):
    """ Removes RFI data from the antenna data """   

    medians = np.median(data, axis=0)
    flattened = data - medians
    filtered = ndimage.median_filter(flattened, [1, median_filter_window])
    corrected = flattened - filtered
    MAD = np.median(np.abs(corrected - np.median(corrected)))
    x = (corrected - np.median(corrected) > threshold*MAD)
    rfi_masked_data = np.ma.masked_array(data, x)
   
    return rfi_masked_data


def freq_binning(low, high, dayshift):
    """ Sorts the antenna data into frequency bins matching the GSM (usually 2MHz) and cuts out frequencies that are higher or lower than GSM """
    
    freqlow = []
    freqhigh = []
    for j in range(low-1, high+1, 2):
        freqlow.append(int(4096*j/250))
    for j in range(low+1, high+3, 2):
        freqhigh.append(int(4096*j/250))
    A = []
    for k in range(0, int((high-low)/2)+1):
        pdat = np.sum(dayshift[:, freqlow[k]:freqhigh[k]], axis = 1)/(freqhigh[k] - freqlow[k])
        A.append(pdat)
    A1 = np.array(A)
    A2 = A1.T
    return A2


def align_GSMdata(minutes, GSMdata):
    bin1 = round(1440/minutes)
    #I1 = 346
    #I2 = 530
    I1 = int((360/720)*bin1)
    I2 = int((530/720)*bin1)
    TGSM = []
    zerobin = (bin1 - I2) + I1
    for i in range(zerobin, bin1):
        T2 = GSMdata[i,:].tolist()
        TGSM.append(T2)
    for i in range(0, zerobin):
        T2 = GSMdata[i,:].tolist()
        TGSM.append(T2)
    TGSM = np.array(TGSM)
    return TGSM



def GSM_add_zeros(TGSM, data, minperbin):
    """ Adds zeros to GSM when each antenna data set has no data """

    bin1=int(1440/minperbin)
    GSM = np.zeros((bin1, TGSM.shape[1]))
    on = np.where(data.any(axis=1))[0]
    for i in on:
        GSM[i,:] = TGSM[i,:]
    xx = np.isnan(data)
    for i in range(0, xx.shape[0]):
        for j in range(0, xx.shape[1]):
            if xx[i,j]:
                GSM[i,j] = np.nan 
    return GSM



def antenna_binning(minperbin, data, totaltime, freq_bins):
    
    bin_starts = [i for i in range(0, 1440, minperbin)]
    bin_starts.append(1440)

    minutes = totaltime*1440/6.283185
    start_time = minutes.min()
    end_time = minutes.max()
    first_bin = int((np.floor(start_time/minperbin))*minperbin)
    last_bin = int((np.ceil(end_time/minperbin))*minperbin)

    bin_starts_data = []
    for i in range(first_bin, last_bin, minperbin):
        bin_starts_data.append(i)
    bin_starts_data.append(last_bin)

    binnum = np.zeros(len(minutes))
    for i in range(0, len(minutes)):
        for j in range(0, len(bin_starts)):
            if minutes[i] >= bin_starts[j] and minutes[i] < bin_starts[j+1]:
                binnum[i] = j

    binned = np.zeros((len(bin_starts)-1, freq_bins))
    for i in range(0, len(bin_starts) - 1):
        tmp = []
        for j in range(0, len(minutes)):
            if binnum[j] == i:
                tmp.append(data[j,:])
            binned[i,:] = np.sum(tmp, axis=0)/len(tmp)

    binned[np.isnan(binned)] = 0

    return binned




#FUNCTIONS TO CREATE GSM TEMP ARRAYS


def save_GSM_maps(flow, fhigh):
    gsm_2016 = GlobalSkyModel2016(freq_unit='MHz')

    for i in range(flow,fhigh+2,2):
        gsm_map = gsm_2016.generate(i)
        gsm_map_eq = change_coord(gsm_map, ['G', 'C'])
        gsm_map_lowres = hp.ud_grade(gsm_map_eq, 8, order_in = 'RING', order_out = 'RING')
        np.save('gsm' + 'i', gsm_map_lowres)
        
        

def get_GSM_temps(healpy_beam, flow, fhigh, minperbin):
    
    temperatures1 =[]
    gsm_2016 = GlobalSkyModel2016(freq_unit='MHz')

    for i in range(flow,fhigh+2,2):
        gsm_map = gsm_2016.generate(i)
        gsm_map_eq = change_coord(gsm_map, ['G', 'C'])
        gsm_map_lowres = hp.ud_grade(gsm_map_eq, 256, order_in = 'RING', order_out = 'RING')
        nside=256
        alm_map_eq = hp.map2alm(gsm_map_lowres)
        alm_BEAM = hp.map2alm(healpy_beam[i])
        temp_map = np.full(gsm_map_lowres.size, 1)
        alm_temp_map = hp.map2alm(temp_map)
        integral_beam0 = np.real(np.sum(alm_temp_map*alm_BEAM))
        lmax = np.int(np.round(np.sqrt(2*len(alm_BEAM) - 0.5)))  ;
        m = np.zeros(len(alm_BEAM));


        icur = 0;
        for i in range(0, lmax):
            nn = lmax - i;
            m[icur:icur+nn] = i;
            icur = icur+nn
  

        #phi_rot1 = np.linspace(0,2*np.pi,1440/minperbin)
        phi_rot1 = np.linspace(0,2*np.pi,(1440/minperbin)+1)
        phitmp = phi_rot1.tolist()
        phitmp.pop()
        phi_rot1 = np.array(phitmp)

        temperatures0 = []
        for phi in phi_rot1:
    
            new_alm_beam = alm_BEAM*np.exp(-1j*phi*m)
    

 
            y0 = new_alm_beam*np.conj(alm_map_eq)

            integral_beam_map0 = np.real((np.sum(y0[:lmax])+2*np.sum(y0[lmax:])))



            amp_alm_space0 = integral_beam_map0/integral_beam0
    
            temperatures0.append(amp_alm_space0)

        temperatures1.append(temperatures0)
    
    temp = np.array(temperatures1)
    T = temp.T

    return T


def change_coord(m, coord):
    """ Change coordinates of a HEALPIX map

    Parameters
    ----------
    m : map or array of maps
      map(s) to be rotated
    coord : sequence of two character
      First character is the coordinate system of m, second character
      is the coordinate system of the output map. As in HEALPIX, allowed
      coordinate systems are 'G' (galactic), 'E' (ecliptic) or 'C' (equatorial)

    Example
    -------
    The following rotate m from galactic to equatorial coordinates.
    Notice that m can contain both temperature and polarization.
    >>>> change_coord(m, ['G', 'C'])
    """
    # Basic HEALPix parameters
    npix = m.shape[-1]
    nside = hp.npix2nside(npix)
    ang = hp.pix2ang(nside, np.arange(npix))

    # Select the coordinate transformation
    rot = hp.Rotator(coord=reversed(coord))

    # Convert the coordinates
    new_ang = rot(*ang)
    new_pix = hp.ang2pix(nside, *new_ang)

    return m[..., new_pix]


#FUNCTIONS TO CREATE 'GOOD' DATA PRODUCTS TO BE SAVED

def run_data_noeff(prizm_data):
    """ A bunch of functions put into one function to save space on Notebook """
    

    #makes the switch flags 

    pzt.add_switch_flags(prizm_data, antennas=['100MHz'])
    #pzt.add_temp_flags(prizm_data, antennas=['100MHz'])


    #finds the indices of each mode

    select_antenna = (pzt.shrink_flag(prizm_data['100MHz']['switch_flags']['antenna.scio'], (1,1)) == 1)
    #select_res100 = (pzt.shrink_flag(prizm_data['100MHz']['switch_flags']['res100.scio'], (1,1)) == 1)
    #select_res50 = (pzt.shrink_flag(prizm_data['100MHz']['switch_flags']['res50.scio'], (1,1)) == 1)
    select_short = (pzt.shrink_flag(prizm_data['100MHz']['switch_flags']['short.scio'], (1,1)) == 1)
    #select_noise = (pzt.shrink_flag(prizm_data['100MHz']['switch_flags']['noise.scio'], (1,1)) == 1)
    #select_temp = (prizm_data['100MHz']['temp_flags'] == 1)



    #finding the times

    data_time = prizm_data['100MHz']['time_sys_start.raw'] - prizm_data['100MHz']['time_sys_start.raw'][0]
    #therm_time_start = prizm_data['switch']['time_start_therms.raw'] - prizm_data['100MHz']['time_sys_start.raw'][0]
    #therm_time_stop = prizm_data['switch']['time_stop_therms.raw'] - prizm_data['100MHz']['time_sys_start.raw'][0]

    
    #sorts the data into each type
    
    antenna_data_pol0 = prizm_data['100MHz']['pol0.scio'][select_antenna]
    #res100_data_pol0 = prizm_data['100MHz']['pol0.scio'][select_res100]
    #res50_data_pol0 = prizm_data['100MHz']['pol0.scio'][select_res50]
    short_data_pol0 = prizm_data['100MHz']['pol0.scio'][select_short]
    #noise_data_pol0 = prizm_data['100MHz']['pol0.scio'][select_noise]
    antenna_data_pol1 = prizm_data['100MHz']['pol1.scio'][select_antenna]
    #res100_data_pol1 = prizm_data['100MHz']['pol1.scio'][select_res100]
    #res50_data_pol1 = prizm_data['100MHz']['pol1.scio'][select_res50]
    short_data_pol1 = prizm_data['100MHz']['pol1.scio'][select_short]
    #noise_data_pol1 = prizm_data['100MHz']['pol1.scio'][select_noise]
    
    
    shorton=[i for i, x in enumerate(select_short) if x]
    antennaon=[i for i, x in enumerate(select_antenna) if x]
    


    #functions to find the times when the antenna is on and the values of the short corresponding with each

    (antstart, antend) = find_antenna_times(antennaon)
    (short0, short1) = find_short100(shorton, prizm_data, antend)

    #short0 = pzt.interpolate_short(prizm_data, antenna='100MHz', polarization='pol0.scio')
    #short1 = pzt.interpolate_short(prizm_data, antenna='100MHz', polarization='pol1.scio')
    
    

    #calculate (power - short)/eff for each antenna chunk and then put all the chunks back together into an array

    pmeas0=[]
    pmeas1=[]

    for i in range(len(antstart)):
        pmeas0.append((prizm_data['100MHz']['pol0.scio'][antstart[i]:antend[i]+1,:])-short0[i])
        pmeas1.append((prizm_data['100MHz']['pol1.scio'][antstart[i]:antend[i]+1,:])-short1[i])

    p0 = np.concatenate(pmeas0, axis=0)
    print(p0.shape)
    p1 = np.concatenate(pmeas1, axis=0)
    print(p1.shape)
    
    return p0, p1, select_antenna




def run_data(prizm_data, eff0, eff1):
    """ A bunch of functions put into one function to save space on Notebook """
    

    #makes the switch flags 

    pzt.add_switch_flags(prizm_data, antennas=['100MHz'])
    #pzt.add_temp_flags(prizm_data, antennas=['100MHz'])


    #finds the indices of each mode

    select_antenna = (pzt.shrink_flag(prizm_data['100MHz']['switch_flags']['antenna.scio'], (1,1)) == 1)
    #select_res100 = (pzt.shrink_flag(prizm_data['100MHz']['switch_flags']['res100.scio'], (1,1)) == 1)
    #select_res50 = (pzt.shrink_flag(prizm_data['100MHz']['switch_flags']['res50.scio'], (1,1)) == 1)
    select_short = (pzt.shrink_flag(prizm_data['100MHz']['switch_flags']['short.scio'], (1,1)) == 1)
    #select_noise = (pzt.shrink_flag(prizm_data['100MHz']['switch_flags']['noise.scio'], (1,1)) == 1)
    #select_temp = (prizm_data['100MHz']['temp_flags'] == 1)



    #finding the times

    data_time = prizm_data['100MHz']['time_sys_start.raw'] - prizm_data['100MHz']['time_sys_start.raw'][0]
    #therm_time_start = prizm_data['switch']['time_start_therms.raw'] - prizm_data['100MHz']['time_sys_start.raw'][0]
    #therm_time_stop = prizm_data['switch']['time_stop_therms.raw'] - prizm_data['100MHz']['time_sys_start.raw'][0]

    
    #sorts the data into each type
    
    antenna_data_pol0 = prizm_data['100MHz']['pol0.scio'][select_antenna]
    #res100_data_pol0 = prizm_data['100MHz']['pol0.scio'][select_res100]
    #res50_data_pol0 = prizm_data['100MHz']['pol0.scio'][select_res50]
    short_data_pol0 = prizm_data['100MHz']['pol0.scio'][select_short]
    #noise_data_pol0 = prizm_data['100MHz']['pol0.scio'][select_noise]
    antenna_data_pol1 = prizm_data['100MHz']['pol1.scio'][select_antenna]
    #res100_data_pol1 = prizm_data['100MHz']['pol1.scio'][select_res100]
    #res50_data_pol1 = prizm_data['100MHz']['pol1.scio'][select_res50]
    short_data_pol1 = prizm_data['100MHz']['pol1.scio'][select_short]
    #noise_data_pol1 = prizm_data['100MHz']['pol1.scio'][select_noise]
    
    
    shorton=[i for i, x in enumerate(select_short) if x]
    antennaon=[i for i, x in enumerate(select_antenna) if x]
    


    #functions to find the times when the antenna is on and the values of the short corresponding with each

    (antstart, antend) = find_antenna_times(antennaon)
    (short0, short1) = find_short100(shorton, prizm_data, antend)

    #short0 = pzt.interpolate_short(prizm_data, antenna='100MHz', polarization='pol0.scio')
    #short1 = pzt.interpolate_short(prizm_data, antenna='100MHz', polarization='pol1.scio')
    
    

    #calculate (power - short)/eff for each antenna chunk and then put all the chunks back together into an array

    pmeas0=[]
    pmeas1=[]

    for i in range(len(antstart)):
        pmeas0.append(((prizm_data['100MHz']['pol0.scio'][antstart[i]:antend[i]+1,:])-short0[i])/(eff0))
        pmeas1.append(((prizm_data['100MHz']['pol1.scio'][antstart[i]:antend[i]+1,:])-short1[i])/(eff1))

    p0 = np.concatenate(pmeas0, axis=0)
    print(p0.shape)
    p1 = np.concatenate(pmeas1, axis=0)
    print(p1.shape)
    
    return p0, p1, select_antenna



def find_efficiency1(ant_s11, xsmooth):
    """ Finds the antenna efficiency from the antenna s11 only. """
    
    #read the sll files
    s11=cs.read_vna_file(ant_s11)
    s11freqs = s11[:,0]
    
    #convert the s11 from dB to linear 
    s11_eff =[0 for x in range((len(s11[:,0])))] 
    for i in range(len(s11[:,0])):
        s11_eff[i]= 10**(s11[i,1]/20)
   
    #interpolate and smooth efficiency (to match frequencies used in data)
    sos = signal.butter(1, xsmooth, btype='lp', output='sos')
    lin=interp1d(s11freqs, signal.sosfilt(sos, s11_eff), kind='slinear',fill_value="extrapolate")
    xnew= np.linspace(0,250000000, num = 4096, endpoint= True)
    eff=1-((lin(xnew))**2)
    
    return eff
  


def find_efficiency2(ant_s11, frontend_s11, xsmooth):
    """ Finds the antenna efficiency from the antenna and front end s11 data. """
    
    #read the sll files
    s11=cs.read_vna_file(ant_s11)
    j6=cs.read_vna_file(frontend_s11)
    s11freqs = s11[:,0]
    
    #convert the angles to radians and amplitudes from dB to linear
    s11_ang = [0 for x in range((len(s11[:,0])))] 
    j6_ang = [0 for x in range((len(s11[:,0])))] 
    s11_eff =[0 for x in range((len(s11[:,0])))] 
    j6_eff =[0 for x in range((len(s11[:,0])))] 
    for i in range(len(s11[:,0])):
        s11_ang[i] = s11[i,2]*(np.pi/180)
        j6_ang[i] = j6[i,2]*(np.pi/180)
        s11_eff[i]= 10**(s11[i,1]/20)
        j6_eff[i]= 10**(j6[i,1]/20)
    
    #find the real and imaginary parts of the s11 data
    s11_real = [0 for x in range((len(s11[:,0])))] 
    s11_imag = [0 for x in range((len(s11[:,0])))] 
    j6_real = [0 for x in range((len(s11[:,0])))] 
    j6_imag = [0 for x in range((len(s11[:,0])))] 
    for i in range(len(s11[:,0])):
        s11_real[i]= s11_eff[i]*np.cos(s11_ang[i])
        s11_imag[i]= s11_eff[i]*np.sin(s11_ang[i])
        j6_real[i]= j6_eff[i]*np.cos(j6_ang[i])
        j6_imag[i]= j6_eff[i]*np.sin(j6_ang[i])
        
    #convert the reflection coefficients (s11) to impedences (Z)    
    Z_s11_real = [0 for x in range((len(s11[:,0])))] 
    Z_s11_imag = [0 for x in range((len(s11[:,0])))] 
    Z_j6_real = [0 for x in range((len(s11[:,0])))] 
    Z_j6_imag = [0 for x in range((len(s11[:,0])))] 
    for i in range(len(s11[:,0])):
        Z_s11_imag[i] =(((50*s11_imag[i])*(1-s11_real[i]))-((50+(50*s11_real[i]))*(0-s11_imag[i])))/(((1-s11_real[i])**2)+((0-s11_imag[i])**2))
        Z_s11_real[i] =(((50+(50*s11_real[i]))*(1-s11_real[i])) + ((50*s11_imag[i])*(0-s11_imag[i])))/(((1-s11_real[i])**2)+((0-s11_imag[i])**2))
        Z_j6_real[i] = (((50+(50*j6_real[i]))*(1-j6_real[i])) + ((50*j6_imag[i])*(0-j6_imag[i])))/(((1-j6_real[i])**2)+((0-j6_imag[i])**2))
        Z_j6_imag[i] = (((50*j6_imag[i])*(1-j6_real[i]))-((50+(50*j6_real[i]))*(0-j6_imag[i])))/(((1-j6_real[i])**2)+((0-j6_imag[i])**2))
        
   #find the top and bottom of the total reflection co-efficent (gamma) after combining the antenna and front end     
    topreal = [0 for x in range((len(s11[:,0])))]
    botreal = [0 for x in range((len(s11[:,0])))]
    topimag = [0 for x in range((len(s11[:,0])))]
    botimag = [0 for x in range((len(s11[:,0])))]
    for i in range(len(s11[:,0])):
        topreal[i] = Z_s11_real[i] - Z_j6_real[i]
        botreal[i] = Z_s11_real[i] + Z_j6_real[i]
        topimag[i] = Z_s11_imag[i] + Z_j6_imag[i]
        botimag[i] = Z_s11_imag[i] + Z_j6_imag[i]
        
    #find the real and imaginary parts of gamma   
    gammareal = [0 for x in range((len(s11[:,0])))]
    gammaimag = [0 for x in range((len(s11[:,0])))]
    for i in range(len(s11[:,0])):
        gammareal[i] = ((topreal[i]*botreal[i])+(topimag[i]*botimag[i]))/((botreal[i]**2)+(botimag[i]**2))
        gammaimag[i] = ((topimag[i]*botreal[i])-(topreal[i]*botimag[i]))/((botreal[i]**2)+(botimag[i]**2))
    
    mag = [0 for x in range((len(s11[:,0])))] 
    for i in range(len(s11[:,0])):
        mag[i] = np.sqrt((gammareal[i])**2 + (gammaimag[i])**2)
   
    #interpolate and smooth gamma to match frequencies used in data
    sos = signal.butter(1, xsmooth, btype='lp', output='sos')
    lingamma=interp1d(s11freqs, signal.sosfilt(sos, mag), kind='slinear',fill_value="extrapolate")
    xnew= np.linspace(0,250000000, num = 4096, endpoint= True)
    eff=1-((lingamma(xnew))**2)
    
    return eff


def find_short100(shorton, prizm_data, newlist_antend):

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

    short_lengths = list(np.array(newlist_shortend) - np.array(newlist_shortstart))
    short0 = []
    for i in range(0, len(newlist_shortend)):
        sum = np.zeros(4096)
        for j in range(0, short_lengths[i]+1):
            sum = sum + prizm_data['100MHz']['pol0.scio'][newlist_shortstart[i]+j]
        short0.append(sum/(short_lengths[i]+1))

 
    short1 = []
    for i in range(0, len(newlist_shortend)):
        sum = np.zeros(4096)
        for j in range(0, short_lengths[i]+1):
            sum = sum + prizm_data['100MHz']['pol1.scio'][newlist_shortstart[i]+j]
        short1.append(sum/(short_lengths[i]+1))
     
       
    dif=[]
    for i in range(0, len(newlist_shortend)-1):
        dif.append(newlist_shortend[i+1] - newlist_shortend[i])
    base = np.average(dif)
  
    for i in range(0, len(dif)):
        if dif[i] > base + base/2:
            short0.insert(i+1, pzt.interpolate_short(prizm_data, antenna='100MHz', polarization='pol0.scio')[i]) 
            short1.insert(i+1, pzt.interpolate_short(prizm_data, antenna='100MHz', polarization='pol1.scio')[i])

    #if newlist_shortend[-1] < newlist_antend[-1]:
        #newlist_shortend = newlist_shortend + newlist_shortend[-1:]
   
    if newlist_shortend[-1] < newlist_antend[-1]:
        short0.append(pzt.interpolate_short(prizm_data, antenna='100MHz', polarization='pol0.scio')[-1]) 
        short1.append(pzt.interpolate_short(prizm_data, antenna='100MHz', polarization='pol1.scio')[-1])
    
    return short0, short1
    
    #short_0=prizm_data['100MHz']['pol0.scio'][newlist_short]
    #short_1=prizm_data['100MHz']['pol1.scio'][newlist_short]
    
   
    #return short_0, short_1



def find_antenna_times(antennaon): 
    somelist = antennaon
    newlist_antend = []

    for i in range(0, len(somelist)-1):
        if somelist[i+1] != somelist[i] + 1:
            newlist_antend.append(somelist[i])
    #to get last number
    #get out if last data is antenna
    x=len(somelist)-1        
    newlist_antend.append(somelist[x])
    print('There are ' +str(len(newlist_antend))+ ' end times')
    print(newlist_antend)

    somelist = antennaon
    newlist_antstart = []
    newlist_antstart.append(somelist[0])


    for i in range(1, len(somelist)):
        if somelist[i-1] != somelist[i] - 1:
            newlist_antstart.append(somelist[i])
   
    #x=len(somelist)-1        
    #newlist_antstart.append(somelist[x])
    
    print('There are ' +str(len(newlist_antstart))+ ' start times')
    print(newlist_antstart)
    
    return newlist_antstart, newlist_antend 


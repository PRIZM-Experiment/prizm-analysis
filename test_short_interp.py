import numpy as np
from matplotlib import pyplot as plt

# datadir = '../test_data/2021/100MHz/NS/'

def make_acf(dat,t,dt,tmax):
    '''Function to compute the autocorrelation of a single datastream (t,dat).'''
    n=int(tmax/dt)
    tot=np.zeros(n)
    wt=np.zeros(n)
    nn=len(dat)
    for i in range(nn):
        # dt = 0 included if line below says range(i,nn) and dt<minimum-data-timestep. Not included if it says range(i+1,nn).
        for j in range(i,nn):
            delt=np.abs(t[i]-t[j])
            k=int(delt/dt) 
            if k<n:
                tot[k]=tot[k]+dat[i]*dat[j]
                wt[k]=wt[k]+1
            else:
                break
    return tot,wt

def make_ccf(dat1,t1,dat2,t2,dt,tmax):
    '''DO NOT USE, INCORRECT BECAUSE CCF IS NOT SYMMETRIC IN -dt AND +dt. Function to compute the cross-correlation of two datastreams (t1,dat1) and (t2,dat2).'''
    n=int(tmax/dt)
    tot=np.zeros(n)
    wt=np.zeros(n)
    
    if (len(dat1) != len(t1)) or (len(dat2) != len(t2)):
        raise ValueError('Check that dat1 and t1, and dat2 and t2 have the same length.')
        return 0
    
    nn1=len(dat1)
    nn2=len(dat2)
    
    for i in range(nn1):
        # Since in the CCF we use two different datasets, we need to go through every dat2 point at every dat1 point
        for j in range(nn2):
            delt=np.abs(t1[i]-t2[j])
            k=int(delt/dt) 
            if k<n:
                tot[k]=tot[k]+dat1[i]*dat2[j]
                wt[k]=wt[k]+1
            else:
                continue
    return tot, wt


def make_ccf_2sided(dat1,t1,dat2,t2,dt,tmax):
    '''Function to compute the cross-correlation of two datastreams (t1,dat1) and (t2,dat2). Algorithm is based on looping through every '''
#     n=int(2*tmax/dt)+1 # fully define the bins here instead, 0 in the middle
    bins = np.arange(-tmax,tmax+dt,dt)
    n=len(bins)
    tot=np.zeros(n)
    wt=np.zeros(n)
    
    if (len(dat1) != len(t1)) or (len(dat2) != len(t2)):
        raise ValueError('Check that dat1 and t1, and dat2 and t2 have the same length.')
        return 0
    
    nn1=len(dat1)
    nn2=len(dat2)
    
    for i in range(nn1):
        if i%100 == 0: print('On loop #',i)
        # Since in the CCF we use two different datasets, we need to go through every dat2 point at every dat1 point
        for j in range(nn2):
            delt=t1[i]-t2[j]
            k = np.digitize(delt,bins)
#             k=int(delt/dt) # use numpy digitize instead, remove first and last bin
            if k < n:
                tot[k]=tot[k]+dat1[i]*dat2[j]
                wt[k]=wt[k]+1
            else:
                continue
    
    # Remove first and last bin because... 
    # I think it's something about the counts just being off because of how numpy digitize works when delt does not belong in a bin defined by bins. 
    # The value will overflow into the first or last bin depending on the case.
    tot = tot[1:-1]
    wt = wt[1:-1]
    bins = bins[1:-1]
    
    return tot, wt, bins


def make_ccf_2sided_binning(dat1,t1,dat2,t2,dt,tmax):
    '''Function to compute the cross-correlation of two datastreams (t1,dat1) and (t2,dat2). Algorithm is based on binning each dataset based on the same time bins so we have samples at the same times.'''
    
    
    # Since the two datastreams have different lengths, we need to bin them in time before we can cross-correlate.
    # Bin in bins of dt width
    bins=np.arange(min(t1[0],t2[0]),max(t1[-1],t2[-1]),dt) # bins span the full combined time range of both datasets
    
    # dat1 binning
    bin_indices = np.digitize(x=t1,bins=bins)
    dat1_binned = np.array([np.nanmean(dat1[bin_indices == i]) for i in range(len(bins))])
    # dat1_binned[np.where(np.isnan(dat1_binned))] = 0 # replace all the empty 'nan' bins by zeros
    
    # dat2 binning
    bin_indices = np.digitize(x=t2,bins=bins)
    dat2_binned = np.array([np.nanmean(dat2[bin_indices == i]) for i in range(len(bins))])
    # dat2_binned[np.where(np.isnan(dat2_binned))] = 0 # replace all the empty 'nan' bins by zeros
    
    # Plotting -- for testing purposes
#     plt.figure(figsize=(20,5))
#     plt.plot(t1,dat1,'o',label='d1',alpha=0.1)
#     plt.plot(bins,dat1_binned,'.',label='d1 binned')
#     plt.legend()
#     plt.show()

#     plt.figure(figsize=(20,5))
#     plt.plot(t2,dat2,'o',label='d2',alpha=0.2)
#     plt.plot(bins,dat2_binned,'.',label='d2 binned')
#     plt.legend()
#     plt.show()

#     plt.figure(figsize=(20,5))
#     plt.plot(bins,dat1_binned,'.',label='d1 binned')
#     plt.plot(bins,dat2_binned,'.',label='d2 binned')
#     plt.legend()
#     plt.show()
    
    # Define the dt bins
    dt_bins = np.arange(-len(bins)-1,len(bins),1)*dt
    dt_bins = dt_bins[abs(dt_bins)<=tmax]
    dt_bins_pos = dt_bins[dt_bins>=0] # positive dt's
    dt_bins_neg = dt_bins[dt_bins<0] # negative dt's
    ccf_pos = np.zeros(len(dt_bins_pos))
    ccf_neg = np.zeros(len(dt_bins_neg)+1)
    
    # Loop over positive and negative dt's at the same time
    for ii, dti in enumerate(dt_bins_pos):
        # Shift dat2 with respect to dat1, padding with NaNs
        dat2_shifted_pos = np.insert(arr=dat2_binned,obj=0,values=np.full(ii,np.nan)) # pos dt's
        dat2_shifted_neg = np.append(arr=dat2_binned,values=np.full(ii,np.nan)) # neg dt's
        
        # Pad dat1 array with NaNs
        dat1_padded_pos = np.append(arr=dat1_binned,values=np.full(ii,np.nan)) # pos dt's
        dat1_padded_neg = np.insert(arr=dat1_binned,obj=0,values=np.full(ii,np.nan)) # neg dt's

        # --------------- ARCHIVE - in dvpmt ---------------------------- #
        # Number of samples of pairs of points going into this bin
        # We don't count when empty bins (NaNs) get multiplied, or the NaN padded bins
        # Npos = len(dat1_padded_pos[(~np.isnan(dat1_padded_pos)) & (~np.isnan(dat2_shifted_pos))])
        # Nneg = len(dat1_padded_pos[(~np.isnan(dat1_padded_neg)) & (~np.isnan(dat2_shifted_neg))])
        # ccf_nn_pos = 1/Npos * np.nansum(dat1_padded_pos*dat2_shifted_pos) # pos dt's
        # ccf_nn_neg = 1/Nneg * np.nansum(dat1_padded_neg*dat2_shifted_neg) # neg dt's
        # ---------------------------------------------------------------- #
        
        # Multiply the padded arrays and take the mean (NaNs are automatically masked), 'nn' stands for 'not normalized'
        ccf_nn_pos = np.nanmean(dat1_padded_pos*dat2_shifted_pos) # pos dt's
        ccf_nn_neg = np.nanmean(dat1_padded_neg*dat2_shifted_neg) # neg dt's

        # Divide by the standard deviation of each dataset
        # This has to be re-evaluted for each shift because we only want to evalute the std of points that
        # are included in computing the CCF at this dt shift.
        sig1_pos = np.nanstd(dat1_padded_pos[(~np.isnan(dat1_padded_pos)) & (~np.isnan(dat2_shifted_pos))]) # pos dt's
        sig2_pos = np.nanstd(dat2_shifted_pos[(~np.isnan(dat1_padded_pos)) & (~np.isnan(dat2_shifted_pos))]) # pos dt's

        sig1_neg = np.std(dat1_padded_neg[(~np.isnan(dat1_padded_neg)) & (~np.isnan(dat2_shifted_neg))]) # neg dt's
        sig2_neg = np.std(dat2_shifted_neg[(~np.isnan(dat1_padded_neg)) & (~np.isnan(dat2_shifted_neg))]) # neg dt's

        # Now save to the positive dt and negative dt arrays
        ccf_pos[ii] = ccf_nn_pos/ (sig1_pos*sig2_pos)
        ccf_neg[ii] = ccf_nn_neg/ (sig1_neg*sig2_neg)
            
    # At the end of the loop, put the arrays together
    ccf_tot = np.append(arr=ccf_neg[::-1][:-1],values=ccf_pos)
        
    return dt_bins, ccf_tot


def smooth_acf(tvec,acf,cycle_jump=60,isACF=True):
        '''
        TAKEN FROM kriging.py FOR AVERAGING THE CALIBRATOR DATA WITHIN EACH CAL CYCLE.
        Function to smooth the ACF data by averaging ACF values obtained from calibrator data taken in a single calibrator measurement cycle, while conserving the dt=0 peak due to measurement noise. This translates to averaging data taken in close succession (~a few seconds time gap). Larger timegaps (e.g. order 0.5-1h) separate measurement cycles. We look for large peaks in the timegaps between subsequent data to separate into measurement cycles.
        
        Parameters
        -----------
        cycle_jump: Minimum timegape between subsequent measurements to separate them into two different measurement cycles, in seconds. Default 60 seconds.
        isACF: set to False if using this function for general averaging of another time of (non-ACF) dataset. Default True.
        '''
        # Separate the current dt values into bins within a few minutes of each other, I think this normally corresponds to data taken during the same rotation through calibrators before going back to antenna
        
        if isACF == True:
            time_values = tvec[1:] # we only smooth for dt>0
            acf_values = acf[1:] # "     "      "
            zerobin = acf[0] # save the zero-bin
        else:
            time_values = tvec
            acf_values = acf
            
        tsteps = np.diff(time_values)

        # Create bin edges based on big jumps in the timestep between subsequent measurements
        #bins = find_peaks(tsteps,height=cycle_jump)[0] 
        bins = np.where(tsteps>cycle_jump)[0] # bin edges, minimum jump for separate cycle set to 60 by default
        
        # Split data into the bins
        bin_tgroups = np.split(time_values,bins+1,axis=0)
        bin_acf_groups = np.split(acf_values,bins+1,axis=0)
        
        t_binavg = []
        acf_binavg = []
        acf_binstd = []
        #largest_binwidth = 0 # for testing

        for i in range(len(bin_tgroups)):
            t_binavg.append(np.mean(bin_tgroups[i]))
            acf_binavg.append(np.mean(bin_acf_groups[i]))
            acf_binstd.append(np.std(bin_acf_groups[i]))

        # Add the dt=0 datapoint back in post-smoothing
        if isACF == True:
            smoothed_times = np.concatenate(([0],t_binavg))
            smoothed_acf = np.concatenate(([zerobin],acf_binavg))
            smoothed_std = np.concatenate(([0],acf_binstd)) # here we assume the dt=0 value is exact
        else:
            smoothed_times = np.array(t_binavg)
            smoothed_acf = np.array(acf_binavg)
            smoothed_std = np.array(acf_binstd)
        
        return smoothed_times, smoothed_acf, smoothed_std


def make_ccf_2sided_advanced_binning(dat1,t1,dat2,t2,tmax,cycle_jump=5*60,max_time_diff=5*60,bin_spacing=35*60):
    '''Function to compute the cross-correlation of two datastreams (t1,dat1) and (t2,dat2). Algorithm is based on binning each dataset based on the same time bins so we have samples at the same times. The binning is "advanced" compared to make_ccf_2sided_binning, because it makes sure measurements from separate calibration cycles are in separate bins, and selects the closest in time/does a weighted average (TBD) of the temperature datapoints that fall within the same bin. At the same time it makes sure both time series end up regularly sampled and sampled at the same times before actually computing the CCF.
    
    Parameters
    -----------
    dat1: Calibrator data (primary variable -- the one we want to interpolate using kriging)
    t1: Calibrator data times, in seconds.
    dat2: Temperature data (auxiliary variable used for co-kriging)
    t2: Temperature data times, in seconds.
    (dt: No dt right now)
    dtmax: Maximum CCF time shift to compute, in seconds.
    cycle_jump: In seconds. Time separations larger than the set value of cycle_jump are considered to denote a change of calibration cycle. Default: 5*60 seconds = 5 minutes.
    bin_spacing: Mean separation of two consecutive calibration cycles, in seconds. Default: 35*60 seconds = 35 minutes.
    
    '''
    # Since the two datastreams have different lengths, we need to bin them in time before we can cross-correlate.
    # Bin in ideal bins of width ~35mins, which is the mean separation of two consecutive calibration cycles
    # bin_spacing = 35*60 # seconds
    
    # ---- DAT1 BINNING ----: separating into individual calibration cycles (=icc)
    # retain the mean value and mean time
    t1_icc, dat1_icc, dat1_icc_std = smooth_acf(tvec=t1,acf=dat1,cycle_jump=cycle_jump,isACF=False) # using smooth_acf method from kriging.py to find and average within individual cal cycles

    # associate the mean times to the closest ideal time bin
    # og_bins = np.arange(t1_icc[0],t1_icc[-1]+bin_spacing/2,bin_spacing) # archive
    # make sure we're covering the whole range of *both* datasets
    num_left = max(0,int(np.ceil((t1_icc[0] - t2[0]) / bin_spacing)))
    num_right = max(0,int(np.ceil((t2[-1] - t1_icc[-1]) / bin_spacing)))
    new_start = t1_icc[0] - num_left * bin_spacing
    new_end = t1_icc[-1] + num_right * bin_spacing
    bins = np.arange(new_start, new_end + bin_spacing / 2, bin_spacing) # Generate expanded x array
    # FOR TESTING --------------
    # start = np.where(bins == og_bins[0])[0]
    # end = np.where(bins == og_bins[-1])[0]
    # print(np.array_equal(bins[start[0]:end[0]+1], og_bins))
    # -------------------------
    idx = np.searchsorted(bins,t1_icc)
    idx = np.clip(idx, 1, len(bins) - 1)
    left = bins[idx - 1]
    right = bins[idx]
    closer_to_left = np.abs(t1_icc - left) < np.abs(t1_icc - right)
    bin_indices = np.where(closer_to_left, idx - 1, idx) # ideal bins that are closest to the calibration cycle times
    dat1_binned = np.array([np.nanmean(dat1_icc[bin_indices == i]) for i in range(len(bins))]) # in case there are two that are closest to one bin
    
    # ---- DAT2 BINNING ----: find the temperature times closest to t1_icc (binned calibrator times)
    idx = np.searchsorted(t2,t1_icc) # search for t2 time closest to the t1_icc time
    idx = np.clip(idx, 1, len(t2) - 1)
    left = t2[idx - 1]
    right = t2[idx]
    closer_to_left = np.abs(t1_icc - left) < np.abs(t1_icc - right)
    temp_indices = np.where(closer_to_left, idx - 1, idx)
    temps = dat2[temp_indices]
    # If the closest temperature is over 5 mins away, just set to NaN
    temps[np.abs(t2[temp_indices]-t1_icc) > max_time_diff] = np.nan
    dat2_binned = np.array([np.nanmean(temps[bin_indices == i]) for i in range(len(bins))])
    
    # Where dat2_binned is NaN (aka bin is empty), look for temp time closest to bin time
    # We treat this separately because in the previous step we wanted to make sure we took the temp closest to
    # the actual mean time of the cal cycle, and not the assigned time of the bin. Here since there is no cal cycle 
    # to match, we default to using the temp closest to the assigned bin time.
    nan_mask = np.isnan(dat2_binned)
    idx = np.searchsorted(t2, bins[nan_mask])
    idx = np.clip(idx, 1, len(t2) - 1)
    left = t2[idx - 1]
    right = t2[idx]
    closer_to_left = np.abs(bins[nan_mask] - left) < np.abs(bins[nan_mask] - right)
    closest_times = np.where(closer_to_left, left, right)
    closest_temps = np.where(closer_to_left, dat2[idx - 1], dat2[idx])
    time_diffs = np.abs(bins[nan_mask] - closest_times)
    valid = time_diffs <= max_time_diff
    dat2_binned[nan_mask] = np.where(valid, closest_temps, np.nan)
    
    # Plotting -- for testing purposes
    # plt.figure(figsize=(20,15))
    # ax1 = plt.subplot(311)
    # plt.plot(t1,dat1,'o',label='d1',alpha=0.1)
    # plt.plot(bins,dat1_binned,'.',label='d1 binned')
    # plt.legend()

    # # plt.figure(figsize=(20,5))
    # ax2 = plt.subplot(312,sharex=ax1)
    # plt.plot(t2,dat2,'o',label='d2',alpha=0.2)
    # plt.plot(bins,dat2_binned,'.',label='d2 binned')
    # plt.legend()

    # # plt.figure(figsize=(20,5))
    # ax3 = plt.subplot(313,sharex=ax1)
    # plt.plot(bins,dat1_binned,'.',label='d1 binned')
    # plt.plot(bins,dat2_binned,'.',label='d2 binned')
    # plt.legend()
    # plt.show()
    
    # Define the dt bins
    dt = bin_spacing # the time resolution is hard-coded to 35 mins for now
    dt_bins = np.arange(-len(bins)-1,len(bins),1)*dt
    dt_bins = dt_bins[abs(dt_bins)<=tmax]
    dt_bins_pos = dt_bins[dt_bins>=0] # positive dt's
    dt_bins_neg = dt_bins[dt_bins<0] # negative dt's
    ccf_pos = np.zeros(len(dt_bins_pos))
    ccf_neg = np.zeros(len(dt_bins_neg)+1)
    
    # Loop over positive and negative dt's at the same time
    for ii, dti in enumerate(dt_bins_pos):
        # Shift dat2 with respect to dat1, padding with NaNs
        dat2_shifted_pos = np.insert(arr=dat2_binned,obj=0,values=np.full(ii,np.nan)) # pos dt's
        dat2_shifted_neg = np.append(arr=dat2_binned,values=np.full(ii,np.nan)) # neg dt's
        
        # Pad dat1 array with NaNs
        dat1_padded_pos = np.append(arr=dat1_binned,values=np.full(ii,np.nan)) # pos dt's
        dat1_padded_neg = np.insert(arr=dat1_binned,obj=0,values=np.full(ii,np.nan)) # neg dt's

        # --------------- ARCHIVE - in dvpmt ---------------------------- #
        # Number of samples of pairs of points going into this bin
        # We don't count when empty bins (NaNs) get multiplied, or the NaN padded bins
        # Npos = len(dat1_padded_pos[(~np.isnan(dat1_padded_pos)) & (~np.isnan(dat2_shifted_pos))])
        # Nneg = len(dat1_padded_pos[(~np.isnan(dat1_padded_neg)) & (~np.isnan(dat2_shifted_neg))])
        # ccf_nn_pos = 1/Npos * np.nansum(dat1_padded_pos*dat2_shifted_pos) # pos dt's
        # ccf_nn_neg = 1/Nneg * np.nansum(dat1_padded_neg*dat2_shifted_neg) # neg dt's
        # ---------------------------------------------------------------- #
        
        # Multiply the padded arrays and take the mean (NaNs are automatically masked), 'nn' stands for 'not normalized'
        ccf_nn_pos = np.nanmean(dat1_padded_pos*dat2_shifted_pos) # pos dt's
        ccf_nn_neg = np.nanmean(dat1_padded_neg*dat2_shifted_neg) # neg dt's

        # Divide by the standard deviation of each dataset
        # This has to be re-evaluted for each shift because we only want to evalute the std of points that
        # are included in computing the CCF at this dt shift.
        sig1_pos = np.nanstd(dat1_padded_pos[(~np.isnan(dat1_padded_pos)) & (~np.isnan(dat2_shifted_pos))]) # pos dt's
        sig2_pos = np.nanstd(dat2_shifted_pos[(~np.isnan(dat1_padded_pos)) & (~np.isnan(dat2_shifted_pos))]) # pos dt's

        sig1_neg = np.std(dat1_padded_neg[(~np.isnan(dat1_padded_neg)) & (~np.isnan(dat2_shifted_neg))]) # neg dt's
        sig2_neg = np.std(dat2_shifted_neg[(~np.isnan(dat1_padded_neg)) & (~np.isnan(dat2_shifted_neg))]) # neg dt's

        # Now save to the positive dt and negative dt arrays
        ccf_pos[ii] = ccf_nn_pos/ (sig1_pos*sig2_pos)
        ccf_neg[ii] = ccf_nn_neg/ (sig1_neg*sig2_neg)
            
    # At the end of the loop, put the arrays together
    ccf_tot = np.append(arr=ccf_neg[::-1][:-1],values=ccf_pos)
    
    return dt_bins, ccf_tot






def make_acf_original(dat,t,dt,tmax):
    n=int(tmax/dt)
    tot=np.zeros(n)
    wt=np.zeros(n)
    nn=len(dat)
    for i in range(nn):
        for j in range(i+1,nn):
            delt=np.abs(t[i]-t[j])
            k=int(delt/dt)
            if k<n:
                tot[k]=tot[k]+dat[i]*dat[j]
                wt[k]=wt[k]+1
            else:
                break
    return tot,wt

plt.ion()

# Original Jon code
# dat=np.load(datadir+'shortdata_meas_2021_100MHz_NS.npy')
# lst=np.load(datadir+'shortlst_2021_100MHz_NS.npy')
# t=np.load(datadir+'shortsystime_2021_100MHz_NS.npy')

# tmax=1.6374e9
# tmin=1.6365e9

# mask=(t>tmin)&(t<tmax)
# tt=t[mask]
# dd=dat[mask,:]
# ll=lst[mask]

# dt=600
# tot,wt=make_acf(dd[:,1000]-dd[:,1000].mean(),tt,dt=dt,tmax=2*86400)
# tvec=np.arange(len(tot))*dt
# mm=wt>30
# plt.clf()
# plt.plot(tvec[mm]/3600,(tot[mm]/wt[mm]),'.')
# plt.show()
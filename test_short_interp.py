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
import numpy as np
import scipy
from zoneinfo import ZoneInfo
import datetime as datetime
import PrizmCalibration as cl
import util
import helper_functions as hf
from data import get_temperatures, interpolate_temperature
from numpy.polynomial.legendre import Legendre
from numpy.polynomial.chebyshev import Chebyshev
from numpy.polynomial.polynomial import Polynomial


def P_Z_func(Z, P_50, P_100, eta_50, eta_100, P_short):
    """
    Compute power contribution from current noise.
    For calibrator current noise, Z=50 or 100.
    For antenna current noise, Z=Z_ant(nu).
    """
    # current noise for 50Ohm calibrator
    P_Z = ((P_50-P_short)/eta_50 - (P_100-P_short)/eta_100) / (1/eta_50 - 4 / eta_100)
    # apply factor for current noise for Z
    P_Z *= (Z / 50)**2

    return P_Z

def JNC_calib(
    Z,
    P_ant,
    P_short,
    P_50,
    P_100,
    T_amb,
    eta_ant,
    eta_50,
    eta_100,
    Z_ant,
    delta_t=6.4,
    delta_nu=250/4096
    ):
    P_Z50 = P_Z_func(50, P_50, P_100, eta_50, eta_100, P_short)
    if Z == 50:
        P_load = P_50
        eta_load = eta_50
        P_Zload = P_Z50
    elif Z == 100:
        P_load = P_100
        eta_load = eta_100
        P_Zload = P_Z50 * 4
    else:
        print('invalid Z')

    P_Zant = P_Z50 * (Z_ant / 50) ** 2
        
    K = T_amb[:, None] * eta_load / (P_load - P_short - P_Zload)
    Tsky = K * (P_ant - P_short - P_Zant) / eta_ant

    return Tsky, K

def JNC_calib_nocurrent(
    P_ant,
    P_short,
    P_load,
    T_amb,
    eta_ant,
    delta_t=6.4,
    delta_nu=250/4096
    ):
    """JNC calibration assuming current noise = 0."""

    K = T_amb[:, None] / (P_load - P_short)
    Tsky = K * (P_ant - P_short) / eta_ant

    return Tsky, K


def fit_logpoly(nu, data, base, deg, nu_ref=100, Tcmb=2.725):
    """Fit log polynomial to data"""
    x = np.log10(nu / nu_ref)
    y = np.log10(data - Tcmb)

    notnan = np.isfinite(data)
    if base == 'legendre':
        fit = Legendre.fit(x[notnan], y[notnan], deg=deg)
    elif base == 'chebyshev':
        fit = Chebyshev.fit(x[notnan], y[notnan], deg=deg)
    elif base == 'monomial':
        fit = Polynomial.fit(x[notnan], y[notnan], deg=deg)

    return 10 ** fit(x) + Tcmb


def split_lst_days(lst, systime):
    systime_splits = np.digitize((systime-systime[0])/3600/24, np.arange(1,360))
    num_days = max(systime_splits)+1

    lst_splits = np.zeros(len(lst), dtype=int)
    start = 0
    systime_start = 0
    for i in range(num_days):
        if np.any(systime_splits==i):
            lst_stop = np.where(np.diff(lst[systime_splits==i]) < 0)[0]
            if len(lst_stop) == 1:
                stop = systime_start + lst_stop[0] + 1
            else:
                stop = np.where(systime_splits == i)[0][-1] + 1
            lst_splits[start:stop] = i

            start = stop
            systime_start = np.where(systime_splits == i)[0][-1] + 1
    lst_splits[start:] = i+1

    num_lst_days = max(lst_splits)+1
    days_w_data = np.unique(lst_splits)
    return lst_splits, days_w_data


def prep_data(instrument, channel, year, t=-1):
    path2file = f'../Data/{year}/{instrument[:-3]}/{channel}/'
    file_ending = f'_{year}_{instrument[:-3]}{channel}.npy'

    with open(path2file + 'systime' + file_ending, 'rb') as f:
        systime = np.load(f)[:t]
    with open(path2file + 'lst' + file_ending, 'rb') as f:
        lst = np.load(f)[:t]

    raw_freq = np.linspace(0, 250, 4096)

    path2file = f'jobs/filtering/'
    file_ending = f'_{instrument[:-3]}{channel}_filtered_sym4.npy'

    with open(path2file + 'res50' + file_ending, 'rb') as f:
        P_50 = np.load(f)[:t]
    with open(path2file + 'res100' + file_ending, 'rb') as f:
        P_100 = np.load(f)[:t]
    with open(path2file + 'shorts' + file_ending, 'rb') as f:
        P_short = np.load(f)[:t]
    with open(path2file + 'rfi_flagged' + file_ending, 'rb') as f:
        P_ant = np.load(f)[:t]

    weights = get_weights(P_ant, lst, tsize=0.5)
    P_ant = freq_bin_weighted(P_ant, raw_freq, 1, 30, 200, weights)
    weights = get_weights(P_short, lst, tsize=0.5)
    P_short = freq_bin_weighted(P_short, raw_freq, 1, 30, 200, weights)
    weights = get_weights(P_50, lst, tsize=0.5)
    P_50 = freq_bin_weighted(P_50, raw_freq, 1, 30, 200, weights)
    weights = get_weights(P_100, lst, tsize=0.5)
    P_100 = freq_bin_weighted(P_100, raw_freq, 1, 30, 200, weights)
    
    rfi_freq = np.arange(30,201)

    #### discard all nan spectra
    mask = np.isfinite(P_ant)
    spec_mask = np.any(mask, axis=1)
    mask = mask[spec_mask]

    P_ant = P_ant[spec_mask]
    P_short = P_short[spec_mask]
    P_50 = P_50[spec_mask]
    P_100 = P_100[spec_mask]
    systime = systime[spec_mask]
    lst = lst[spec_mask]

    ### ambient temperature
    T_phys, interp_inds = get_temp(instrument, channel, year, systime, calib='ambient')

    P_ant = P_ant[interp_inds]
    P_short = P_short[interp_inds]
    P_50 = P_50[interp_inds]
    P_100 = P_100[interp_inds]
    systime = systime[interp_inds]
    lst = lst[interp_inds]
    mask = mask[interp_inds]

    return P_ant, P_short, P_50, P_100, systime, lst, rfi_freq, mask, T_phys, interp_inds


def interpolate_eta_eff(systime, instrument, channel, date_id, xsmooth=0.1):
    # get fe path
    dates = ['2022-01-09', '2022-01-14', '2022-02-22']
    date = dates[date_id]
    delimiter = '\t'
    frontend_s11_file_path = f'/project/s/sievers/prizm/prizm_vna_2021/{instrument[:-3]}-LNA/{date}-PRIZM-{instrument[:-3]}-{channel}-LNA.txt'
    
    # get ant paths
    month = [10,10,11,11,11,11,12,12,'01', '01', '02']
    day = [21, 27, '03', 12, 14, 26, '02', 10, '09', 14, 22]
    ant_effs = []
    ant_ts = []
    for i in range(len(month)):
        if type(month[i]) == str:
            y='2022'
        else:
            y='2021'
        antenna_s11_file_path = f'/project/s/sievers/prizm/prizm_vna_2021/{instrument[:-3]}/{y}-{month[i]}-{day[i]}-PRIZM-{instrument[:-3]}-{channel}.txt'   
        ant_eff, t, _ = cl.find_efficiency2(
            antenna_s11_file_path,
            frontend_s11_file_path,
            xsmooth=xsmooth,
            delimiter='\t'
            )
        ant_effs.append(ant_eff)
        ant_ts.append(t)
    ant_effs = np.array(ant_effs)

    systime_s11 = np.zeros(len(ant_ts))
    for i, t in enumerate(ant_ts):
        local_time = datetime.datetime(t[0], t[1], t[2], t[3], t[4], t[5])
        # Make sure the datetime object is aware of its timezone (UTC in this case)
        local_time = local_time.replace(tzinfo=ZoneInfo('Africa/Johannesburg'))
        # Convert to Unix time (seconds since epoch)
        unix_time = int(local_time.timestamp())
        systime_s11[i] = unix_time

    eta_ant = np.zeros((len(systime), ant_effs.shape[1]))
    for i in range(ant_effs.shape[1]):
        interp = scipy.interpolate.interp1d(
            systime_s11-systime_s11[0],
            ant_effs[:,i],
            kind='linear',
            fill_value=(ant_effs[0,i], ant_effs[-1,i])
            )
        eta_ant[:, i] = interp(systime - systime_s11[0])
    return eta_ant, ant_effs, systime_s11


def interpolate_Z_ant(systime, instrument, channel, date_id, xsmooth=0.1):
    # get ant paths
    month = [10,10,11,11,11,11,12,12,'01', '01', '02']
    day = [21, 27, '03', 12, 14, 26, '02', 10, '09', 14, 22]
    Zant_reals = []
    Zant_ts = []
    for i in range(len(month)):
        if type(month[i]) == str:
            y='2022'
        else:
            y='2021'
        antenna_s11_file_path = f'/project/s/sievers/prizm/prizm_vna_2021/{instrument[:-3]}/{y}-{month[i]}-{day[i]}-PRIZM-{instrument[:-3]}-{channel}.txt'   
        Zant_real, _, Zant_t = cl.s11file2Z(antenna_s11_file_path, xsmooth=xsmooth, delimiter='\t')
        Zant_reals.append(Zant_real)
        Zant_ts.append(Zant_t)
    Zant_reals = np.array(Zant_reals)

    systime_s11 = np.zeros(len(Zant_ts))
    for i, t in enumerate(Zant_ts):
        local_time = datetime.datetime(t[0], t[1], t[2], t[3], t[4], t[5])
        # Make sure the datetime object is aware of its timezone (UTC in this case)
        local_time = local_time.replace(tzinfo=ZoneInfo('Africa/Johannesburg'))
        # Convert to Unix time (seconds since epoch)
        unix_time = int(local_time.timestamp())
        systime_s11[i] = unix_time

    Zant_interp = np.zeros((len(systime), Zant_reals.shape[1]))
    for i in range(Zant_reals.shape[1]):
        interp = scipy.interpolate.interp1d(
            systime_s11-systime_s11[0],
            Zant_reals[:,i],
            kind='linear',
            fill_value=(Zant_reals[0,i], Zant_reals[-1,i])
            )
        Zant_interp[:, i] = interp(systime - systime_s11[0])
    return Zant_interp, Zant_reals, systime_s11
    

def get_eta_ant(instrument, channel, year, highpass, lowpass, date_id):
# transmission efficiency (eta ant)
    dates = ['2022-01-09', '2022-01-14', '2022-02-22']
    date = dates[date_id]
    if year == '2021':
        delimiter = '\t'
        antenna_s11_file_path = f'/project/s/sievers/prizm/prizm_vna_2021/{instrument[:-3]}/{date}-PRIZM-{instrument[:-3]}-{channel}.txt'
        frontend_s11_file_path = f'/project/s/sievers/prizm/prizm_vna_2021/{instrument[:-3]}-LNA/{date}-PRIZM-{instrument[:-3]}-{channel}-LNA.txt'

    fe_eff, ant_t, fe_t = cl.find_efficiency2(antenna_s11_file_path,frontend_s11_file_path, xsmooth=0.1, delimiter=delimiter)

    # eta, _ = hf.truncate(fe_eff[None,:], np.arange(30,201), highpass, lowpass)
    # eta = eta[0]
    return fe_eff


def get_temp(instrument, channel, year, systime, calib='ambient'):
    # path to temp erature directories (2021 only)
    path2dir = f'/project/s/sievers/prizm/marion2022/prizm-{instrument[:-3]}/data_{instrument}/temperatures'

    if calib == 'ambient':
        temp_type = 'ambient'
    # elif calib == 'noise':
    #     temp_type = f'{instrument[:-3]}{channel}_noise'
        
    T_phys = get_temperatures(instrument, path2dir, temp_type=temp_type)
    T_phys, systime_phys, interp_inds = interpolate_temperature(T_phys[f'temp_{temp_type}'], (T_phys['time_sys_start'] + T_phys['time_sys_stop']) / 2, systime)

    return T_phys, interp_inds


def freq_bin_weighted(data, freq, binsize, flow, fhigh, weights): 
    weights = np.where(np.isfinite(weights), weights, 1e-16)
    bins_f = np.arange(flow, fhigh + binsize, binsize)
    nan_mask = np.isnan(data)
    data_binned = np.array([
        np.nansum(data[:, np.abs(freq - i) < binsize/2] * weights[np.abs(freq - i) < binsize/2], axis=1) / \
            np.nansum(~nan_mask[:, np.abs(freq - i) < binsize/2] * weights[np.abs(freq - i) < binsize/2], axis=1) \
            for i in bins_f]).T  
    flag_rate = np.array([
        np.sum(nan_mask[:, np.abs(freq - i) < binsize/2], axis=1) / np.sum(np.abs(freq - i) < binsize/2) \
            for i in bins_f]).T  
    data_binned[flag_rate > 0.5] = np.nan
    return data_binned

def get_weights(data, lst, tsize=0.5):
    temp_stability = []   
    for LST in np.arange(0,24+tsize, tsize):
        idx = np.where(np.abs(lst - LST) < tsize/2)[0]
        curr = data[idx]
        temp_stability.append(np.nanstd(curr, axis=0))   
    weights = 1 / np.nanmean(temp_stability, axis=0) 
    return weights


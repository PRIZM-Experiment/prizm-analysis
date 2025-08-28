import sys
import h5py
import numpy as np
from mpi4py import MPI
from rfi_flagging import RFI_flagging
from Ripple_Filter import filter_ripple, fill_nans
from time import time


def upload_data(instrument, channel, year, calib, nans_in_spec=True):
    '''
    Uploads antenna, short, lst.
    '''
    if nans_in_spec:
        path2file = f'../../Data/{year}/{instrument[:-3]}/{channel}/'
        file_ending = f'_{year}_{instrument[:-3]}{channel}.npy'
    else:
        path2file = 'filtering/'
        file_ending = f'_{instrument[:-3]}{channel}_spl_fill_nan.npy'

    with open(path2file + calib + file_ending, 'rb') as f:
        calib_data = np.load(f)
    
    return calib_data


if __name__ == '__main__':

    instrument = sys.argv[1]
    channel = sys.argv[2]
    year = sys.argv[3]
    lam = float(sys.argv[4])
    calib = sys.argv[5]

    print(instrument, channel, year, calib)
    
    # set False if starting from data with nans filled
    nans_in_spec = True

    nans_filled_file = f'filtering/{calib}_{instrument[:-3]}{channel}_spl_fill_nan.npy'
    filtered_file = f'filtering/{calib}_{instrument[:-3]}{channel}_filtered_sym4.npy'

    #######################################################

    freq = np.linspace(0, 250, 4096)

    # FPGA harmonics
    fpga = [0, 31.25, 62.5, 93.75, 125, 156.25, 187.5, 218.75, 250]
    
    # Initialize MPI
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()  

    if rank == 0:
        timer1 = time()

        data_prod, raw_freq = upload_data(instrument, channel, year, calib, nans_in_spec)
        original_shape = data_prod.shape

        # replace spikes from fpga in spectra
        if nans_in_spec:
            RFI = RFI_flagging(data_prod, raw_freq, None)
            RFI.flag_known_rfi(fpga, 0.3)
            data_prod = RFI.data

        # remove all nan spectra
        notnan = np.isfinite(data_prod)
        valid_spec = np.any(notnan, axis=1)
        valid_data = data_prod[valid_spec]

        # chunk up
        chunks = np.array_split(valid_data, size) 
        file_headers = range(size)
        with h5py.File('filtering/data_chunks.h5', 'w') as f:
            for i in file_headers:
                f.create_dataset(str(i), data=chunks[i])

    else:
        file_headers = None 
    # Scatter chunks to all ranks
    file_header = comm.scatter(file_headers, root=0)

    with h5py.File('filtering/data_chunks.h5', 'r') as f:
        chunk = f[str(file_header)][:]

    # replace nans
    if nans_in_spec:
        chunk_nonans = fill_nans(chunk, freq, lam=lam)
        np.save(f'filtering/nonans_chunk{file_header}.npy', chunk_nonans)
    else:
        chunk_nonans = chunk

    # filter
    chunk_filtered = filter_ripple(chunk_nonans, remove_levels=[5,6], wavelet='sym4')
    np.save(f'filtering/filtered_chunk{file_header}.npy', chunk_filtered)

    # Gather results on rank 0
    file_headers = comm.gather(file_header, root=0)

    if rank == 0:
        print(f'time: {(time() - timer1) / 60} min')

        # gather chunks
        valid_filtered = [np.load(f'filtering/filtered_chunk{i}.npy') for i in file_headers]
        print([chunk.shape for chunk in valid_filtered])

        # Combine results into a single array
        valid_filtered = np.vstack(valid_filtered)

        # put back all nan spectra
        data_filtered = np.full(original_shape, np.nan)
        print(data_filtered.shape, valid_spec.shape, valid_filtered.shape)
        data_filtered[valid_spec] = valid_filtered

        # save
        np.save(filtered_file, data_filtered)

        if nans_in_spec:
            valid_nonans = [np.load(f'filtering/nonans_chunk{i}.npy') for i in file_headers]
            valid_nonans = np.vstack(valid_nonans)
            data_nonans = np.full(original_shape, np.nan)
            data_nonans[valid_spec] = valid_nonans
            np.save(nans_filled_file, data_nonans)
            



        


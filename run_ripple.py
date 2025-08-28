import sys
import h5py
import numpy as np
from mpi4py import MPI
from Ripple_Filter import filter_ripple, fill_nans
from time import time


if __name__ == '__main__': 

    instrument = sys.argv[1]
    channel = sys.argv[2]
    year = sys.argv[3]
    lam = float(sys.argv[4])

    # set False if starting from data with nans filled
    nans_in_spec = True

    flagged_data_file = f'rfi_flagged_{instrument[:-3]}{channel}_20250523_specthresh2_noshortsub.npy'
    nans_filled_file = f'filtering/rfi_flagged_{instrument[:-3]}{channel}_spl_fill_nan.npy'
    filtered_file = f'filtering/rfi_flagged_{instrument[:-3]}{channel}_filtered_sym4.npy'


    freq = np.linspace(0, 250, 4096)
    #######################################################

    # Initialize MPI
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size() 
    
    if rank == 0:
        timer1 = time()

        # upload data
        with open(flagged_data_file, 'rb') as f:
            data_prod = np.load(f)
        original_shape = data_prod.shape
        flag_mask = np.isnan(data_prod)

        if not nans_in_spec:
            with open(nans_filled_file, 'rb') as f:
                data_prod = np.load(f)
        
        # remove all nan spectra
        notnan = np.isfinite(data_prod)
        valid_spec = np.any(notnan, axis=1)
        valid_data = data_prod[valid_spec]
        print(f'valid data shape {valid_data.shape}')
        print(f'size={size}')

        # chunk up
        chunks = np.array_split(valid_data, size) 
        print([chunk.shape for chunk in chunks])
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

    # replace nans in data
    if nans_in_spec:
        chunk_nonans = fill_nans(chunk, freq, lam=lam)
        np.save(f'filtering/nonans_chunk{file_header}.npy', chunk_nonans)
    else:
        print('no filling nans')
        chunk_nonans = chunk

    # filter!
    chunk_filtered = filter_ripple(chunk_nonans, remove_levels=[5,6], wavelet='sym4')
    np.save(f'filtering/filtered_chunk{file_header}.npy', chunk_filtered)

    # Gather results on rank 0
    file_headers = comm.gather(file_header, root=0)

    if rank == 0:
        print(f'time: {(time() - timer1) / 60} min')

        # gather chunks
        valid_filtered = [np.load(f'filtering/filtered_chunk{i}.npy') for i in file_headers]

        # Combine results into a single array
        valid_filtered = np.vstack(valid_filtered)

        # put back all nans
        data_filtered = np.full(original_shape, np.nan)
        data_filtered[valid_spec] = valid_filtered
        data_filtered[flag_mask] = np.nan

        # save filtered data
        np.save(filtered_file, data_filtered)

        # save nan filled data
        if nans_in_spec:
            valid_nonans = [np.load(f'filtering/nonans_chunk{i}.npy') for i in file_headers]
            valid_nonans = np.vstack(valid_nonans)
            data_nonans = np.full(original_shape, np.nan)
            data_nonans[valid_spec] = valid_nonans
            np.save(nans_filled_file, data_nonans)
            



        


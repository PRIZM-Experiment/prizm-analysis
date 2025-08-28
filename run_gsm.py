import sys
import numpy as np
from gsm_data import GSMData

if __name__ == '__main__':
    instrument = sys.argv[1]
    channel = sys.argv[2]
    instrument = '100MHz'
    channel = 'EW'
    for instrument in ['100MHz', '70MHz']:
        for channel in ['NS', 'EW']:
            minperbin = 10
            model = 'GSM16'
            beta = None
            nside=128
            zerobin = 0
            saved_maps = False

            # antenna site altitude and tilt
            if instrument == '70MHz':
                tilt_N = 2.25 # recorded 2-2.5
                tilt_E = 1.5
                site_latitude = -49.88714
            elif instrument == '100MHz':
                tilt_N = 1.5
                tilt_E = 2.5
                site_latitude = -49.88722

            # horizon profile
            if instrument == '70MHz':
                SHAPES = np.load('./Horizon/PRIZM_70MHz_height_horizon.npz')
            elif instrument == '100MHz':
                SHAPES = np.load('./Horizon/PRIZM_100MHz_height_horizon.npz')
                
            profile, az = SHAPES['profile'], SHAPES['azimuth']

            gsm = GSMData(
                instrument,
                channel,
                minperbin,
                horizon=profile,
                tilt_N=tilt_N,
                tilt_E=tilt_E,
                site_latitude=site_latitude,
                nside=nside
                )
            Tgsm = gsm(model=model, beta=beta, saved_maps=saved_maps, zerobin=zerobin)
            print(Tgsm.shape)
            np.save(f'./GSM_averages/new/{instrument}_{channel}_GSM_average_{minperbin}minbins_20250617.npy', Tgsm)

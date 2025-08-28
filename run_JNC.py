import numpy as np
import JNC as jnc
import PrizmCalibration as cl


if __name__ == '__main__':
    load = 50
    instrument = '100MHz'
    channel = 'NS' 
    year = '2021'
    isinterp = 'interp'

    for instrument in ['100MHz', '70MHz']:
        for channel in ['EW', 'NS']:
            # clear memory
            P_ant = None
            P_short = None
            P_50 = None
            P_100 = None
            eta_ant_interps = None
            Zant_interps = None

            # get data
            P_ant, P_short, P_50, P_100, systime, lst, freq, mask, T_phys, interp_inds = jnc.prep_data(
                instrument,
                channel,
                year
                )
            
            for load in [50, 100]:    
                xsmooth_ant = 0.1
                xsmooth_load = 0.01

                # get measured and interpolated antenna and load transmission efficiencies
                eta_ant_interps = []
                eta_ant_raws = []
                Zant_interps = []
                Zant_raws = []
                eta_100s = []
                eta_50s = []
                for date_id in range(1,3):
                    curr_interp, curr_raw, systime_s11 = jnc.interpolate_eta_eff(
                        systime,
                        instrument,
                        channel,
                        date_id,
                        xsmooth=xsmooth_ant)

                    curr_zinterp, curr_zraw, _ = jnc.interpolate_Z_ant(systime, instrument, channel, date_id, xsmooth=0.1)

                    dates = ['2022-01-09', '2022-01-14', '2022-02-22']
                    frontend_s11_file_path = f'/project/s/sievers/prizm/prizm_vna_2021/{instrument[:-3]}-LNA/{dates[date_id]}-PRIZM-{instrument[:-3]}-{channel}-LNA.txt'
                    curr_100, _ = cl.find_calib_eff(
                        100,
                        frontend_s11_file_path,
                        xsmooth=xsmooth_load,
                        delimiter='\t'
                        )
                    curr_50, _ = cl.find_calib_eff(
                        50,
                        frontend_s11_file_path,
                        xsmooth=xsmooth_load,
                        delimiter='\t'
                        )

                    eta_ant_interps.append(curr_interp)
                    eta_ant_raws.append(curr_raw)
                    Zant_interps.append(curr_zinterp)
                    Zant_raws.append(curr_zraw)
                    eta_100s.append(curr_100)
                    eta_50s.append(curr_50)

                eta_ant_interp = np.mean(eta_ant_interps, axis=0)
                eta_ant_raw = np.mean(eta_ant_raws, axis=0)
                Zant_interp = np.mean(Zant_interps, axis=0)
                Zant_raw = np.mean(Zant_raws, axis=0)
                eta_100 = np.mean(eta_100s, axis=0)
                eta_50 = np.mean(eta_50s, axis=0)

                # JNC calib, interpolated transmission efficiencies
                Tsky, K_JNC = jnc.JNC_calib(
                    load,
                    P_ant,
                    P_short,
                    P_50,
                    P_100,
                    T_phys,
                    eta_ant_interp,
                    eta_50,
                    eta_100,
                    Zant_interp
                    )

                # uncertainties from LNA impedance
                Tsky_bounds = []
                for i in range(2):
                    Tsky_curr, _ = jnc.JNC_calib(
                        load,
                        P_ant,
                        P_short,
                        P_50,
                        P_100,
                        T_phys,
                        eta_ant_interps[i],
                        eta_50s[i],
                        eta_100s[i],
                        Zant_interps[i]
                        )
                    Tsky_bounds.append(Tsky_curr)
                LNA_err = np.abs(Tsky_bounds[0] - Tsky_bounds[1]) / 2

                # uncertainties from antenna impedance
                ant_group = np.digitize(systime, bins=systime_s11[1:-1])
                eta_ant_bounds = [eta_ant_raw[ant_group], eta_ant_raw[ant_group+1]]
                Zant_bounds = [Zant_raw[ant_group], Zant_raw[ant_group+1]]
                Tsky_bounds = []
                for i in range(2):
                    Tsky_curr, _ = jnc.JNC_calib(
                        load,
                        P_ant,
                        P_short,
                        P_50,
                        P_100,
                        T_phys,
                        eta_ant_bounds[i],
                        eta_50,
                        eta_100,
                        Zant_bounds[i]
                        )
                    Tsky_bounds.append(Tsky_curr)
                ant_err = np.abs(Tsky_bounds[0] - Tsky_bounds[1]) / 2

                save_file = f'JNC_calibs/{instrument}_{channel}_{load}_{isinterp}_zant_actual.npz'
                np.savez(
                    save_file,
                    Tsky=Tsky, 
                    K=K_JNC,
                    LNA_err = LNA_err,
                    ant_err = ant_err,
                    lst=lst,
                    systime=systime,
                    eta_ant=eta_ant_interp,
                    eta_ant_raw=eta_ant_raw,
                    Zant=Zant_interp,
                    Zant_raw=Zant_raw,
                    systime_s11=systime_s11,
                    Tsky_bounds=Tsky_bounds
                    )

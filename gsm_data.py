import numpy as np
import healpy
import scipy
import sys
from pygdsm import GlobalSkyModel16, HaslamSkyModel, LowFrequencySkyModel

class GSMData:

    def __init__(self, instrument, channel, min_per_bin, horizon=[], site_latitude=-46.88694, tilt_N=0, tilt_E=0, ant_orientation=0, nside=128):
        self.min_per_bin = min_per_bin
        self.instrument = instrument
        self.channel = channel
        self.nside = nside
        self.beam_dict = self.get_beam_dict(ant_orientation)
        self.healpy_beam = self.get_healpy_beam(site_latitude, tilt_N, tilt_E)
        self.healpy_horizon = self.get_healpy_horizon(horizon, site_latitude)
        self.gsm_data = None
        self.chromaticity = None

    def __call__(self, model='GSM16', beta=None, saved_maps=True, zerobin=0):
        self.get_GSM_temps(ref=None, model='GSM16', beta=None, saved_maps=saved_maps).align_GSMdata(zerobin)
        return self.gsm_data

    def get_beam_dict(self, ant_orientation):
        """ 
        Returns dictionary containing beam patterns retrieved from data files.
        Dictionary entries are labeled by frequency, and are a phi x theta grid.
        """
       
        dir_parent='./Beams'
        if self.instrument == '100MHz':
            file_name='results_pattern_100mhz_total90.dat'
        
        elif self.instrument == '70MHz':
            file_name='results_pattern_70mhz_total90.dat'

        # Initializes the dictionary which will hold the beam information.
        beam_dict = {}

        # Establishes the `file_path` which points to the beam simulation of interest.
        file_path = dir_parent + '/' + file_name

        # Stores the beam simulation data in the NumPy array `beam_sim_data`, and
        # ignores the header as a comment starting with '#'.
        beam_sim_data = np.loadtxt(file_path, delimiter=',', comments='#')

        # Reads the beam file header, cleans it from unwanted characters, and keeps
        # only the numerical entries - these correspond to the different frequencies
        # for which the beam has been simulated.
        beam_file = open(file_path, 'r')
        header = beam_file.readline()
        frequencies = header.strip('#\n, ').split(',')[2:]
        beam_file.close()

        # Converts the `frequencies` list to a NumPy array and converts its values
        # to MHz through a division by 1e6.
        frequencies = np.asarray(frequencies, dtype='float') / 1e6

        # Extracts the spherial coordinates `theta` and `phi` stored in
        # `beam_sim_data` and converts their units from degrees to radians.
        # coordinates have resolution 2 degrees, phi [0,360] and theta [0,90]
        theta = np.unique(beam_sim_data[:, 0]) * np.pi / 180
        phi = np.unique(beam_sim_data[:, 1]) * np.pi / 180

        # Discards the coordinate information from `beam_sim_data` since this is
        # already stored in the meshgrid, as well as in `theta` and `phi`.
        beam_sim_data = beam_sim_data[:, 2:]

        # Stores spherical coordinates in `beam_dict`.
        beam_dict['theta'] = theta
        beam_dict['phi'] = phi
        
        # index for rotation of beam
        if ant_orientation < 0:
            # ensure angle is positive, rotate in degrees E from N
            ant_orientation += 360
        if self.channel == 'NS':
            ant_orientation += 90
        rot_phi = (ant_orientation)%360 // 2

        # Stores the beam profile for each frequency in `beam_dict`.
        for index, entry in enumerate(frequencies):
            # Reshape the `beam_sim_data` so that its dimensions are compatible with
            # those of `theta` and `phi`. This way different slices of `beam_sim_data`
            # correspond to the beam for different frequencies.
            reshaped_beam_sim = np.reshape(beam_sim_data[:, index],
                                           [len(phi), len(theta)])
                
            reshaped_beam_sim = np.concatenate((reshaped_beam_sim[rot_phi:-1], reshaped_beam_sim[:rot_phi], reshaped_beam_sim[rot_phi,None]), axis=0)

            # Stores the reshaped beam in `beam_dict` under the appropriate
            # frequency key.
            beam_dict[entry] = reshaped_beam_sim

        # Returns the beam information in a dictionary format.
        return beam_dict

    def get_healpy_beam(self, site_latitude, tilt_N, tilt_E):
        # Initializes the dictionary which will hold the HealPy version of the beam.
        healpy_beam_dict = {}

        # Extracts the frequencies for which beams are available in `beam_dict`.
        frequencies = [key for key in self.beam_dict.keys() if isinstance(key, float)]
        n_freq = len(frequencies)

        # Initializes a HealPy pixelization and associated spherical coordinates.
        healpy_npix = healpy.nside2npix(self.nside)
        healpy_theta, healpy_phi = healpy.pix2ang(self.nside,
                                                  np.arange(healpy_npix))

        # Stores spherical coordinates in `healpy_beam_dict`.
        healpy_beam_dict['theta'] = healpy_theta
        healpy_beam_dict['phi'] = healpy_phi

        # SciPy 2D interpolation forces us to proceed in chunks of constant
        # coordinate `healpy_theta`. Below we find the indices at which
        # `healpy_theta` changes.
        indices = np.where(np.diff(healpy_theta) != 0)[0]
        indices = np.append(0, indices + 1)

        # Initializes the NumPy array which will contain the normalization factor
        # for each beam.
        beam_norms = np.zeros(n_freq)
        
        # initialize rotators for latitude and tilt
        rotator_phi = healpy.Rotator(deg=True, rot=[0, 90 - site_latitude - tilt_N])
        rotator_theta = healpy.Rotator(deg=True, rot=[tilt_E, 0])
        
        
        # Loops over the frequencies for which the beam has been
        # simulated.
        for i, frequency in enumerate(frequencies):

            # Computes the actual beam from the information contained in
            # `beam_dict`.
            beam = 10 ** (self.beam_dict[frequency] / 10)

            # Interpolates beam.
            beam_interp = scipy.interpolate.interp2d(self.beam_dict['theta'],
                                                     self.beam_dict['phi'],
                                                     beam,
                                                     kind='cubic',
                                                     fill_value=0)

            # Initializes `healpy_beam`, the HealPy version of the beam.
            healpy_beam = np.zeros(len(healpy_theta))

            # Constructs the HealPy beam.
            for j in range(int(len(indices) / 2) + 2):
                start = indices[j]
                end = indices[j + 1]
                healpy_beam[start:end] = beam_interp(healpy_theta[start],
                                                     healpy_phi[start:end])[:, 0]                

            # Fills `beam_norms` with the appropriate normalization factors for
            # each HealPy beam.
            beam_norms[i] = np.sqrt(np.sum(healpy_beam ** 2)) 

            # rotate to site latitude & antenna tilts
            healpy_beam = rotator_phi.rotate_map_pixel(healpy_beam)
            healpy_beam = rotator_theta.rotate_map_pixel(healpy_beam)           
            
            # normalize
            healpy_beam_dict[frequency] = healpy_beam / beam_norms[i]

        # Adds the beam normalizations as a separate entry in `heapy_beam_dict`.
        healpy_beam_dict['normalization'] = beam_norms

        # Returns the HealPy version of the beam in a dictionary format.
        return healpy_beam_dict

    def get_healpy_horizon(self, horizon, site_latitude):
        # Initializes 'horizon_mask', horizon blockage mask in healpy pix
        horizon_mask = np.ones(len(self.healpy_beam['theta']))
    
        if len(horizon) > 0:
            # convert horizon angle to phi in radian
            horizon = (90 - horizon) * np.pi / 180
            
            # interp horizon profile
            interp_horizon = scipy.interpolate.interp1d(np.arange(361) * np.pi / 180, horizon, kind='cubic')
            healpy_horizon = interp_horizon(self.healpy_beam['phi'])
         
            # set mask=0 below horizon
            horizon_mask[self.healpy_beam['theta'] > healpy_horizon] = 0
            # set mask to 0.5 for pixels on horizon
            horizon_mask[np.abs(self.healpy_beam['theta'] - healpy_horizon) < 1e-16] = 0.5
        else:
            # set mask=0 below horizon
            horizon_mask[self.healpy_beam['theta'] > np.pi/2] = 0
            # set mask to 0.5 for pixels on horizon
            horizon_mask[np.abs(self.healpy_beam['theta'] - np.pi/2) < 1e-16] = 0.5
            
        # rotate horizon to site latitude
        beam_rotation = healpy.rotator.Rotator([0, 90 - site_latitude, 0])
        horizon_mask = beam_rotation.rotate_map_pixel(horizon_mask)
        
        # round interpolated mask to 0 or 1
        horizon_mask[horizon_mask - 0.5 > 1e-16] = 1
        horizon_mask[horizon_mask - 0.5 < -1e-16] = 0
            
        return horizon_mask

    @staticmethod
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
        """
        # Basic HEALPix parameters
        npix = m.shape[-1]
        nside = healpy.npix2nside(npix)
        ang = healpy.pix2ang(nside, np.arange(npix))

        # Select the coordinate transformation
        rot = healpy.Rotator(coord=reversed(coord))

        # Convert the coordinates
        new_ang = rot(*ang)
        new_pix = healpy.ang2pix(nside, *new_ang)

        return m[..., new_pix]

    def get_GSM_temps(self, ref=None, model='GSM16', beta=None, saved_maps=True):
        temperatures1 = []
        # upload saved or generate GSM maps 

        if ref:
            if saved_maps:
                gsm_map_lowres = np.load('./gsm_maps/gsm_{}.npy'.format(ref))
            else:
                gsm_map_lowres = self.get_GSM_map(ref, beta=beta, model=model)
                
        for i in range(30, 202, 2):
            if i%10 == 0:
                print(i)
            if not ref:
                if saved_maps:
                    gsm_map_lowres = np.load('./gsm_maps/gsm_{}.npy'.format(i))
                else:
                    gsm_map_lowres = self.get_GSM_map(i, beta=beta, model=model)
            
            # convert map in spherical coordinates to spherical harmonic coefficients
            alm_map_eq = healpy.map2alm(gsm_map_lowres)
            alm_BEAM = healpy.map2alm(self.healpy_beam[i])
            alm_BEAM_horizon = healpy.map2alm(self.healpy_horizon * self.healpy_beam[i])
            
            temp_map = np.full(gsm_map_lowres.size, 1)
            alm_temp_map = healpy.map2alm(temp_map)
            integral_beam0 = np.real(np.sum(alm_temp_map * alm_BEAM))
            
            # spherical harmonics
            lmax = int(np.round(np.sqrt(2 * len(alm_BEAM) - 0.5)))
            m = np.zeros(len(alm_BEAM))
            icur = 0
            # alm only gives postive m bc real values == symmetric
            for i in range(0, lmax):
                nn = lmax - i
                m[icur:icur + nn] = i
                icur = icur + nn
            
            phi_rot1 = np.linspace(0, 2 * np.pi, int((1440 / self.min_per_bin) + 1))
            phitmp = phi_rot1.tolist()
            phitmp.pop()
            phi_rot1 = np.array(phitmp)

            temperatures0 = []
            for phi in phi_rot1:
                new_alm_beam = alm_BEAM_horizon * np.exp(-1j * phi * m)

                y0 = new_alm_beam * np.conj(alm_map_eq)
                
                # y0[:lmax] is m=0, 2 * y0[lmax:] for +-m
                integral_beam_map0 = np.real((np.sum(y0[:lmax]) + 2 * np.sum(y0[lmax:])))

                amp_alm_space0 = integral_beam_map0 / integral_beam0

                temperatures0.append(amp_alm_space0)

            temperatures1.append(temperatures0)
        self.gsm_data = np.array(temperatures1).T
        return self
    
    
    def get_chromaticity(self, ref=76, lst=None, lstbinned=True, beta=None, model='GSM16', saved_maps=True):
        temperatures1 = []
        # generate GSM maps at ref freq
        if saved_maps:
            gsm_map_lowres = np.load('./gsm_maps/gsm_{}.npy'.format(ref))
        else:
            gsm_map_lowres = self.get_GSM_map(ref, beta=beta, model=model)
        # compute spherical harmonics coefficients of ref maps
        alm_map_eq = healpy.map2alm(gsm_map_lowres)
        alm_BEAM_ref = healpy.map2alm(self.healpy_horizon * self.healpy_beam[ref])
        
        lmax_ref = int(np.round(np.sqrt(2 * len(alm_BEAM_ref) - 0.5)))
        m_ref = np.zeros(len(alm_BEAM_ref))
        icur = 0
        for i in range(0, lmax_ref):
            nn = lmax_ref - i
            m_ref[icur:icur + nn] = i
            icur = icur + nn
        
        # LST dependence as phi
        if lstbinned:
            min_per_day = 24 * 60
            phi_rot1 = np.linspace(0, 2 * np.pi, int((min_per_day / self.min_per_bin) + 1))
            phitmp = phi_rot1.tolist()
            phitmp.pop()
            phi_rot1 = np.array(phitmp)
        else:
            phi_rot1 = lst * (2 * np.pi) / 24 

        integral_beam_map0_ref = []
        for phi in phi_rot1:
            new_alm_beam_ref = alm_BEAM_ref * np.exp(-1j * phi * m_ref)
            y0_ref = new_alm_beam_ref * np.conj(alm_map_eq)
            integral_beam_map0_ref.append(np.real((np.sum(y0_ref[:lmax_ref]) + 2 * np.sum(y0_ref[lmax_ref:]))))
        
        for i in np.arange(40, 140, 2):
            # compute spherical harmonics coefficients of beam
            alm_BEAM = healpy.map2alm(self.healpy_horizon * self.healpy_beam[i])
            
            # spherical harmonics, m
            lmax = int(np.round(np.sqrt(2 * len(alm_BEAM) - 0.5)))
            m = np.zeros(len(alm_BEAM))
            icur = 0
            for i in range(0, lmax):
                nn = lmax - i
                m[icur:icur + nn] = i
                icur = icur + nn

            temperatures0 = []
            for n, phi in enumerate(phi_rot1):
                new_alm_beam = alm_BEAM * np.exp(-1j * phi * m)

                y0 = new_alm_beam * np.conj(alm_map_eq)

                integral_beam_map0 = np.real((np.sum(y0[:lmax]) + 2 * np.sum(y0[lmax:])))
                
                amp_alm_space0 = integral_beam_map0 / integral_beam_map0_ref[n]

                temperatures0.append(amp_alm_space0)

            temperatures1.append(temperatures0)
        self.chromaticity = np.array(temperatures1).T
        return self    
    
    
    def align_GSMdata(self, zerobin):
        min_per_day = 24 * 60
        num_lst_bins = round(min_per_day / self.min_per_bin)
        self.gsm_data = self.gsm_data[(np.arange(num_lst_bins) + zerobin) % num_lst_bins]
        return self
    
    def get_GSM_map(self, freq, beta=None, model='GSM16'):
        ''' 
        Generate low resolution GSM map in equatorial coordinates using pygdsm.
        '''
        if model == 'GSM16':
            gsm = GlobalSkyModel16(freq_unit='MHz', resolution='low')
        elif model == 'Haslam':
            gsm = HaslamSkyModel(freq_unit='MHz', spectral_index=beta)
        elif model == 'LFSS':
            gsm = LowFrequencySkyModel(freq_unit='MHz')
        else:
            print('Model must be GSM16, Haslam, or LFSS')
        gsm_map = gsm.generate(freq)
            
        # convert galactic to equatorial coordinates
        gsm_map_eq = self.change_coord(gsm_map, ['G', 'C'])
            
        # lower resolution of maps
        gsm_map_lowres = healpy.ud_grade(gsm_map_eq, self.nside, order_in='RING', order_out='RING')
        return gsm_map_lowres
                
    def save_GSM_maps(self, nside=256):
        for i in range(30, 202, 2):
            gsm_map_lowres = self.get_GSM_map(i, nside)
            np.save('./gsm_maps/gsm_{}.npy'.format(i), gsm_map_lowres)
            
            
def get_desired_frequencies(Tgsm, flow, fhigh):
    start = int((flow-30)/2)
    end = int((fhigh-flow)/2) + start +1
    return Tgsm[:, start:end]
    

import copy
import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter1d

class Pulsestack:

    def load_from_pdv(self, filename, stokes):
        # Read in the pdv data using numpy's handy loadtxt
        self.pdvfile = filename
        dat = np.loadtxt(filename)

        # Record what Stokes parameter is being read
        self.stokes = stokes

        # Figure out from the first few columns what the dimensions of the
        # pulsestack are, and store these to class variables
        self.npulses = int(dat[-1,0] + 1)
        self.nfreqs  = int(dat[-1,1] + 1)
        self.nbins   = int(dat[-1,2] + 1)

        # Pull out the column for this Stokes, and reshape it into a
        # pulsestack (i.e. 2D array)
        stokes_col = "IQUV".find(stokes)
        if stokes_col == -1:
            raise ValueError("Unrecognised Stokes parameter {}".format(stokes))
        stokes_col += 3 # (Stokes I starts in column 3)

        try:
            self.values = np.reshape(dat[:,stokes_col], (self.npulses, self.nfreqs, self.nbins))
            # Frequency scrunch
            self.values = np.mean(self.values, axis=1)
            self.nfreqs = 1
        except:
            raise IndexError("Could not read Stokes {} data from {}".format(stokes, filename))

        # We will assume that the pulsestack array is always contiguous
        # (i.e. no gaps), so that the pulse numbers and longitude bins can
        # be represented by just two numbers: a reference pulse/bin and a
        # step size
        self.first_pulse = 0
        self.first_phase = 0

        self.dpulse     = 1 # i.e. 1 pulse per row
        self.dphase_deg = 360/self.nbins

    def get_pulse_bin(self, pulse, inrange=True):
        # This can return fractional (non-integer) values
        pulse_bin = (pulse - self.first_pulse)/self.dpulse
        if inrange == True:
            if pulse_bin < 0:
                pulse_bin = 0
            elif pulse_bin > self.values.shape[0] - 1:
                pulse_bin = self.values.shape[0] - 1
        return pulse_bin

    def get_phase_bin(self, phase_deg, inrange=True):
        # This can return fractional (non-integer) values
        return (phase_deg - self.first_phase)/self.dphase_deg
        if inrange == True:
            if phase_deg_bin < 0:
                phase_deg_bin = 0
            elif phase_deg_bin > self.values.shape[0] - 1:
                phase_deg_bin = self.values.shape[0] - 1
        return phase_deg_bin

    def set_fiducial_phase(self, phase_deg):
        self.first_phase -= phase_deg

    def crop(self, pulse_range=None, phase_deg_range=None, inplace=True):
        '''
        pulse_range and phase_deg_range are expected to be two-element, 1D lists/arrays
        Putting in None for either of the limits will default to no cropping for that side
        inplace, if set to true, will change this instance of the class
        Otherwise it will create a cropped copy
        By default, this is a very forgiving function. If ranges are given outside the
        available pulse/phase range, it will simply not crop (rather than raise an error)
        '''
        if inplace == True:
            newps = self
        else:
            newps = copy.copy(self)

        # Set defaults (= no cropping)
        pulse_bin_range = [0, self.values.shape[0]]
        phase_bin_range = [0, self.values.shape[1]]

        if pulse_range is not None:

            if pulse_range[0] is not None:
                pulse_bin_range[0] = int(np.round(newps.get_pulse_bin(pulse_range[0])))
            if pulse_range[1] is not None:
                pulse_bin_range[1] = int(np.round(newps.get_pulse_bin(pulse_range[1])))

            newps.values       = newps.values[pulse_bin_range[0]:pulse_bin_range[1], :]
            newps.first_pulse += pulse_bin_range[0]*newps.dpulse

        if phase_deg_range is not None:

            if phase_deg_range[0] is not None:
                phase_bin_range[0] = int(np.round(newps.get_phase_bin(phase_deg_range[0])))
            if phase_deg_range[1] is not None:
                phase_bin_range[1] = int(np.round(newps.get_phase_bin(phase_deg_range[1])))

            newps.values        = newps.values[:,phase_bin_range[0]:phase_bin_range[1]]
            newps.first_phase  += phase_bin_range[0]*newps.dphase_deg

        newps.npulses, newps.nbins = newps.values.shape

        return newps

    def smooth_with_gaussian(self, sigma, inplace=True):
        '''
        Smooth the pulses with a gaussian filter using scipy's gaussian_filter1d function
        sigma is the gaussian width (analogous to the sigma parameter in
        gaussian_filter1d) in degrees
        '''
        if inplace == True:
            newps = self
        else:
            newps = copy.copy(self)

        newps.values = gaussian_filter1d(self.values, sigma/self.dphase_deg, mode='wrap')
        return newps

    def calc_image_extent(self):
        return (self.first_phase - 0.5*self.dphase_deg,
                  self.first_phase + (self.values.shape[1] - 0.5)*self.dphase_deg,
                  self.first_pulse - 0.5*self.dpulse,
                  self.first_pulse + (self.values.shape[0] - 0.5)*self.dpulse)

    def plot_image(self, **kwargs):
        # Plots the pulsestack as an image
        self.fig, self.ax = plt.subplots()
        extent = self.calc_image_extent()
        self.ps_image = plt.imshow(self.values, aspect='auto', origin='lower', interpolation='none', extent=extent, cmap='hot', **kwargs)
        self.cbar = plt.colorbar()
        plt.xlabel("Pulse phase (deg)")
        plt.ylabel("Pulse number")

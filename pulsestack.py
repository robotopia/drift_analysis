import copy
import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter1d

class Pulsestack:

    def __init__(self):
        self.pdvfile = None
        self.stokes  = None
        self.npulses = None
        self.nbins   = None
        self.first_pulse = None
        self.first_phase = None
        self.dpulse      = None
        self.dphase_deg  = None
        self.onpulse     = None

    def serialize(self):
        serialized = {}

        if self.pdvfile is not None:
            serialized["pdvfile"] = self.pdvfile

        if self.stokes is not None:
            serialized["stokes"] = self.stokes

        if self.npulses is not None:
            serialized["npulses"] = self.npulses

        if self.nbins is not None:
            serialized["nbins"] = self.nbins

        if self.first_pulse is not None:
            serialized["first_pulse"] = self.first_pulse

        if self.first_phase is not None:
            serialized["first_phase"] = self.first_phase

        if self.dpulse is not None:
            serialized["dpulse"] = self.dpulse

        if self.dphase_deg is not None:
            serialized["dphase_deg"] = self.dphase_deg

        if self.onpulse is not None:
            serialized["onpulse"] = list(self.onpulse)

        if self.values is not None:
            serialized["values"] = list(self.values.flatten())

        return serialized

    def unserialize(self, data):

        if "pdvfile" in data.keys():
            self.pdvfile = data["pdvfile"]
        else:
            self.pdvfile = None

        if "stokes" in data.keys():
            self.stokes = data["stokes"]
        else:
            self.stokes = None

        if "npulses" in data.keys():
            self.npulses = data["npulses"]
        else:
            self.npulses = None

        if "nbins" in data.keys():
            self.nbins = data["nbins"]
        else:
            self.nbins = None

        if "first_pulse" in data.keys():
            self.first_pulse = data["first_pulse"]
        else:
            self.first_pulse = None

        if "first_phase" in data.keys():
            self.first_phase = data["first_phase"]
        else:
            self.first_phase = None

        if "dpulse" in data.keys():
            self.dpulse = data["dpulse"]
        else:
            self.dpulse = None

        if "dphase_deg" in data.keys():
            self.dphase_deg = data["dphase_deg"]
        else:
            self.dphase_deg = None

        if "onpulse" in data.keys():
            self.onpulse = data["onpulse"]
        else:
            self.onpulse = None

        if "values" in data.keys() and self.npulses is not None and self.nbins is not None:
            self.values = np.reshape(data["values"], (self.npulses, self.nbins))
        else:
            self.values = None

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

    def set_onpulse(self, ph_lo, ph_hi):
        self.onpulse = [ph_lo, ph_hi]

    def get_pulse_bin(self, pulse, inrange=True):
        # This can return fractional (non-integer) values
        pulse_bin = (np.array(pulse) - self.first_pulse)/self.dpulse
        if inrange == True:
            if np.any(pulse_bin < 0):
                pulse_bin[pulse_bin < 0] = 0
            if np.any(pulse_bin > self.values.shape[0] - 1):
                pulse_bin[pulse_bin > self.values.shape[0] - 1] = self.values.shape[0] - 1
        return pulse_bin

    def get_phase_bin(self, phase_deg, inrange=True):
        # This can return fractional (non-integer) values
        phase_deg_bin = (np.array(phase_deg) - self.first_phase)/self.dphase_deg
        if inrange == True:
            phase_deg_bin[phase_deg_bin < 0] = 0
            phase_deg_bin[phase_deg_bin > self.values.shape[1] - 1] = self.values.shape[1] - 1
        return phase_deg_bin

    def get_pulse_from_bin(self, pulse_bin):
        return pulse_bin*self.dpulse + self.first_pulse

    def get_phase_from_bin(self, phase_bin):
        return phase_bin*self.dphase_deg + self.first_phase

    def get_pulses_array(self):
        return self.get_pulse_from_bin(np.arange(self.npulses))

    def get_phases_array(self):
        return self.get_phase_from_bin(np.arange(self.nbins))

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
        return [self.first_phase - 0.5*self.dphase_deg,
                  self.first_phase + (self.values.shape[1] - 0.5)*self.dphase_deg,
                  self.first_pulse - 0.5*self.dpulse,
                  self.first_pulse + (self.values.shape[0] - 0.5)*self.dpulse]

    def plot_image(self, **kwargs):
        # Plots the pulsestack as an image
        self.fig, self.ax = plt.subplots()
        extent = self.calc_image_extent()
        self.ps_image = plt.imshow(self.values, aspect='auto', origin='lower', interpolation='none', extent=extent, cmap='hot', **kwargs)
        self.cbar = plt.colorbar()
        plt.xlabel("Pulse phase (deg)")
        plt.ylabel("Pulse number")

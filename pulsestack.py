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
        self.complex     = None
        self.xlabel      = "Pulse phase (deg)"
        self.ylabel      = "Pulse number"
        self.vmin        = None
        self.vmax        = None

    def serialize(self):
        serialized = {}

        if self.pdvfile is not None:
            serialized["pdvfile"] = self.pdvfile

        if self.stokes is not None:
            serialized["stokes"] = self.stokes

        if self.npulses is not None:
            serialized["npulses"] = int(self.npulses)

        if self.nbins is not None:
            serialized["nbins"] = int(self.nbins)

        if self.first_pulse is not None:
            serialized["first_pulse"] = float(self.first_pulse)

        if self.first_phase is not None:
            serialized["first_phase"] = float(self.first_phase)

        if self.dpulse is not None:
            serialized["dpulse"] = float(self.dpulse)

        if self.dphase_deg is not None:
            serialized["dphase_deg"] = float(self.dphase_deg)

        if self.onpulse is not None:
            serialized["onpulse"] = list(self.onpulse)

        if self.complex is not None:
            serialized["complex"] = self.complex

        if self.xlabel is not None:
            serialized["xlabel"] = self.xlabel

        if self.ylabel is not None:
            serialized["ylabel"] = self.ylabel

        if self.values is not None:
            flattened = self.values.flatten()
            if self.complex is None or self.complex == "real":
                serialized["values"] = list(flattened)
            elif self.complex == "complex":
                serialized["values"] = {"real": list(np.real(flattened)), "imag": list(np.imag(flattened))}

        if self.vmin is not None:
            serialized["vmin"] = self.vmin

        if self.vmax is not None:
            serialized["vmax"] = self.vmax

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

        if "complex" in data.keys():
            self.complex = data["complex"]
        else:
            self.complex = "real" # Assume real for backwards compatibility

        if "xlabel" in data.keys():
            self.xlabel = data["xlabel"]
        else:
            self.xlabel = None

        if "ylabel" in data.keys():
            self.ylabel = data["ylabel"]
        else:
            self.ylabel = None

        if "values" in data.keys() and self.npulses is not None and self.nbins is not None:
            if self.complex is None or self.complex == "real":
                self.values = np.reshape(data["values"], (self.npulses, self.nbins))
            elif self.complex == "complex":
                re = np.array(data["values"]["real"])
                im = np.array(data["values"]["imag"])
                self.values = np.reshape(re + 1j*im, (self.npulses, self.nbins))
        else:
            self.values = None

        if "vmin" in data.keys():
            self.vmin = data["vmin"]

        if "vmax" in data.keys():
            self.vmax = data["vmax"]

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

        self.complex = "real"

    def set_onpulse(self, ph_lo, ph_hi):
        self.onpulse = [ph_lo, ph_hi]

    def get_pulse_bin(self, pulse, inrange=True):
        # This can return fractional (non-integer) values
        pulse_bin = (np.array(pulse) - self.first_pulse)/self.dpulse
        if inrange == True:
            if np.any(pulse_bin < 0):
                pulse_bin[pulse_bin < 0] = 0
            if np.any(pulse_bin > self.npulses - 1):
                pulse_bin[pulse_bin > self.npulses - 1] = self.npulses - 1
        return pulse_bin

    def get_phase_bin(self, phase_deg, inrange=True):
        # This can return fractional (non-integer) values
        phase_deg_bin = (np.array(phase_deg) - self.first_phase)/self.dphase_deg
        if inrange == True:
            if np.any(phase_deg_bin < 0):
                phase_deg_bin[phase_deg_bin < 0] = 0
            if np.any(phase_deg_bin > self.nbins - 1):
                phase_deg_bin[phase_deg_bin > self.nbins - 1] = self.nbins - 1
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
        if self.onpulse is not None:
            ph_lo, ph_hi = self.onpulse
            self.set_onpulse(ph_lo - phase_deg, ph_hi - phase_deg)

    def get_values(self, pulses, phases):
        '''
        Get the pulsestack values at the given (list of) pulses and phases
        '''
        pulse_bins = np.round(self.get_pulse_bin(pulses, inrange=False)).astype(int)
        phase_bins = np.round(self.get_phase_bin(phases, inrange=False)).astype(int)

        pulse_inrange = np.logical_and(pulse_bins >= 0, pulse_bins < self.npulses)
        phase_inrange = np.logical_and(phase_bins >= 0, phase_bins < self.nbins)
        inrange       = np.logical_and(pulse_inrange, phase_inrange)

        pulse_bins = pulse_bins[inrange]
        phase_bins = phase_bins[inrange]

        return np.array([self.values[pulse_bins[i], phase_bins[i]] for i in range(len(pulse_bins))])

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

        if pulse_range is not None:

            pulse_bin_range    = np.round(newps.get_pulse_bin(pulse_range)).astype(int)
            newps.values       = newps.values[pulse_bin_range[0]:pulse_bin_range[1], :]
            newps.first_pulse += pulse_bin_range[0]*newps.dpulse

        if phase_deg_range is not None:

            phase_bin_range     = np.round(newps.get_phase_bin(phase_deg_range)).astype(int)
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

    def cross_correlate_successive_pulses(self, dphase_deg=None):
        # Calculate the cross correlation via the Fourier Transform method and
        # put the result into its own "pulsestack"
        crosscorr = copy.copy(self)

        # Work out the number of phase bins from requested dphase_deg
        if dphase_deg is None:
            nbins = self.nbins
            dphase_deg = self.dphase_deg
        else:
            nbins = int(np.round((self.nbins * self.dphase_deg) / dphase_deg))
            dphase_deg = (self.nbins * self.dphase_deg) / nbins
        padding_size = nbins - self.nbins

        # Correlate each pulse with its successor
        ffted    = np.fft.fft(self.values, axis=1)
        corred   = np.conj(ffted[:-1,:]) * ffted[1:,:]

        # There are now one fewer pulses and the bin size is different
        crosscorr.npulses -= 1
        crosscorr.dphase_deg = dphase_deg
        crosscorr.nbins = nbins

        # Include padded zeros (downsampling is not supported)
        if padding_size > 0:

            nyquist_idx = (self.nbins + 1) // 2

            # If there's a Nyquist bin, zero it
            if self.nbins % 2 == 0:
                corred[:,nyquist_idx] = 0. + 0.j

            # Insert the zero padding at the Nyquist frequency
            corred = np.hstack((corred[:,:nyquist_idx], np.zeros((crosscorr.npulses, padding_size)), corred[:,nyquist_idx:]))

        # Go back to the time (=lag) domain
        crosscorr.values = np.real(np.fft.ifft(corred, axis=1))

        # Put zero lag in the centre
        shift = nbins//2
        crosscorr.values = np.roll(crosscorr.values, shift, axis=1)
        crosscorr.first_phase = -shift*dphase_deg
        crosscorr.xlabel = "Correlation lag (deg)"

        return crosscorr

    def auto_correlate_pulses(self):
        # Calculate the auto correlation via the Fourier Transform and
        # put the result into its own "pulsestack"
        autocorr = copy.copy(self)

        rffted   = np.fft.rfft(self.values, axis=1)
        corred   = np.conj(rffted) * rffted
        autocorr.values = np.fft.irfft(corred, axis=1)

        # Put zero lag in the centre
        shift = self.nbins//2
        autocorr.values = np.roll(autocorr.values, shift, axis=1)
        autocorr.first_phase = -shift*self.dphase_deg
        autocorr.xlabel = "Correlation lag (deg)"

        return autocorr

    def LRFS(self, pulse_range=None, phase_deg_range=None, window=None):

        lrfs = self.crop(pulse_range=pulse_range, phase_deg_range=phase_deg_range, inplace=False)

        if window == "hamming":
            lrfs.values = lrfs.values * np.hamming(lrfs.npulses)[:,np.newaxis]

        lrfs.values = np.fft.rfft(lrfs.values, axis=0)[1:,:]
        lrfs.complex = "complex"
        freqs = np.fft.rfftfreq(lrfs.npulses, lrfs.dpulse)
        df    = freqs[1] - freqs[0]
        lrfs.npulses  = lrfs.values.shape[0]
        lrfs.dpulse   = df
        lrfs.first_pulse = df
        lrfs.xlabel   = self.xlabel
        lrfs.ylabel   = "Frequency (cycles/$P$)"
        return lrfs

    def TDFS(self, pulse_range=None, phase_deg_range=None, window=None):

        tdfs = self.crop(pulse_range=pulse_range, phase_deg_range=phase_deg_range, inplace=False)

        if window == "hamming":
            tdfs.values = tdfs.values * np.hamming(tdfs.npulses)[:,np.newaxis]

        tdfs.values = np.fft.rfft(tdfs.values, axis=0)[1:,:]
        tdfs.values = np.fft.fft(tdfs.values, axis=1)
        tdfs.complex = "complex"

        freqs2 = np.fft.fftfreq(tdfs.nbins, tdfs.dphase_deg/360.0)
        freqs3 = np.fft.rfftfreq(tdfs.npulses, tdfs.dpulse)

        df2    = freqs2[1] - freqs2[0]
        df3    = freqs3[1] - freqs3[0]

        freqs2 = np.fft.fftshift(freqs2)
        tdfs.values = np.fft.fftshift(tdfs.values, axes=1)

        tdfs.npulses     = tdfs.values.shape[0]
        tdfs.dphase_deg  = df2
        tdfs.dpulse      = df3
        tdfs.first_phase = freqs2[0]
        tdfs.first_pulse = df3
        tdfs.xlabel      = "Frequency (cycles/$P$)"
        tdfs.ylabel      = "Frequency (cycles/$P$)"

        return tdfs

    def calc_mean_energies(self, on_and_off_pulse=True):
        if on_and_off_pulse == True:
            bin_lo, bin_hi = self.get_phase_bin(self.onpulse)
            bin_lo = int(np.floor(bin_lo))
            bin_hi = int(np.ceil(bin_hi))

            onpulse  = self.values[:,bin_lo:bin_hi]
            offpulse = np.hstack((self.values[:,:bin_lo], self.values[:,bin_hi:]))
            if offpulse.shape[1] > bin_hi - bin_lo:
                offpulse = offpulse[:,:bin_hi-bin_lo]
            else:
                print("warning: offpulse region has fewer bins than onpulse region")

            onpulse_energies = np.mean(onpulse, axis=1)
            offpulse_energies = np.mean(offpulse, axis=1)

            return onpulse_energies, offpulse_energies
        else:
            return np.mean(self.values, axis=1)

    def plot_image(self, ax, colorbar=True, **kwargs):
        # Plots the pulsestack as an image
        extent = self.calc_image_extent()
        if self.complex == "real":
            self.ps_image = ax.imshow(self.values, aspect='auto', origin='lower', interpolation='none', extent=extent, cmap='hot', vmin=self.vmin, vmax=self.vmax, **kwargs)
        else:
            self.ps_image = ax.imshow(np.abs(self.values), aspect='auto', origin='lower', interpolation='none', extent=extent, cmap='hot', vmin=self.vmin, vmax=self.vmax, **kwargs)
        if colorbar:
            self.cbar = plt.colorbar(mappable=self.ps_image, ax=ax)
        ax.set_xlabel("Pulse phase (deg)" if self.xlabel is None else self.xlabel)
        ax.set_ylabel("Pulse number" if self.ylabel is None else self.ylabel)

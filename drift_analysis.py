__version__ = "0.9.1"

import sys
import copy

import numpy as np
from numpy.polynomial.polynomial import polyfit, polyval

from mpl_toolkits import mplot3d
import matplotlib.pyplot as plt

from scipy.interpolate import interp1d
from scipy.ndimage import gaussian_filter1d

import tkinter
import tkinter.filedialog
import tkinter.simpledialog

import json
import bisect
import pulsestack

class Subpulses:

    def __init__(self):

        self.delete_all_subpulses() # Apart from deleting, this also initialises everything

    def add_subpulses(self, phases, pulses, widths=None, driftbands=None):
        if len(phases) != len(pulses):
            print("Lengths of phases and pulses don't match. No subpulses added.")
            return

        nnewsubpulses = len(phases)

        if widths is None:
            widths = np.full((nnewsubpulses), np.nan)
        elif np.isscalar(widths):
            widths = np.full((nnewsubpulses), widths)
        elif len(widths) != nnewsubpulses:
            print("Length of widths doesn't match phases and pulses. No subpulses added.")
            return

        if driftbands is None:
            driftbands = np.full((nnewsubpulses), np.nan)
        elif np.isscalar(driftbands):
            driftbands = np.full((nnewsubpulses), driftbands)
        elif len(driftbands) != nnewsubpulses:
            print("Length of driftbands doesn't match phases and pulses. No subpulses added.")
            return

        newsubpulses = np.transpose([phases, pulses, widths, driftbands])
        self.data = np.vstack((self.data, newsubpulses))

        # Keep track of the number of subpulses
        self.nsubpulses = self.data.shape[0]

    def delete_subpulses(self, delete_idxs):
        self.data = np.delete(self.data, delete_idxs, axis=0)
        self.nsubpulses = self.data.shape[0]

    def delete_all_subpulses(self):
        self.nsubpulses  = 0
        self.nproperties = 4

        self.data = np.full((self.nsubpulses, self.nproperties), np.nan)

    def get_phases(self, subset=None):
        if subset is None:
            return self.data[:,0]
        else:
            return self.data[subset, 0]

    def get_pulses(self, subset=None):
        if subset is None:
            return self.data[:,1]
        else:
            return self.data[subset, 1]

    def get_widths(self, subset=None):
        if subset is None:
            return self.data[:,2]
        else:
            return self.data[subset, 2]

    def get_driftbands(self, subset=None):
        if subset is None:
            return self.data[:,3]
        else:
            return self.data[subset, 3]

    def get_positions(self):
        '''
        Returns an Nx2 numpy array of subpulse positions (phase, pulse)
        '''
        return self.data[:,:2]

    def set_phases(self, phases, subset=None):
        if subset is None:
            self.data[:,0] = phases
        else:
            self.data[subset,0] = phases

    def set_pulses(self, pulses, subset=None):
        if subset is None:
            self.data[:,1] = pulses
        else:
            self.data[subset,1] = pulses

    def set_widths(self, widths, subset=None):
        if subset is None:
            self.data[:,2] = widths
        else:
            self.data[subset,2] = widths

    def set_driftbands(self, driftbands, subset=None):
        if subset is None:
            self.data[:,3] = driftbands
        else:
            self.data[subset,3] = driftbands

    def shift_all_subpulses(self, dphase=None, dpulse=None):
        if dphase is not None:
            self.data[:,0] += dphase

        if dpulse is not None:
            self.data[:,1] += dpulse

    def calc_drift(self, subpulse_idxs):
        '''
        Fits a line to the given set of subpulses
        Returns the driftrate and y-intercept of the line at the fiducial point
        '''
        if len(subpulse_idxs) < 2:
            print("Not enough subpulses selected. No drift line fitted")
            return

        phases = self.get_phases()
        pulses = self.get_pulses()

        linear_fit = polyfit(phases, pulses, 1)
        driftrate = 1/linear_fit[1]
        yintercept = linear_fit[0]

        return driftrate, yintercept

    def in_pulse_range(self, pulse_range):
        '''
        Returns a logical mask for those subpulses within the specified range
        '''
        p  = self.get_pulses()
        p_lo, p_hi = pulse_range
        return np.logical_and(p >= p_lo, p <= p_hi)

    def assign_quadratic_driftbands_to_subpulses(self, model_fit, idx2pulse_func):
        '''
        model_fit - an object of ModelFit
        This will classify each subpulse in the appropriate pulse range
        into a driftband according to the given model.
        '''
        # Get only those subpulses within the valid range of the quadratic fit
        pulse_range = idx2pulse_func(np.array(model_fit.get_pulse_idx_bounds()))
        subset = self.in_pulse_range(pulse_range)

        ph = self.get_phases(subset=subset)
        p  = self.get_pulses(subset=subset)
        d = model_fit.get_nearest_driftband(p, ph)

        self.set_driftbands(d, subset=subset)

        # For good measure, return the subset of affected subpulse
        return subset

class DriftSequences:
    def __init__(self):
        self.boundaries = []  # Contains indexes of pulse numbers preceding the boundary

    def number_of_sequences(self):
        return len(self.boundaries) + 1

    def has_boundary(self, pulse_idx):
        if pulse_idx in self.boundaries:
            return True
        else:
            return False

    def add_boundary(self, pulse_idx):
        if not self.has_boundary(pulse_idx):
            bisect.insort(self.boundaries, pulse_idx)

    def delete_boundaries(self, boundary_idxs):
        self.boundaries = [v for i, v in enumerate(self.boundaries) if i not in boundary_idxs]

    def get_bounding_pulse_idxs(self, sequence_idx, npulses):
        '''
        Returns the first and last pulses indexes in this sequence
        '''
        if len(self.boundaries) == 0:
            return 0, npulses - 1

        if sequence_idx < 0:
            sequence_idx = len(self.boundaries) + 1 + sequence_idx

        if sequence_idx == 0:
            first_idx = 0
        else:
            first_idx = self.boundaries[sequence_idx - 1] + 1

        if sequence_idx == len(self.boundaries):
            last_idx = npulses - 1
        else:
            last_idx = self.boundaries[sequence_idx]

        return first_idx, last_idx

    def is_pulse_in_sequence(self, sequence_idx, pulse_idx, npulses):
        first_idx, last_idx = self.get_bounding_pulse_idxs(sequence_idx, npulses)

        if pulse_idx >= first_idx and pulse_idx <= last_idx:
            return True
        else:
            return False

    def get_pulse_mid_idxs(self, boundary_idxs=None):
        if boundary_idxs is None:
            return np.array(self.boundaries) + 0.5
        else:
            return np.array(self.boundaries)[boundary_idxs] + 0.5

    def get_sequence_number(self, pulse_idx, npulses):
        # There are lots of corner cases!
        # Remember, the first boundary sits between sequences 0 and 1,
        # and the last sits between sequences n and n+1, where
        # n = len(self.boundaries) - 1
        # All the "0.5"s around the place is to make this function give sensible
        # results when pulse_idx is a fractional value
        if pulse_idx < -0.5 or pulse_idx >= npulses - 0.5:
            sequence_number = None
        elif self.number_of_sequences() == 1:
            sequence_number = 0
        elif pulse_idx <= self.boundaries[0] + 0.5:
            sequence_number = 0
        elif len(self.boundaries) == 0:
            sequence_number = 0
        elif pulse_idx > self.boundaries[-1] + 0.5:
            sequence_number = len(self.boundaries)
        else:
            is_no_more_than = pulse_idx <= np.array(self.boundaries) + 0.5
            sequence_number = np.argwhere(is_no_more_than)[0][0]

        return sequence_number

class ModelFit(pulsestack.Pulsestack):
    def __init__(self):
        # For the meaning of these variables, refer to McSweeney et al. (2017)
        self.parameters = None

        self.first_pulse_idx = None
        self.last_pulse_idx  = None

        # A dictionary for keeping track of line plot objects
        # Dictioney keys are intended to be driftband numbers
        self.driftband_plts = {}

    def set_pulse_bounds(self, first_pulse_idx, last_pulse_idx):
        self.first_pulse_idx = first_pulse_idx
        self.last_pulse_idx  = last_pulse_idx

    def get_pulse_idx_bounds(self):
        return [self.first_pulse_idx, self.last_pulse_idx]

    def least_squares_fit_to_subpulses(self, phases, pulses, driftbands):
        # phases, pulses, and driftbands must be vectors with the same length
        npoints = len(phases)
        if len(pulses) != npoints or len(driftbands)!= npoints:
            print("phases, pulses, driftbands have lengths {}, {}, and {}, but they must be the same".format(len(phases), len(pulses), len(driftbands)))
            return

        # Form the matrices for least squares fitting
        # The model parameters are:
        #   self.parameters = [a1, a2, a3, a4]
        # where
        #   ph = a1*p^2 + a2*p + a3 + a4*d
        ph = np.array(phases)
        p  = np.array(pulses)
        d  = np.array(driftbands)

        Y  = ph
        X  = np.array([p**2, p, np.ones(p.shape), d]).T

        XTX = X.T @ X
        XTY = X.T @ Y
        self.parameters = np.linalg.pinv(XTX) @ XTY

    def serialize(self):
        return [list(self.parameters), self.first_pulse_idx, self.last_pulse_idx]

    def unserialize(self, data):
        self.parameters      = data[0]
        self.first_pulse_idx = data[1]
        self.last_pulse_idx  = data[2]

    def calc_phase(self, pulse, driftband):
        a1, a2, a3, a4 = self.parameters
        p = pulse
        d = driftband
        return a1*p**2 + a2*p + a3 + a4*d

    def calc_driftrate(self, pulse):
        a1, a2, _, _ = self.parameters
        return 2*a1*pulse + a2

    def calc_driftrate_derivative(self, pulse):
        '''
        Returns the driftrate derivative w.r.t. pulse
        '''
        a1, _, _, _ = self.parameters
        return 2*a1

    def get_nearest_driftband(self, pulse, phase):
        a1, a2, a3, a4 = self.parameters
        driftband = np.round((phase - a1*pulse**2 - a2*pulse - a3)/a4)
        return driftband

    def calc_P2(self):
        return self.parameters[3]

    def get_driftband_range(self, phlim, idx2pulse_func):
        a1, a2, a3, a4 = self.parameters

        # Figure out if drift rate is positive or negative (at the first pulse)
        first_p = idx2pulse_func(self.first_pulse_idx)
        last_p  = idx2pulse_func(self.last_pulse_idx)

        if self.calc_driftrate(first_p) < 0:
            first_ph = phlim[0]
        else:
            first_ph = phlim[1]

        if self.calc_driftrate(last_p) < 0:
            last_ph  = phlim[1]
        else:
            last_ph  = phlim[0]

        first_d = np.ceil((first_ph - a1*first_p**2 - a2*first_p - a3)/a4)
        last_d  = np.floor((last_ph - a1*last_p**2 - a2*last_p - a3)/a4)

        return int(first_d), int(last_d)

    def plot_driftband(self, ax, driftband, idx2pulse_func, phlim=None, **kwargs):
        '''
        driftband: driftband number to plot
        phlim: phase limits between which to draw the driftband
        '''
        d = driftband

        p = idx2pulse_func(np.arange(self.first_pulse_idx, self.last_pulse_idx+1))
        ph = self.calc_phase(p, d)

        if phlim is not None:
            in_phase_range = np.logical_and(ph >= phlim[0], ph <= phlim[1])
            p  = p[in_phase_range]
            ph = ph[in_phase_range]

        # If there's no part of this driftband that falls inside the phase range
        # (and pulse range), then do nothing
        if len(p) == 0:
            if d in self.driftband_plts.keys():
                self.driftband_plts.pop(d)
            return

        if d in self.driftband_plts.keys():
            self.driftband_plts[d][0].set_data(ph, p)
        else:
            self.driftband_plts[d] = ax.plot(ph, p, **kwargs)

    def plot_all_driftbands(self, ax, phlim, idx2pulse_func, **kwargs):
        first_d, last_d = self.get_driftband_range(phlim, idx2pulse_func)
        for d in range(first_d, last_d+1):
            self.plot_driftband(ax, d, idx2pulse_func, phlim=phlim, **kwargs)

    def clear_all_plots(self):
        for d in self.driftband_plts:
            self.driftband_plts[d][0].set_data([], [])
        self.driftband_plts = {}

class DriftAnalysis(pulsestack.Pulsestack):
    def __init__(self):
        self.fig                       = None
        self.ax                        = None
        self.subpulses                 = Subpulses()
        self.subpulses_plt             = None
        self.subpulses_fmt             = 'gx'
        self.maxima_threshold          = 0.0
        self.drift_sequences           = DriftSequences()
        self.dm_boundary_plt           = None
        self.jsonfile                  = None
        self.candidate_quadratic_model = ModelFit()
        self.onpulse                   = None
        self.model_fits            = {}  # Keys = drift sequence numbers
        self.quadratic_visible         = True

    def set_onpulse(self, phlim):
        if phlim == []:
            self.onpulse = None
        else:
            self.onpulse = phlim

    def serialize_onpulse(self):
        if self.onpulse is None:
            return []
        else:
            return self.onpulse

    def get_local_maxima(self, maxima_threshold=None):
        if maxima_threshold is None:
            maxima_threshold = self.maxima_threshold
        else:
            self.maxima_threshold = maxima_threshold

        is_bigger_than_left  = self.values[:,1:-1] >= self.values[:,:-2]
        is_bigger_than_right = self.values[:,1:-1] >= self.values[:,2:]
        is_local_max = np.logical_and(is_bigger_than_left, is_bigger_than_right)

        if maxima_threshold is not None:
            is_local_max = np.logical_and(is_local_max, self.values[:,1:-1] > maxima_threshold)

        self.max_locations = np.array(np.where(is_local_max)).astype(float)

        # Add one to phase (bin) locations because of previous splicing
        self.max_locations[1,:] += 1

        # Convert locations to data coordinates (pulse and phase)
        self.max_locations[0,:] = self.max_locations[0,:]*self.dpulse + self.first_pulse
        self.max_locations[1,:] = self.max_locations[1,:]*self.dphase_deg + self.first_phase

    def save_json(self, jsonfile=None):

        if jsonfile is None:
            jsonfile = self.jsonfile

        # If jsonfile is STILL unspecified, open file dialog box
        if jsonfile is None:
            root = tkinter.Tk()
            root.withdraw()
            jsonfile = tkinter.filedialog.asksaveasfilename(filetypes=(("All files", "*.*"),))

        # And if THAT still didn't produce a filename, cancel
        if not jsonfile:
            return

        drift_dict = {
                "version":             __version__,
                "pdvfile":             self.pdvfile,
                "stokes":              self.stokes,
                "npulses":             self.npulses,
                "nbins":               self.nbins,
                "first_pulse":         self.first_pulse,
                "first_phase":         self.first_phase,
                "dpulse":              self.dpulse,
                "dphase_deg":          self.dphase_deg,
                "onpulse":             self.serialize_onpulse(),

                "subpulses_pulse":     list(self.subpulses.get_pulses()),
                "subpulses_phase":     list(self.subpulses.get_phases()),
                "subpulses_width":     list(self.subpulses.get_widths()),
                "subpulses_driftband": list(self.subpulses.get_driftbands()),

                "quadratic_fits":      [[int(i), self.model_fits[i].serialize()] for i in self.model_fits],

                "maxima_threshold":    self.maxima_threshold,
                "subpulses_fmt":       self.subpulses_fmt,
                "drift_mode_boundaries": self.drift_sequences.boundaries,
                "values":              list(self.values.flatten())
                }

        with open(jsonfile, "w") as f:
            json.dump(drift_dict, f)

        self.jsonfile = jsonfile

    def load_json(self, jsonfile=None):

        if jsonfile is None:
            jsonfile = self.jsonfile

        with open(jsonfile, "r") as f:
            drift_dict = json.load(f)

        if drift_dict["version"] != __version__:
            print("Warning: version mismatch, File = {}, Software = {}".format(drift_dict["version"], __version__))

        self.pdvfile          = drift_dict["pdvfile"]
        self.stokes           = drift_dict["stokes"]
        self.npulses          = drift_dict["npulses"]
        self.nbins            = drift_dict["nbins"]
        self.first_pulse      = drift_dict["first_pulse"]
        self.first_phase      = drift_dict["first_phase"]
        self.dpulse           = drift_dict["dpulse"]
        self.dphase_deg       = drift_dict["dphase_deg"]
        self.set_onpulse(drift_dict["onpulse"])

        subpulses_pulse       = drift_dict["subpulses_pulse"]
        subpulses_phase       = drift_dict["subpulses_phase"]
        subpulses_width       = drift_dict["subpulses_width"]
        subpulses_driftband   = drift_dict["subpulses_driftband"]

        self.subpulses.add_subpulses(subpulses_phase, subpulses_pulse, subpulses_width, subpulses_driftband)

        for item in drift_dict["quadratic_fits"]:
            self.model_fits[item[0]] = ModelFit()
            self.model_fits[item[0]].unserialize(item[1])

        self.maxima_threshold = drift_dict["maxima_threshold"]
        self.subpulses_fmt    = drift_dict["subpulses_fmt"]
        self.drift_sequences.boundaries = drift_dict["drift_mode_boundaries"]
        self.values           = np.reshape(drift_dict["values"], (self.npulses, self.nbins))

        self.jsonfile = jsonfile

    def plot_subpulses(self, **kwargs):
        if self.subpulses_plt is None:
            self.subpulses_plt, = self.ax.plot(self.subpulses.get_phases(), self.subpulses.get_pulses(), self.subpulses_fmt, **kwargs)
        else:
            self.subpulses_plt.set_data(self.subpulses.get_phases(), self.subpulses.get_pulses())

    def plot_drift_mode_boundaries(self):
        xlo = self.first_phase
        xhi = self.first_phase + self.nbins*self.dphase_deg

        ys = self.get_pulse_from_bin(self.drift_sequences.get_pulse_mid_idxs())
        if self.dm_boundary_plt is not None:
            segments = np.array([[[xlo, y], [xhi, y]] for y in ys])
            self.dm_boundary_plt.set_segments(segments)
        else:
            self.dm_boundary_plt = self.ax.hlines(ys, xlo, xhi, colors=["k"], linestyles='dashed')

    def plot_all_model_fits(self):
        for i in self.model_fits:
            if self.onpulse is None:
                phlim = self.get_phase_from_bin(np.array([0, self.nbins-1]))
            else:
                phlim = self.onpulse

            self.model_fits[i].plot_all_driftbands(self.ax, phlim, self.get_pulse_from_bin, color='k')

    def unplot_all_model_fits(self):
        for i in self.model_fits:
            self.model_fits[i].clear_all_plots()

    def cross_correlate_successive_pulses(self, do_shift=True):
        # Calculate the cross correlation via the Fourier Transform
        rffted    = np.fft.rfft(self.values, axis=1)
        corred    = np.conj(rffted[:-1,:]) * rffted[1:,:]
        shift     = self.nbins//2
        crosscorr = np.fft.irfft(corred, axis=1)
        # Calculate the lags and put zero lag in the centre
        # Even though there is a np.fftshift function for this, I'm doing it
        # "by hand" so that I don't have to worry by how much it gets shifted depending on
        # whether it's an odd or even number of bins, as it doesn't really matter where
        # the "Nyquist" bin ends up.
        lags      = np.arange(crosscorr.shape[1])*self.dphase_deg
        if do_shift:
            crosscorr = np.roll(crosscorr, shift, axis=1)
            lags     -= shift*self.dphase_deg

        return crosscorr, lags, shift

    def auto_correlate_pulses(self, do_shift=True, set_DC_value=None):
        # Calculate the auto correlation via the Fourier Transform
        rffted   = np.fft.rfft(self.values, axis=1)
        corred   = np.conj(rffted) * rffted
        shift    = self.nbins//2
        autocorr = np.fft.irfft(corred, axis=1)
        lags     = np.arange(autocorr.shape[1])*self.dphase_deg

        if set_DC_value is not None:
            autocorr[:,0] = set_DC_value

        if do_shift:
            autocorr = np.roll(autocorr, shift, axis=1)
            lags    -= shift*self.dphase_deg

        return autocorr, lags, shift

    def driftrates_via_cross_correlation(self, approx_P2, smoothing_kernel_size=None):
        '''
        This function calculates the drift rate for each pulse in the following way:
        It first calculates the cross correlation of each pulse with its successor.
        Then it smooths with a gaussian kernel, and finds the position of the peak
        (via linear interpolation) that lies in the range
        [-0.5*approx_P2:0.5*approx_P2]
        The default smoothing kernel size is 0.1*approx_P2
        '''
        crosscorr, lags, shift = self.cross_correlate_successive_pulses()

        lo_idx = shift - int(0.5*approx_P2/self.dphase_deg)
        hi_idx = shift + int(0.5*approx_P2/self.dphase_deg) + 1

        if smoothing_kernel_size is None:
            smoothing_kernel_size = 0.1*approx_P2

        smoothed = gaussian_filter1d(crosscorr[:,lo_idx:hi_idx], smoothing_kernel_size/self.dphase_deg, mode='wrap')

        max_idxs = np.argmax(smoothed, axis=1)

        # Let p1, p2, p3 be the values before, at, and after the peak respectively.
        # Then linear interpolation of the differences gives a root at interpolated bin number
        # relative to the peak bin
        # (p2-p1)/((p2-p1)-(p3-p2)) + 0.5
        #   = (p2-p1)/(2*p2 - p1 - p3) + 0.5
        interpolated_idxs = np.array([(smoothed[p,max_idxs[p]] - smoothed[p,max_idxs[p]-1])/(2*smoothed[p,max_idxs[p]] - smoothed[p,max_idxs[p]-1] - smoothed[p,max_idxs[p]+1]) + 0.5 for p in range(len(max_idxs))])

        # Now just convert the interpolated bin numbers into absolute phases
        driftrates = (interpolated_idxs + max_idxs + lo_idx - shift)*self.dphase_deg
        return driftrates

class DriftAnalysisInteractivePlot(DriftAnalysis):
    def __init__(self):
        super().__init__()
        self.mode            = "default"
        self.selected        = None
        self.selected_plt    = None

        self.smoothed_ps  = None
        self.show_smooth  = False

    def deselect(self):
        self.selected = None
        if self.selected_plt is not None:
            self.selected_plt.set_data([], [])

    def on_button_press_event(self, event):

        ##############################################
        # Interpret mouse clicks for different modes #
        ##############################################

        if self.mode == "delete_subpulse":
            idx, dist = self.closest_subpulse(event.x, event.y)

            # Deselect if mouse click is more than 10 pixels away from the nearest point
            if dist > 10:
                self.deselect()
                self.fig.canvas.draw()
                return

            self.selected = idx

            if self.selected_plt is None:
                self.selected_plt, = self.ax.plot([self.subpulses.get_phases()[self.selected]], [self.subpulses.get_pulses()[self.selected]], 'wo')
            else:
                self.selected_plt.set_data([self.subpulses.get_phases()[self.selected]], [self.subpulses.get_pulses()[self.selected]])
            
            self.fig.canvas.draw()

        elif self.mode == "delete_drift_mode_boundary":
            idx, dist = self.closest_drift_mode_boundary(event.y)

            if dist > 10: # i.e. if mouse click is more than 10 pixels away from the nearest point
                self.selected = None
            else:
                self.selected = idx

            if self.selected is not None:
                y = self.get_pulse_from_bin(self.drift_sequences.get_pulse_mid_idxs([self.selected]))
                if self.selected_plt is None:
                    self.selected_plt = self.ax.axhline(y, color='w')
                else:
                    self.selected_plt.set_data([0, 1], [y, y])
            else:
                self.deselect()
            
            if self.selected_plt is not None:
                self.fig.canvas.draw()

        elif self.mode == "add_subpulse":
            # Snap to nearest pulse, but let phase be continuous
            pulse_bin = np.round(self.get_pulse_bin(event.ydata))
            pulse = self.first_pulse + pulse_bin*self.dpulse
            phase = event.xdata
            self.selected = np.array([pulse, phase])

            if self.selected_plt is None:
                self.selected_plt, = self.ax.plot([self.selected[1]], [self.selected[0]], 'wo')
            else:
                self.selected_plt.set_data(np.flip(self.selected))

            self.fig.canvas.draw()

        elif self.mode == "add_drift_mode_boundary":
            # Snap to nearest "previous" pulse
            pulse_idx = int(np.floor(self.get_pulse_bin(event.ydata) - 0.5))

            # Don't let the user add a drift mode boundary that already exists
            if self.drift_sequences.has_boundary(pulse_idx):
                return

            self.selected = pulse_idx
            y = self.get_pulse_from_bin(pulse_idx + 0.5)

            if self.selected_plt is None:
                self.selected_plt = self.ax.axhline(y, color="w")
            else:
                self.selected_plt.set_data([[0, 1], [y, y]])

            self.fig.canvas.draw()

        elif self.mode == "set_threshold":
            if event.inaxes == self.cbar.ax:
                self.threshold_line.set_data([0, 1], [event.ydata, event.ydata])
                if self.show_smooth == False:
                    self.get_local_maxima(maxima_threshold=event.ydata)
                else:
                    self.smoothed_ps.get_local_maxima(maxima_threshold=event.ydata)
                    self.maxima_threshold = self.smoothed_ps.maxima_threshold
                    self.max_locations = self.smoothed_ps.max_locations
                self.subpulses_plt.set_data(self.max_locations[1,:], self.max_locations[0,:])
                self.fig.canvas.draw()

        elif self.mode == "set_fiducial":
            if event.inaxes == self.ax:
                # Set the fiducial phase for this pulsestack
                self.set_fiducial_phase(event.xdata)

                # If necessary, also set the same fiducial phase for the smoothed pulsestack
                if self.smoothed_ps is not None:
                    self.smoothed_ps.set_fiducial_phase(event.xdata)

                # Adjust all the maxima points
                if self.subpulses.nsubpulses > 0:
                    self.subpulses.shift_all_subpulses(dphase=-event.xdata)

                # Replot everything
                current_xlim = self.ax.get_xlim()
                current_ylim = self.ax.get_ylim()
                self.ax.set_xlim(current_xlim - event.xdata)
                self.ax.set_ylim(current_ylim)
                self.ps_image.set_extent(self.calc_image_extent())
                self.plot_subpulses()

                # Add the * to the window title
                if self.jsonfile is not None:
                    self.fig.canvas.manager.set_window_title(self.jsonfile + "*")

                # Go back to default mode
                self.set_default_mode()

        elif self.mode == "set_onpulse_leading":
            if event.inaxes == self.ax:
                self.ph_lo = event.xdata
                self.ax.set_title("Click on the pulsestack to set the trailing edge of the on-pulse region")
                self.fig.canvas.draw()
                self.mode = "set_onpulse_trailing"

        elif self.mode == "set_onpulse_trailing":
            if event.inaxes == self.ax:
                self.ph_hi = event.xdata
                self.set_onpulse([self.ph_lo, self.ph_hi])

                # Add the * to the window title
                if self.jsonfile is not None:
                    self.fig.canvas.manager.set_window_title(self.jsonfile + "*")

                self.set_default_mode()

        elif self.mode == "zoom_drift_sequence" or self.mode == "improve_model_fits":
            if event.inaxes == self.ax:
                pulse_idx = self.get_pulse_bin(event.ydata, inrange=False)
                self.selected = self.drift_sequences.get_sequence_number(pulse_idx, self.npulses)
                if self.selected is not None:

                    # If in '#' mode, then only can select sequences with quadratic driftband models
                    if self.mode == "improve_model_fits" and self.selected not in self.model_fits.keys():
                        return

                    first_idx, last_idx = self.drift_sequences.get_bounding_pulse_idxs(self.selected, self.npulses)
                    self.ax.set_title("Select a drift sequence by clicking on the pulsestack.\n({}, {}) - Press enter to confirm, esc to cancel.".format(first_idx, last_idx))
                else:
                    self.ax.set_title("Select a drift sequence by clicking on the pulsestack.\nPress enter to confirm, esc to cancel.")
                self.fig.canvas.draw()

        elif self.mode == "model_fit":

            subpulse_idx, dist = self.closest_subpulse(event.x, event.y)
            pulse = self.subpulses.get_pulses()[subpulse_idx]
            pulse_idx = self.get_pulse_bin(pulse, inrange=False)

            # Deselect if mouse click is more than 10 pixels away from the nearest point
            if dist > 10:
                self.deselect()
                self.fig.canvas.draw()
                return

            # If a drift sequence has already been selected, only let subpulses in the same
            # sequence be selected
            if self.drift_sequence_selected is not None:
                if not self.drift_sequences.is_pulse_in_sequence(self.drift_sequence_selected, pulse_idx, self.npulses):
                    self.deselect()
                    self.fig.canvas.draw()
                    return

            # All's well, so select the subpulse
            self.selected = subpulse_idx

            if self.selected_plt is None:
                self.selected_plt, = self.ax.plot([self.subpulses.get_phases()[self.selected]], [self.subpulses.get_pulses()[self.selected]], 'wo')
            else:
                self.selected_plt.set_data([self.subpulses.get_phases()[self.selected]], [self.subpulses.get_pulses()[self.selected]])
            
            self.fig.canvas.draw()


    def set_default_mode(self):
        self.ax.set_title("Press (capital) 'H' for command list")
        self.fig.canvas.draw()
        self.mode = "default"

    def on_key_press_event(self, event):

        ############################
        # When in the DEFAULT MODE #
        ############################

        if self.mode == "default":

            if event.key == "H":
                print("Key   Description")
                print("----------------------------------------------")
                print("[Standard Matplotlib interface]")
                print("h     Go 'home' (default view)")
                print("f     Make window full screen")
                print("s     Save plot")
                print("l     Toggle y-axis logarithmic")
                print("L     Toggle x-axis logarithmic")
                print("q     Quit")
                print("[Drift analysis]")
                print("H     Prints this help")
                print("j     Save analysis to (json) file")
                print("J     'Save as' to (json) file")
                print("^     Set subpulses to local maxima")
                print("S     Toggle pulsestack smoothed with Gaussian filter")
                print("F     Set fiducial point")
                print("O     Set on-pulse region")
                print("C     Crop pulsestack to current visible image")
                print(".     Add a subpulse")
                print(">     Delete a subpulse")
                print("P     Plot the profile of the current view")
                print("T     Plot the LRFS of the current view")
                print("/     Add a drift mode boundary")
                print("?     Delete a drift mode boundary")
                print("v     Toggle visibility of plot feature")
                print("z     Zoom to selected drift sequence")
                print("d     Plot the cross-correlation of pulses with their successor")
                print("A     Plot the auto-correlation of each pulse")
                #print("r     Calculate drift rates from cross correlations")
                print("@     Perform quadratic fitting via subpulse selection (McSweeney et al, 2017)")
                print("#     Use all subpulses in sequence to improve quadratic fit")
                print("$     Plot the drift rate of the quadratic fits against pulse number")
                print("%     3D plot of drift rate (d) vs d-dot vs pulse number")

            elif event.key == "j":
                self.save_json()

                if self.fig is not None and self.jsonfile is not None:
                    self.fig.canvas.manager.set_window_title(self.jsonfile)


            elif event.key == "J":
                old_jsonfile = self.jsonfile
                self.jsonfile = None
                self.save_json()
                if self.jsonfile is None:
                    self.jsonfile = old_jsonfile
                if self.fig is not None:
                    self.fig.canvas.manager.set_window_title(self.jsonfile)

            elif event.key == "S":
                if self.smoothed_ps is None:
                    self.show_smooth = False

                if self.show_smooth == False:
                    root = tkinter.Tk()
                    root.withdraw()
                    sigma = tkinter.simpledialog.askfloat("Smoothing kernel", "Input Gaussian kernel size (deg)", parent=root)
                    if sigma:
                        self.smoothed_ps = self.smooth_with_gaussian(sigma, inplace=False)
                        self.ps_image.set_data(self.smoothed_ps.values)
                        self.show_smooth = True
                        # Update the colorbar
                        self.cbar.update_normal(self.ps_image)
                        self.fig.canvas.draw()

                else:
                    self.ps_image.set_data(self.values)
                    self.show_smooth = False
                    # Update the colorbar
                    self.cbar.update_normal(self.ps_image)
                    self.fig.canvas.draw()

            elif event.key == "^":
                self.ax.set_title("Set threshold on colorbar. Press enter when done, esc to cancel.")
                self.old_maxima_threshold = self.maxima_threshold # Save value in case they cancel
                if self.show_smooth == True:
                    self.smoothed_ps.get_local_maxima(maxima_threshold=self.smoothed_ps.maxima_threshold)
                    self.max_locations = self.smoothed_ps.max_locations
                else:
                    self.get_local_maxima()
                self.subpulses_plt.set_data(self.max_locations[1,:], self.max_locations[0,:])
                self.threshold_line = self.cbar.ax.axhline(self.maxima_threshold, color='g')
                self.fig.canvas.draw()
                self.mode = "set_threshold"

            elif event.key == "F":
                self.ax.set_title("Click on the pulsestack to set a new fiducial point")
                self.fig.canvas.draw()
                self.mode = "set_fiducial"

            elif event.key == "O":
                self.ax.set_title("Click on the pulsestack to set the leading edge of the on-pulse region")
                self.fig.canvas.draw()
                self.mode = "set_onpulse_leading"

            elif event.key == "C":
                self.ax.set_title("Press enter to confirm cropping to this view, esc to cancel")
                self.fig.canvas.draw()
                self.mode = "crop"

            elif event.key == ">":
                self.deselect()
                self.ax.set_title("Select a subpulse to delete.\nThen press enter to confirm, esc to leave delete mode.")
                self.fig.canvas.draw()
                self.mode = "delete_subpulse"

            elif event.key == ".":
                self.deselect()
                self.ax.set_title("Add subpulses by clicking on the pulsestack.\nThen press enter to confirm, esc to leave add mode.")
                self.fig.canvas.draw()
                self.mode = "add_subpulse"

            elif event.key == "/":
                self.deselect()
                self.ax.set_title("Add drift mode boundaries by clicking on the pulsestack.\nThen press enter to confirm, esc to leave add mode.")
                self.fig.canvas.draw()
                self.mode = "add_drift_mode_boundary"

            elif event.key == "?":
                self.deselect()
                self.ax.set_title("Delete drift mode boundaries by clicking on the pulsestack.\nThen press enter to confirm, esc to leave delete mode.")
                self.fig.canvas.draw()
                self.mode = "delete_drift_mode_boundary"

            elif event.key == "P":
                cropped = self.crop(pulse_range=self.ax.get_ylim(), phase_deg_range=self.ax.get_xlim(), inplace=False)

                # Make the profile and an array of phases
                profile = np.mean(cropped.values, axis=0)
                phases = np.arange(cropped.nbins)*cropped.dphase_deg + cropped.first_phase

                profile_fig, profile_ax = plt.subplots()
                profile_ax.plot(phases, profile)
                profile_ax.set_xlabel("Pulse phase (deg)")
                profile_ax.set_ylabel("Flux density (a.u.)")
                profile_ax.set_title("Profile of pulses {} to {}".format(cropped.first_pulse, cropped.first_pulse + (cropped.npulses - 1)*cropped.dpulse))
                profile_fig.show()

            elif event.key == "T":
                cropped = self.crop(pulse_range=self.ax.get_ylim(), phase_deg_range=self.ax.get_xlim(), inplace=False)

                # Make the LRFS
                lrfs = np.fft.rfft(cropped.values, axis=0)
                freqs = np.fft.rfftfreq(cropped.npulses, cropped.dpulse)
                df    = freqs[1] - freqs[0]

                profile_fig, profile_ax = plt.subplots()
                extent = cropped.calc_image_extent()
                extent = (extent[0], extent[1], freqs[1] - df/2, freqs[-1] + df/2)
                profile_ax.imshow(np.abs(lrfs[1:,:]), aspect='auto', origin='lower', interpolation='none', cmap='hot', extent=extent)
                profile_ax.set_xlabel("Pulse phase (deg)")
                profile_ax.set_ylabel("cycles per period")
                profile_ax.set_title("LRFS of pulses {} to {}".format(cropped.first_pulse, cropped.first_pulse + (cropped.npulses - 1)*cropped.dpulse))
                profile_fig.show()

            elif event.key == "v":
                self.ax.set_title("Toggle visibility mode. Press escape when finished.\nsubpulses (.), drift mode boundaries (/), quadratic fits (@)")
                self.fig.canvas.draw()
                self.mode = "toggle_visibility"

            elif event.key == "z":
                self.ax.set_title("Select a drift sequence by clicking on the pulsestack.\nPress enter to confirm, esc to cancel.")
                self.fig.canvas.draw()
                self.mode = "zoom_drift_sequence"

            elif event.key == "#":
                self.ax.set_title("Select a drift sequence by clicking on the pulsestack.\nPress enter to confirm, esc to cancel.")
                self.fig.canvas.draw()
                self.mode = "improve_model_fits"

            elif event.key == "h":
                # If some other function has explicitly changed the axis limits,
                # then pressing 'h' will default back to those changed limits.
                # This is to ensure that it always goes back to the whole
                # pulsestack
                extent = self.calc_image_extent()

                xlim = [extent[0], extent[1]]
                ylim = [extent[2], extent[3]]

                self.ax.set_xlim(xlim)
                self.ax.set_ylim(ylim)

                self.fig.canvas.draw()

            elif event.key == "d":
                crosscorr, lags, shift = self.cross_correlate_successive_pulses()

                # Calculate extent "by hand"
                extent = (lags[0] - 0.5*self.dphase_deg,
                        lags[-1] + 0.5*self.dphase_deg,
                        self.first_pulse - 0.5*self.dpulse,
                        self.first_pulse + (crosscorr.shape[0] - 0.5)*self.dpulse)

                corr_fig, corr_axs = plt.subplots(2, 1, sharex=True)

                corr_axs[1].imshow(crosscorr, aspect='auto', origin='lower', interpolation='none', cmap='hot', extent=extent)
                corr_axs[1].set_xlabel("Correlation lag (deg)")
                corr_axs[1].set_ylabel("Pulse number of first pulse")
                corr_axs[1].set_title("Cross correlation of each pulse with its successor")

                corr_axs[0].axvline(0, linestyle='--', color='k')
                corr_axs[0].plot(lags, np.sum(crosscorr, axis=0))
                corr_axs[0].set_title("Sum of (below) cross correlations")

                # Also, get the drift rates and plot them on the cross correlation
                approxP2 = 15 # CHANGE ME! Make more general!
                driftrates = self.driftrates_via_cross_correlation(approxP2)
                corr_axs[1].plot(driftrates, self.get_pulses_array()[:-1], 'gx')

                corr_fig.show()

            elif event.key == "A":
                autocorr, lags, shift = self.auto_correlate_pulses(set_DC_value=np.nan)

                # Calculate extent "by hand"
                extent = (lags[0] - 0.5*self.dphase_deg,
                        lags[-1] + 0.5*self.dphase_deg,
                        self.first_pulse - 0.5*self.dpulse,
                        self.first_pulse + (autocorr.shape[0] - 0.5)*self.dpulse)

                corr_fig, corr_axs = plt.subplots(2, 1, sharex=True)

                corr_axs[1].imshow(autocorr, aspect='auto', origin='lower', interpolation='none', cmap='hot', extent=extent)
                corr_axs[1].set_xlabel("Correlation lag (deg)")
                corr_axs[1].set_ylabel("Pulse number")
                corr_axs[1].set_title("Auto-correlation of each pulse")

                corr_axs[0].axvline(0, linestyle='--', color='k')
                corr_axs[0].plot(lags, np.sum(autocorr, axis=0))
                corr_axs[0].set_title("Sum of (below) auto-correlations")

                corr_fig.show()

            elif event.key == "@":
                self.ax.set_title("Quadratic fit: Choose subpulses ('.' to confirm subpulse,\nenter to confirm fit, esc to cancel)")
                self.fig.canvas.draw()
                self.mode = "model_fit"
                self.quadratic_selected = []
                self.quadratic_selected_plt = None
                self.drift_sequence_selected = None

            elif event.key == "$":
                dr_fig, dr_ax = plt.subplots()
                for seq in self.model_fits:
                    p_lo, p_hi = self.model_fits[seq].get_pulse_idx_bounds()
                    pulse_idxs = np.arange(p_lo, p_hi + 1)
                    pulses     = self.get_pulse_from_bin(pulse_idxs)
                    driftrates = self.model_fits[seq].calc_driftrate(pulses)
                    dr_ax.plot(pulses, driftrates, 'k')
                dr_ax.set_xlabel("Pulse number")
                dr_ax.set_ylabel("Drift rate (deg/pulse)")
                dr_fig.show()

            elif event.key == "%":
                dr_fig = plt.figure()
                dr_ax = plt.axes(projection='3d')
                p  = [] # Pulse number
                d  = [] # Drift rate
                dd = [] # Derivative of drift rate (w.r.t. pulse number)
                for seq in self.model_fits:
                    p_lo, p_hi = self.model_fits[seq].get_pulse_idx_bounds()
                    p.append(self.get_pulse_from_bin(0.5*(p_lo + p_hi)))
                    d.append(self.model_fits[seq].calc_driftrate(p[-1]))
                    dd.append(self.model_fits[seq].calc_driftrate_derivative(p[-1]))
                    dr_ax.plot(p, d, dd, 'ro')
                dr_ax.set_xlabel("Pulse number")
                dr_ax.set_ylabel("Drift rate (deg/pulse)")
                dr_ax.set_zlabel("Drift rate derivative (deg/pulse^2)")
                dr_fig.show()

            '''
            elif event.key == "r":
                approxP2 = 15 # CHANGE ME! Make more general!
                driftrates = self.driftrates_via_cross_correlation(approxP2)

                # Plot results
                dr_fig, dr_ax = plt.subplots()
                dr_ax.plot(self.get_pulses_array()[:-1], driftrates)
                dr_ax.set_xlabel("Pulse number of first pulse in cross correlation")
                dr_ax.set_ylabel("Drift rate (deg)")
                dr_fig.show()
            '''

        ########################################
        # SPECIALISED KEYS FOR DIFFERENT MODES #
        ########################################

        elif self.mode == "toggle_visibility":
            if event.key == ".":
                if self.subpulses_plt is not None:
                    self.subpulses_plt.set_data([], [])
                    self.subpulses_plt = None
                else:
                    self.plot_subpulses()
                self.fig.canvas.draw()

            elif event.key == "/":
                if self.dm_boundary_plt is not None:
                    self.dm_boundary_plt.set_segments(np.empty((0,2)))
                    self.dm_boundary_plt = None
                else:
                    self.plot_drift_mode_boundaries()
                self.fig.canvas.draw()

            elif event.key == "@":
                if self.quadratic_visible:
                    self.unplot_all_model_fits()
                    self.quadratic_visible = False
                else:
                    self.plot_all_model_fits()
                    self.quadratic_visible = True
                self.fig.canvas.draw()

            elif event.key == "escape":
                self.set_default_mode()

        elif self.mode == "set_threshold":
            if event.key == "enter":
                self.threshold_line.set_data([], [])
                self.subpulses.delete_all_subpulses()
                self.subpulses.add_subpulses(self.max_locations[1], self.max_locations[0])
                if self.jsonfile is not None:
                    self.fig.canvas.manager.set_window_title(self.jsonfile + "*")
                self.set_default_mode()
            elif event.key == "escape":
                self.threshold_line.set_data([], [])
                self.plot_subpulses()
                self.maxima_threshold = self.old_maxima_threshold
                self.set_default_mode()

        elif self.mode == "crop":
            if event.key == "enter":
                self.crop(pulse_range=self.ax.get_ylim(), phase_deg_range=self.ax.get_xlim())
                self.ps_image.set_data(self.values)
                self.ps_image.set_extent(self.calc_image_extent())
                if self.jsonfile is not None:
                    self.fig.canvas.manager.set_window_title(self.jsonfile + "*")
                self.set_default_mode()
            elif event.key == "escape":
                self.set_default_mode()

        elif self.mode == "delete_subpulse":
            if event.key == "enter":
                if self.selected is not None:
                    # Delete the selected point from the actual list
                    # Here, "selected" refers to the idx of subpulses[]
                    self.subpulses.delete_subpulses(self.selected)

                    # Delete the point from the plot
                    self.plot_subpulses()

                    # Unselect
                    self.deselect()

                    # Redraw the figure
                    if self.jsonfile is not None:
                        self.fig.canvas.manager.set_window_title(self.jsonfile + "*")
                    self.fig.canvas.draw()

            elif event.key == "escape":
                self.deselect()
                self.set_default_mode()

        elif self.mode == "delete_drift_mode_boundary":
            if event.key == "enter":
                if self.selected is not None:
                    # Here, "selected" refers to the idx of drift_mode_boundaries[]

                    # Before actually deleting the boundary, we have to consider how deleting
                    # it will affect the quadratic fits. Only two changes have to be made:
                    #   1) the fits for the immediately affected sequences should be deleted,
                    #      as there's no (easy) way to decide which of the two surrounding
                    #      sequences should take precedence
                    #   2) all quadratic fits to later sequences have to be assigned a new
                    #      (lower) sequence number
                    seq = self.selected
                    nseq = self.drift_sequences.number_of_sequences()

                    # 1.
                    if seq in self.model_fits.keys():
                        self.model_fits[seq].clear_all_plots()
                        self.model_fits.pop(seq)

                    if seq+1 in self.model_fits.keys():
                        self.model_fits[seq+1].clear_all_plots()
                        self.model_fits.pop(seq+1)

                    # 2.
                    # Start from the next sequence and work up, bumping each one down by 1 as we go
                    for i in range(seq+2, nseq):
                        if i in self.model_fits.keys():
                            self.model_fits[i-1] = self.model_fits.pop(i)

                    # Now, actually delete the selected boundary
                    self.drift_sequences.delete_boundaries([self.selected])

                    # Delete the boundary line from the plot
                    self.plot_drift_mode_boundaries()

                    # Deselect
                    self.deselect()

                    # Redraw the figure
                    if self.jsonfile is not None:
                        self.fig.canvas.manager.set_window_title(self.jsonfile + "*")
                    self.fig.canvas.draw()

            elif event.key == "escape":
                self.deselect()
                self.set_default_mode()

        elif self.mode == "add_subpulse":
            if event.key == "enter":
                # Here, "selected" is [pulse, phase] of new candidate subpulse
                if self.selected is not None:
                    self.subpulses.add_subpulses([self.selected[1]], [self.selected[0]])
                    self.plot_subpulses()
                self.deselect()
                if self.jsonfile is not None:
                    self.fig.canvas.manager.set_window_title(self.jsonfile + "*")
                self.fig.canvas.draw()

            elif event.key == "escape":
                self.deselect()
                self.set_default_mode()

        elif self.mode == "add_drift_mode_boundary":
            if event.key == "enter":
                if self.selected is None:
                    return

                # Here, "selected" refers to the pulse_idx of the drift mode boundary

                # Before actually adding the boundary, we have to consider how adding it
                # will affect the quadratic fits. Only two changes have to be made:
                #   1) all quadratic fits to later sequences have to be assigned a new
                #      (higher) sequence number
                #   2) the fit for the selected sequence should be copied to the two "new"
                #      sequences, with the appropriate adjustments to the pulse ranges
                seq = self.drift_sequences.get_sequence_number(self.selected, self.npulses)
                nseq = self.drift_sequences.number_of_sequences()

                # 1.
                # Start from the last sequence and work down, bumping each one up by 1 as we go
                for i in range(nseq, seq, -1):
                    if i in self.model_fits.keys():
                        self.model_fits[i+1] = self.model_fits.pop(i)

                # 2.
                if seq in self.model_fits.keys():
                    self.model_fits[seq+1] = copy.copy(self.model_fits[seq])
                    self.model_fits[seq].last_pulse_idx = self.selected
                    self.model_fits[seq+1].first_pulse_idx = self.selected + 1

                    # Redraw the affected plots
                    if self.quadratic_visible:
                        if self.onpulse is None:
                            phlim = self.get_phase_from_bin(np.array([0, self.nbins-1]))
                        else:
                            phlim = self.onpulse

                        self.model_fits[seq].clear_all_plots()
                        self.model_fits[seq].plot_all_driftbands(self.ax, phlim, self.get_pulse_from_bin, color='k')

                        self.model_fits[seq+1].clear_all_plots()
                        self.model_fits[seq+1].plot_all_driftbands(self.ax, phlim, self.get_pulse_from_bin, color='k')

                # Now actually add the boundary
                self.drift_sequences.add_boundary(self.selected)

                xlim = self.ax.get_xlim()
                ylim = self.ax.get_ylim()

                self.plot_drift_mode_boundaries()
                self.deselect()

                self.ax.set_xlim(xlim)
                self.ax.set_ylim(ylim)

                if self.jsonfile is not None:
                    self.fig.canvas.manager.set_window_title(self.jsonfile + "*")
                self.fig.canvas.draw()

            elif event.key == "escape":
                self.deselect()
                self.set_default_mode()

        elif self.mode == "zoom_drift_sequence":
            if event.key == "enter":
                if self.selected is None:
                    return

                # Here, "selected" refers to the drift sequence number
                first_pulse, last_pulse = self.drift_sequences.get_bounding_pulse_idxs(self.selected, self.npulses)

                # Set the zoom (only in y-direction)
                ylo = self.get_pulse_from_bin(first_pulse - 0.5)
                yhi = self.get_pulse_from_bin(last_pulse + 0.5)
                self.ax.set_ylim([ylo, yhi])

                self.deselect()
                self.set_default_mode()

            elif event.key == "escape":
                self.deselect()
                self.set_default_mode()

        elif self.mode == "improve_model_fits":
            if event.key == "enter":
                # Here, self.selected is the drift sequence idx
                if self.selected is None:
                    return

                # Assign each subpulse in sequence to the nearest driftband
                subset = self.subpulses.assign_quadratic_driftbands_to_subpulses(self.model_fits[self.selected], self.get_pulse_from_bin)

                # Now use ALL the subpulses in the sequence to get an improved fit
                ph = self.subpulses.get_phases(subset=subset)
                p  = self.subpulses.get_pulses(subset=subset)
                d  = self.subpulses.get_driftbands(subset=subset)
                self.model_fits[self.selected].least_squares_fit_to_subpulses(ph, p, d)

                # Update the plot
                if self.onpulse is None:
                    phlim = self.get_phase_from_bin(np.array([0, self.nbins-1]))
                else:
                    phlim = self.onpulse

                self.model_fits[self.selected].plot_all_driftbands(self.ax, phlim, self.get_pulse_from_bin, color='k')

                # Mark that unsaved changes have been made
                if self.jsonfile is not None:
                    self.fig.canvas.manager.set_window_title(self.jsonfile + "*")

                self.deselect()
                self.set_default_mode()

            elif event.key == "escape":
                self.deselect()
                self.set_default_mode()

        elif self.mode == "model_fit":
            if event.key == ".":
                # There has to be a selected subpulse for this to do anything
                subpulse_idx = self.selected
                if subpulse_idx is None:
                    return

                # First, make the user assign a driftband number
                root = tkinter.Tk()
                root.withdraw()
                driftband = tkinter.simpledialog.askfloat("Driftband number", "Assign a driftband number to this subpulse", parent=root)
                if not driftband:
                    self.deselect()
                    self.fig.canvas.draw()
                    return

                # Get the pulse and phase of the selected subpulse
                phase = self.subpulses.get_phases()[subpulse_idx]
                pulse = self.subpulses.get_pulses()[subpulse_idx]
                pulse_idx = self.get_pulse_bin(pulse, inrange=False)

                # Next, set the selected drift sequence if it hasn't been selected yet
                if self.drift_sequence_selected is None:
                    self.drift_sequence_selected = self.drift_sequences.get_sequence_number(pulse_idx, self.npulses)

                    # Also, if this (newly selected) drift sequence already has plotted driftbands, remove them from the plot
                    if self.drift_sequence_selected in self.model_fits.keys():
                        self.model_fits[self.drift_sequence_selected].clear_all_plots()
                        self.fig.canvas.draw()

                # Add this subpulse's info to the list of previous selections
                self.quadratic_selected.append([phase, pulse, driftband])

                # Try to draw the model so far...
                q  = np.array(self.quadratic_selected)
                ph = q[:,0] # The phases
                p  = q[:,1] # The pulses
                d  = q[:,2] # The driftbands

                # Draw the points selected so far
                if self.quadratic_selected_plt is None:
                    self.quadratic_selected_plt = self.ax.plot(ph, p, 'bo')
                else:
                    self.quadratic_selected_plt[0].set_data(ph, p)

                if len(self.quadratic_selected) >= 4:
                    self.candidate_quadratic_model.least_squares_fit_to_subpulses(ph, p, d)
                    first_pulse_idx, last_pulse_idx = self.drift_sequences.get_bounding_pulse_idxs(self.drift_sequence_selected, self.npulses)
                    self.candidate_quadratic_model.set_pulse_bounds(first_pulse_idx, last_pulse_idx)

                    # Only plot the driftbands in the on pulse region
                    if self.onpulse is None:
                        phlim = self.get_phase_from_bin(np.array([0, self.nbins-1]))
                    else:
                        phlim = self.onpulse

                    # Clear all previous driftbands from the plot and replace them with new ones!
                    self.candidate_quadratic_model.clear_all_plots()
                    self.candidate_quadratic_model.plot_all_driftbands(self.ax, phlim, self.get_pulse_from_bin, color='w')
                    self.fig.canvas.draw()

            elif event.key == "enter" or event.key == "escape":

                # If they push enter too early, do nothing
                if event.key == "enter" and len(self.quadratic_selected) < 4:
                    return

                self.candidate_quadratic_model.clear_all_plots()

                # The only difference between enter and escape is that enter saves the model
                if event.key == "enter":
                    self.model_fits[self.drift_sequence_selected] = copy.copy(self.candidate_quadratic_model)

                    if self.jsonfile is not None:
                        self.fig.canvas.manager.set_window_title(self.jsonfile + "*")

                # Make sure the saved model (whether old or new) is drawn
                if self.onpulse is None:
                    phlim = self.get_phase_from_bin(np.array([0, self.nbins-1]))
                else:
                    phlim = self.onpulse

                if self.drift_sequence_selected in self.model_fits.keys():
                    self.model_fits[self.drift_sequence_selected].plot_all_driftbands(self.ax, phlim, self.get_pulse_from_bin, color='k')

                if self.quadratic_selected_plt is not None:
                    self.quadratic_selected_plt[0].set_data([], [])
                    self.quadratic_selected_plt = None

                self.deselect()
                self.set_default_mode()

    def closest_drift_mode_boundary(self, y):
        dm_boundary_display = self.ax.transData.transform([[0,y] for y in self.get_pulse_from_bin(self.drift_sequences.get_pulse_mid_idxs())])
        dists = np.abs(y - dm_boundary_display[:,1])
        idx = np.argmin(dists)
        return idx, dists[idx]

    def closest_subpulse(self, x, y):
        subpulses_display = self.ax.transData.transform(self.subpulses.get_positions())
        dists = np.hypot(x - subpulses_display[:,0], y - subpulses_display[:,1])
        idx = np.argmin(dists)
        return idx, dists[idx]

    def start(self):
        '''
        Start the interactive plot
        '''

        # Make the plots
        self.plot_image()

        xlim = self.ax.get_xlim()
        ylim = self.ax.get_ylim()

        self.plot_subpulses()
        self.plot_drift_mode_boundaries()
        self.plot_all_model_fits()

        self.ax.set_xlim(xlim)
        self.ax.set_ylim(ylim)

        # Set the mode to "default"
        self.set_default_mode()

        # Make it interactive!
        self.cid = self.fig.canvas.mpl_connect('button_press_event', self.on_button_press_event)
        self.cid = self.fig.canvas.mpl_connect('key_press_event', self.on_key_press_event)

        # Set the window title to the json filename
        if self.jsonfile is not None:
            self.fig.canvas.manager.set_window_title(self.jsonfile)
        else:
            self.fig.canvas.manager.set_window_title("[Unsaved pulsestack]")

        # Show the plot
        plt.show()



if __name__ == '__main__':
    # Start an interactive plot instance
    ps = DriftAnalysisInteractivePlot()

    # Load the data
    # If one argument is given, assume it is in the custom saved (json) format
    if len(sys.argv) == 2:
        ps.load_json(sys.argv[1])
    # Otherwise, assume the first argument is a pdv file, and the second is a stokes parameter
    else:
        pdvfile = sys.argv[1]
        stokes = sys.argv[2]
        ps.load_from_pdv(pdvfile, stokes)

    # Initiate the interactive plots
    ps.start()


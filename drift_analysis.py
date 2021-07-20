__version__ = "0.9.7"

import sys
import copy

import numpy as np
from numpy.polynomial.polynomial import polyfit, polyval

from mpl_toolkits import mplot3d
import matplotlib.pyplot as plt

from scipy.interpolate import interp1d
from scipy.ndimage import gaussian_filter1d
from scipy.optimize import curve_fit

import tkinter
import tkinter.filedialog
import tkinter.simpledialog

import json
import bisect
import pulsestack

class Subpulses:

    def __init__(self):

        self.data = None
        self.with_driftbands_plt = None
        self.no_driftbands_plt = None

    def plot_subpulses(self, ax, pulse_range=None, **kwargs):
        if self.get_nsubpulses() == 0:
            return

        # Get array masks for those subpulses with and without valid driftbands
        with_driftbands_subset = self.in_pulse_range(pulse_range, with_valid_driftband=True)
        no_driftbands_subset = np.logical_not(with_driftbands_subset)

        # WITH DRIFTBANDS
        if np.any(with_driftbands_subset): # If there ARE any with driftbands
            # If not yet plotted, plot them anew (in blue)
            ph = self.get_phases(subset=with_driftbands_subset)
            p = self.get_pulses(subset=with_driftbands_subset)
            if self.with_driftbands_plt is None:
                self.with_driftbands_plt, = ax.plot(ph, p, 'bx', **kwargs)
            else:
                self.with_driftbands_plt.set_data(ph, p)
        else:
            # If there are none to plot, clear any pre-existing ones
            self.clear_plots(with_driftbands=True, no_driftbands=False)

        # NO ASSIGNED DRIFTBANDS
        if np.any(no_driftbands_subset): # If there ARE any without driftbands
            # If not yet plotted, plot them anew (in green)
            ph = self.get_phases(subset=no_driftbands_subset)
            p = self.get_pulses(subset=no_driftbands_subset)
            if self.no_driftbands_plt is None:
                self.no_driftbands_plt, = ax.plot(ph, p, 'gx', **kwargs)
            else:
                self.no_driftbands_plt.set_data(ph, p)
        else:
            # If there are none to plot, clear any pre-existing ones
            self.clear_plots(with_driftbands=False, no_driftbands=True)

    def clear_plots(self, with_driftbands=True, no_driftbands=True):

        if with_driftbands:
            if self.with_driftbands_plt is not None:
                self.with_driftbands_plt.set_data([], [])
                self.with_driftbands_plt = None

        if no_driftbands:
            if self.no_driftbands_plt is not None:
                self.no_driftbands_plt.set_data([], [])
                self.no_driftbands_plt = None

    def serialize(self):

        serialized = {}
        if self.data is not None:
            serialized["pulses"]     = list(self.get_pulses().astype(float))
            serialized["phases"]     = list(self.get_phases().astype(float))
            serialized["widths"]     = list(self.get_widths().astype(float))
            serialized["driftbands"] = list(self.get_driftbands().astype(float))

        return serialized

    def unserialize(self, serialized):
        if len(serialized) > 0:
            pulses       = serialized["pulses"]
            phases       = serialized["phases"]
            widths       = serialized["widths"]
            driftbands   = serialized["driftbands"]

            self.add_subpulses(phases, pulses, widths=widths, driftbands=driftbands)
        else:
            self.data = None

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

        if self.data is None:
            self.data = newsubpulses
        else:
            self.data = np.vstack((self.data, newsubpulses))

    def delete_subpulses(self, delete_idxs):
        self.data = np.delete(self.data, delete_idxs, axis=0)

    def delete_all_subpulses(self):
        self.data = None

    def get_nsubpulses(self):
        if self.data is None:
            return 0
        else:
            return self.data.shape[0]

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

    def in_pulse_range(self, pulse_range=None, with_valid_driftband=False):
        '''
        Returns a logical mask for those subpulses within the specified range
        '''
        p  = self.get_pulses()
        if pulse_range is not None:
            p_lo, p_hi = pulse_range
            is_in_range = np.logical_and(p >= p_lo, p <= p_hi)
        else:
            is_in_range = np.ones(p.shape).astype(bool) # Should be an array of all True

        if with_valid_driftband:
            d = self.get_driftbands()
            is_valid_driftband = np.logical_not(np.isnan(d))
            return np.logical_and(is_in_range, is_valid_driftband)
        else:
            return is_in_range

    def assign_driftbands_to_subpulses(self, model_fit):
        '''
        model_fit - an object of ModelFit
        This will classify each subpulse in the appropriate pulse range
        into a driftband according to the given model.
        '''
        # Get only those subpulses within the valid range of the quadratic fit
        pulse_range = np.array(model_fit.get_pulse_bounds())
        subset = self.in_pulse_range(pulse_range)

        ph = self.get_phases(subset=subset)
        p  = self.get_pulses(subset=subset)
        d = model_fit.get_nearest_driftband(p, ph)

        self.set_driftbands(d, subset=subset)

        # For good measure, return the subset of affected subpulse
        return subset

class DriftSequences:
    def __init__(self):
        self.boundaries = []

    def serialize(self):
        serialized = {}

        serialized["boundaries"] = list(self.boundaries)

        return serialized

    def unserialize(self, serialized):

        if "boundaries" in serialized.keys():
            self.boundaries = serialized["boundaries"]
        else:
            self.boundaries = []

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
        self.boundaries = [int(v) for i, v in enumerate(self.boundaries) if i not in boundary_idxs]

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

        return [first_idx, last_idx]

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
        self.parameters  = None
        self.pcov        = None

        self.first_pulse = None
        self.last_pulse  = None

        self.model_name  = None

        # A dictionary for keeping track of line plot objects
        # Dictioney keys are intended to be driftband numbers
        self.driftband_plts = {}

    def get_nparameters(self):
        return len(self.get_parameter_names())

    def get_parameter_names(self, display_type=None):
        '''
        display_type can be 'latex'. None default to ascii-type output
        '''
        if self.model_name == "quadratic":
            if display_type == "latex":
                return ["a_1", "a_2", "a_3", "a_4"]
            else:
                return ["a1", "a2", "a3", "a4"]
        elif self.model_name == "exponential":
            if display_type == "latex":
                return ["D_0", "k", "\\varphi_0", "P_2"]
            else:
                return ["D0", "k", "phi0", "P2"]
        else:
            self.print_unecognised_model_error()

    def get_parameter_by_name(self, parameter_name):
        try:
            idx = self.get_parameter_names().index(parameter_name)
            return self.parameters[idx]
        except:
            return None

    def __str__(self):

        if self.model_name == "quadratic":
            equation_string = "phi = a1*p^2 + a2*p + a3 + a4*d"
        elif self.model_name == "exponential":
            equation_string = "phi = (D0/k)*(1 - exp(-k*(p - p0)) + phi0 + P2*d)"
        else:
            return "Unrecognised model '{}'".format(self.model)

        model_string = "Model: {} ({})\nPulse range: {:.1f} - {:.1f}\n".format(self.model_name, equation_string, self.first_pulse, self.last_pulse)

        parameter_names = self.get_parameter_names()
        for i in range(len(self.parameters)):
            model_string += "  {:4} = {}".format(parameter_names[i], self.parameters[i])
            if self.pcov is not None:
                model_string += " +- {}".format(np.sqrt(self.pcov[i,i]))
            model_string += "\n"

        return model_string

    def print_unrecognised_model_error(self):
        print("Unrecognised model '" + self.model_name + "'")

    def set_pulse_bounds(self, first_pulse, last_pulse):
        self.first_pulse = first_pulse
        self.last_pulse  = last_pulse

    def get_pulse_bounds(self):
        return [self.first_pulse, self.last_pulse]

    def convert_to_model(self, new_model_name):
        '''
        This will convert the existing into a (very rough!) approximation of the specified model.
        Details of the conversions are given in the comments.
        '''
        if new_model_name == self.model_name:
            return

        p0, pf = self.get_pulse_bounds()

        if new_model_name == "exponential":
            # This conversion keeps the phase and P2 of the driftbands at the beginning of the drift
            # sequence the same, and chooses the exponential model which matches the drift rate at
            # both the beginning and the end of the sequence. It is NOT guaranteed that the phase
            # of the driftbands match at the end of the drift sequence
            P2   = self.calc_P2()
            phi0 = self.calc_phase(p0, 0)
            D0   = self.calc_driftrate(p0)
            Df   = self.calc_driftrate(pf)
            k    = -np.log(Df/D0) / (pf - p0)

            self.parameters = [D0, k, phi0, P2]
            self.model_name = new_model_name

        if new_model_name == "quadratic":
            # This conversion keeps the phase and P2 of the driftbands at the beginning of the drift
            # sequence the same, and chooses the quadratic model which matches the drift rate at
            # both the beginning and the end of the sequence. It is NOT guaranteed that the phase
            # of the driftbands match at the end of the drift sequence
            P2 = self.calc_P2()
            D0 = self.calc_driftrate(p0)
            Df = self.calc_driftrate(pf)
            phi0 = self.calc_phase(p0, 0)

            a1 = (D0 - Df)/(2*(p0 - pf))
            a2 = (D0*pf - Df*p0)/(pf - p0)
            a3 = phi0 - a1*p0**2 - a2*p0
            a4 = P2

            self.parameters = [a1, a2, a3, a4]
            self.model_name = new_model_name

    def optimise_fit_to_subpulses(self, phases, pulses, driftbands):
        # A valid model must be specified in self.model_name
        if self.model_name is None:
            print("Unspecified model. Cannot optimise model fit. Aborting")
            return

        print("Fitting points to mode \"" + self.model_name + "\"")

        # phases, pulses, and driftbands must be vectors with the same length
        npoints = len(phases)
        if len(pulses) != npoints or len(driftbands)!= npoints:
            print("phases, pulses, driftbands have lengths {}, {}, and {}, but they must be the same".format(len(phases), len(pulses), len(driftbands)))
            return

        # Form the matrices for least squares fitting
        ph = np.array(phases)
        p  = np.array(pulses)
        d  = np.array(driftbands)

        # Get things in a form that curve_fit needs
        xdata = np.array([p, d])
        ydata = np.array(ph)

        # Use the existing model as the initial guess
        p0 = self.parameters
        if p0 is None:
            print("No initial guess supplied. Results may vary")
            if self.model_name == "quadratic":
                p0 = np.ones((4,))
            elif self.model_name == "exponential":
                p0 = np.ones((4,))

        # Call curve_fit
        popt, pcov = curve_fit(self.calc_phase_for_curvefit, xdata, ydata, p0=p0)
        # TO DO: Include Jacobian in the above call to curve_fit

        # Set these parameters!
        self.parameters = popt
        self.pcov       = pcov

    def serialize(self):
        serialized = {}

        if self.model_name is not None:
            serialized["model_name"] = self.model_name

        if self.parameters is not None:
            serialized["parameters"] = list(self.parameters)

        if self.first_pulse is not None and self.last_pulse is not None:
            serialized["pulse_range"] = [int(self.first_pulse), int(self.last_pulse)]

        if self.pcov is not None:
            serialized["pcov"] = list(self.pcov.flatten())

        return serialized

    def unserialize(self, data):
        if "model_name" in data.keys():
            self.model_name = data["model_name"]
        else:
            self.model_name = None

        if "parameters" in data.keys():
            self.parameters = data["parameters"]
            nparameters = len(self.parameters)

            # Only attempt to make sense of the pcov if there are already parameters
            # Also, the number of elements in pcov MUST be the square of the number of parameters
            if "pcov" in data.keys():
                if len(data["pcov"]) == nparameters**2:
                    self.pcov = np.reshape(data["pcov"], (nparameters, nparameters))
                else:
                    self.pcov = None
            else:
                self.pcov = None
        else:
            self.parameters = None

        if "pulse_range" in data.keys():
            self.first_pulse, self.last_pulse = data["pulse_range"]
        else:
            self.first_pulse = None
            self.last_pulse  = None


    def calc_phase(self, pulse, driftband):
        p  = pulse
        d  = driftband
        p0 = self.first_pulse

        if self.model_name == "quadratic":
            # See McSweeney et al. (2017)
            a1, a2, a3, a4 = self.parameters
            return a1*p**2 + a2*p + a3 + a4*d

        elif self.model_name == "exponential":
            D0, k, phi0, P2 = self.parameters
            return (D0/k)*(1 - np.exp(-k*(p - p0))) + phi0 + P2*d

        else:
            self.print_unrecognised_model_error()
            return

    def calc_phase_for_curvefit(self, xdata, *params):
        '''
        A wrapper for calc_phase(), so that it can be used with scipy's curvefit
        '''

        p = xdata[0,:]
        d = xdata[1,:]

        model = copy.copy(self)
        model.parameters = np.array([*params])

        return model.calc_phase(p, d)

    def calc_driftrate(self, pulse):
        p  = pulse
        p0 = self.first_pulse

        if self.model_name == "quadratic":
            a1, a2, _, _ = self.parameters
            return 2*a1*p + a2

        elif self.model_name == "exponential":
            D0, k, _, _ = self.parameters
            return D0*np.exp(-k*(p - p0))

        else:
            self.print_unrecognised_model_error()
            return

    def calc_driftrate_derivative(self, pulse):
        '''
        Returns the driftrate derivative w.r.t. pulse
        '''
        p  = pulse
        p0 = self.first_pulse

        if self.model_name == "quadratic":
            a1, _, _, _ = self.parameters
            return 2*a1

        elif self.model_name == "exponential":
            D0, k, _, _ = self.parameters
            return -k*D0*np.exp(-k*(p - p0))  # also could write -k*self.calc_driftrate(p)

        else:
            self.print_unrecognised_model_error()
            return

    def calc_driftrate_decay_rate(self, pulse):
        return -self.calc_driftrate_derivative(pulse)/self.calc_driftrate(pulse)

    def calc_residual(self, pulse, phase, driftband):
        '''
        Calculates subpulse phase minus model phase
        '''
        model_phase = self.calc_phase(pulse, driftband)
        residual = phase - model_phase
        return residual

    def shift_phase(self, phase_shift):
        '''
        Recalculate the model parameters to shift the model in phase.
        This calculation is model-dependent, so in principle has to be done for each
        model separately (even though they may turn out to look the same in some cases)
        '''
        if self.model_name == "quadratic":
            self.parameters[2] += phase_shift
        elif self.model_name == "exponential":
            self.parameters[2] += phase_shift
        else:
            self.print_unrecognised_model_error()
            return

    def get_nearest_driftband(self, pulse, phase):
        p  = pulse
        ph = phase

        if self.model_name == "quadratic":
            a1, a2, a3, a4 = self.parameters
            return np.round((ph - a1*p**2 - a2*p - a3)/a4)

        elif self.model_name == "exponential":
            D0, k, phi0, P2 = self.parameters
            return np.round((ph - (D0/k)*(1 - np.exp(-k*(p - p0))) - phi0)/P2)

        else:
            self.print_unrecognised_model_error()
            return

    def calc_P2(self):

        if self.model_name == "quadratic":
            _, _, _, a4 = self.parameters
            return a4

        elif self.model_name == "exponential":
            _, _, _, P2 = self.parameters
            return P2

        else:
            self.print_unrecognised_model_error()
            return

    def calc_P3(self, pulse):
        return self.calc_P2()/self.calc_driftrate(pulse)

    def get_driftband_range(self, phlim):
        # Figure out if drift rate is positive or negative (at the first pulse)
        p0 = self.first_pulse
        pf = self.last_pulse

        if self.calc_driftrate(p0) < 0:
            ph0 = phlim[0]
        else:
            ph0 = phlim[1]

        if self.calc_driftrate(pf) < 0:
            phf  = phlim[1]
        else:
            phf  = phlim[0]

        if self.model_name == "quadratic":
            a1, a2, a3, a4 = self.parameters
            d0 = np.ceil((ph0 - a1*p0**2 - a2*p0 - a3)/a4)
            df = np.floor((phf - a1*pf**2 - a2*pf - a3)/a4)
        elif self.model_name == "exponential":
            D0, k, phi0, P2 = self.parameters
            d0 = np.ceil((ph0 - phi0)/P2)
            df = np.floor((phf - (D0/k)*(1 - np.exp(-k*(pf - p0))) - phi0)/P2)
        else:
            self.print_unrecognised_model_error()
            return

        return [int(d0), int(df)]

    def plot_driftband(self, ax, driftband, phlim=None, pstep=1, **kwargs):
        '''
        driftband: driftband number to plot
        phlim: phase limits between which to draw the driftband
        '''
        d = driftband

        p = np.arange(self.first_pulse, self.last_pulse + pstep, pstep)
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

    def plot_all_driftbands(self, ax, phlim, pstep=1, **kwargs):
        first_d, last_d = self.get_driftband_range(phlim)
        for d in range(first_d, last_d+1):
            self.plot_driftband(ax, d, phlim=phlim, pstep=pstep, **kwargs)

    def clear_all_plots(self):
        for d in self.driftband_plts:
            self.driftband_plts[d][0].set_data([], [])
        self.driftband_plts = {}

class DriftAnalysis(pulsestack.Pulsestack):
    def __init__(self):
        super().__init__()
        self.fig                       = None
        self.ax                        = None
        self.subpulses                 = Subpulses()
        self.maxima_plt             = None
        self.maxima_threshold          = 0.0
        self.drift_sequences           = DriftSequences()
        self.dm_boundary_plt           = None
        self.jsonfile                  = None
        self.candidate_quadratic_model = ModelFit()
        self.candidate_quadratic_model.model_name = "quadratic"
        self.onpulse                   = None
        self.model_fits            = {}  # Keys = drift sequence numbers
        self.quadratic_visible         = True

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
                "pulsestack":          self.serialize(),
                "subpulses":           self.subpulses.serialize(),
                "model_fits":          [[int(i), self.model_fits[i].serialize()] for i in self.model_fits],

                "maxima_threshold":    self.maxima_threshold,
                "drift_mode_boundaries": self.drift_sequences.serialize()
                }

        try:
            with open(jsonfile, "w") as f:
                json.dump(drift_dict, f)

            self.jsonfile = jsonfile

        except TypeError as err:
            print("Could not save out json file:", err)

    def load_json(self, jsonfile=None):

        if jsonfile is None:
            jsonfile = self.jsonfile

        with open(jsonfile, "r") as f:
            drift_dict = json.load(f)

        if drift_dict["version"] != __version__:
            print("Warning: version mismatch, File = {}, Software = {}".format(drift_dict["version"], __version__))

        # Load the pulsestack data
        self.unserialize(drift_dict["pulsestack"])
        self.subpulses.unserialize(drift_dict["subpulses"])

        for item in drift_dict["model_fits"]:
            self.model_fits[item[0]] = ModelFit()
            self.model_fits[item[0]].unserialize(item[1])

        self.maxima_threshold = drift_dict["maxima_threshold"]
        self.drift_sequences.unserialize(drift_dict["drift_mode_boundaries"])

        self.jsonfile = jsonfile

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

            self.model_fits[i].plot_all_driftbands(self.ax, phlim, pstep=self.dpulse, color='k')

    def unplot_all_model_fits(self):
        for i in self.model_fits:
            self.model_fits[i].clear_all_plots()

class DriftAnalysisInteractivePlot(DriftAnalysis):
    def __init__(self):
        super().__init__()
        self.mode            = "default"
        self.selected        = None
        self.selected_plt    = None

        self.visible_ps  = None
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
                    self.visible_ps.get_local_maxima(maxima_threshold=event.ydata)
                    self.maxima_threshold = self.visible_ps.maxima_threshold
                    self.max_locations = self.visible_ps.max_locations
                self.maxima_plt.set_data(self.max_locations[1,:], self.max_locations[0,:])
                self.fig.canvas.draw()

        elif self.mode == "set_fiducial":
            if event.inaxes == self.ax:
                # Set the fiducial phase for this pulsestack
                self.set_fiducial_phase(event.xdata)

                # If necessary, also set the same fiducial phase for the smoothed pulsestack
                if self.visible_ps is not None:
                    self.visible_ps.set_fiducial_phase(event.xdata)

                # Adjust all the maxima points
                if self.subpulses.get_nsubpulses() > 0:
                    self.subpulses.shift_all_subpulses(dphase=-event.xdata)

                # Shift the models
                for i in self.model_fits:
                    self.model_fits[i].shift_phase(-event.xdata)

                # Replot everything
                current_xlim = self.ax.get_xlim()
                current_ylim = self.ax.get_ylim()
                self.ax.set_xlim(current_xlim - event.xdata)
                self.ax.set_ylim(current_ylim)
                self.ps_image.set_extent(self.calc_image_extent())

                self.subpulses.plot_subpulses(self.ax)
                self.plot_drift_mode_boundaries()
                self.plot_all_model_fits()

                self.fig.canvas.draw()

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
                self.set_onpulse(self.ph_lo, self.ph_hi)

                # Add the * to the window title
                if self.jsonfile is not None:
                    self.fig.canvas.manager.set_window_title(self.jsonfile + "*")

                self.set_default_mode()

        elif self.mode == "zoom_drift_sequence" or self.mode == "switch_to_quadratic_and_solve" or self.mode == "assign_driftbands" or self.mode == "switch_to_exponential_and_solve" or self.mode == "display_model_details" or self.mode == "plot_residuals":
            if event.inaxes == self.ax:
                pulse_idx = self.get_pulse_bin(event.ydata, inrange=False)
                self.selected = self.drift_sequences.get_sequence_number(pulse_idx, self.npulses)
                if self.selected is not None:

                    # In some modes, the user only can select sequences with existing models
                    if self.mode == "switch_to_quadratic_and_solve" or self.mode == "switch_to_exponential_and_solve" or self.mode == "assign_driftbands" or self.mode == "display_model_details" or self.mode == "plot_residuals":
                        if self.selected not in self.model_fits.keys():
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
                print("D     Assign the nearest model driftband to each subpulse")
                print("r     Plot subpulse residuals from driftband model")
                print("@     Perform quadratic fitting via subpulse selection (McSweeney et al, 2017)")
                print("#     Switch to quadratic model and redo fit using all subpulses assigned driftbands in sequence")
                print("3     Plot model P3 as a function of pulse number")
                print("E     Switch to exponential model and redo fit using all subpulses assigned driftbands in sequence")
                print("$     Plot the drift rate of the model fits against pulse number")
                print("&     Plot the driftrate decay rate of the model fits against pulse number")
                print("%     3D plot of drift rate (d) vs d-dot vs pulse number")
                print("*     Plot the (exponential) model parameters as a function of pulse number")
                print("(     Plot the (quadratic) model parameters as a function of pulse number")
                print("m     Print model parameters to stdout")
                print("+/-   Set upper/lower colorbar range")

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
                if self.visible_ps is None:
                    self.show_smooth = False

                if self.show_smooth == False:
                    root = tkinter.Tk()
                    root.withdraw()
                    sigma = tkinter.simpledialog.askfloat("Smoothing kernel", "Input Gaussian kernel size (deg)", parent=root)
                    if sigma:
                        self.visible_ps = self.smooth_with_gaussian(sigma, inplace=False)
                        self.ps_image.set_data(self.visible_ps.values)
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

            elif event.key == "+":
                vmin, _ = self.ps_image.get_clim()
                root = tkinter.Tk()
                root.withdraw()
                vmax = tkinter.simpledialog.askfloat("Set upper dynamic range", "Input value for upper dynamic range", parent=root)
                self.ps_image.set_clim(vmin, vmax)
                self.fig.canvas.draw()

            elif event.key == "-":
                _, vmax = self.ps_image.get_clim()
                root = tkinter.Tk()
                root.withdraw()
                vmin = tkinter.simpledialog.askfloat("Set upper dynamic range", "Input value for upper dynamic range", parent=root)
                self.ps_image.set_clim(vmin, vmax)
                self.fig.canvas.draw()

            elif event.key == "^":
                self.ax.set_title("Set threshold on colorbar. Press enter when done, esc to cancel.")
                self.old_maxima_threshold = self.maxima_threshold # Save value in case they cancel
                if self.show_smooth == True:
                    self.visible_ps.get_local_maxima(maxima_threshold=self.visible_ps.maxima_threshold)
                    self.max_locations = self.visible_ps.max_locations
                else:
                    self.get_local_maxima()

                self.subpulses.clear_plots()
                if self.maxima_plt is None:
                    self.maxima_plt, = self.ax.plot(self.max_locations[1,:], self.max_locations[0,:], 'gx')
                else:
                    self.maxima_plt.set_data(self.max_locations[1,:], self.max_locations[0,:])

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
                if self.show_smooth == True:
                    cropped = self.visible_ps.crop(pulse_range=self.ax.get_ylim(), phase_deg_range=self.ax.get_xlim(), inplace=False)
                else:
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
                self.mode = "switch_to_quadratic_and_solve"

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
                # Start a new instance of DriftAnalysisInteractivePlot for the cross correlation
                self.cc = DriftAnalysisInteractivePlot()

                # Actually do the cross correlation and "copy" it across to this instance
                crosscorr = self.cross_correlate_successive_pulses()
                self.cc.stokes      = crosscorr.stokes
                self.cc.npulses     = crosscorr.npulses
                self.cc.nbins       = crosscorr.nbins
                self.cc.first_pulse = crosscorr.first_pulse
                self.cc.first_phase = crosscorr.first_phase
                self.cc.dpulse      = crosscorr.dpulse
                self.cc.dphase_deg  = crosscorr.dphase_deg
                self.cc.values      = crosscorr.values

                # Remove all the subpulses and models
                self.cc.subpulses = Subpulses()
                self.cc.model_fits = {}
                # ... but keep the drift sequence boundaries
                self.cc.drift_sequences = copy.copy(self.drift_sequences)

                # Make it interactive!
                self.cc.start()

            elif event.key == "D":
                self.ax.set_title("Select a drift sequence by clicking on the pulsestack.\nPress enter to confirm, esc to cancel.")
                self.fig.canvas.draw()
                self.mode = "assign_driftbands"

            elif event.key == "r":
                self.ax.set_title("Select a drift sequence by clicking on the pulsestack.\nPress enter to confirm, esc to cancel.")
                self.fig.canvas.draw()
                self.mode = "plot_residuals"

            elif event.key == "T":
                # Start a new instance of DriftAnalysisInteractivePlot for the LRFS
                self.lrfs = DriftAnalysisInteractivePlot()

                # Make the LRFS and copy the necessary fields across
                lrfs = self.LRFS(pulse_range=self.ax.get_ylim())

                self.lrfs.stokes      = lrfs.stokes
                self.lrfs.npulses     = lrfs.npulses
                self.lrfs.nbins       = lrfs.nbins
                self.lrfs.first_pulse = lrfs.first_pulse
                self.lrfs.first_phase = lrfs.first_phase
                self.lrfs.dpulse      = lrfs.dpulse
                self.lrfs.dphase_deg  = lrfs.dphase_deg
                self.lrfs.complex     = lrfs.complex
                self.lrfs.xlabel      = lrfs.xlabel
                self.lrfs.ylabel      = lrfs.ylabel
                self.lrfs.values      = lrfs.values

                # Remove all the subpulses, models, and drift sequence boundaries
                self.lrfs.subpulses = Subpulses()
                self.lrfs.model_fits = {}
                self.lrfs.drift_sequences = DriftSequences()

                # Make it interactive!
                self.lrfs.start()

            elif event.key == "A":
                # Start a new instance of DriftAnalysisInteractivePlot for the auto correlation
                self.ac = DriftAnalysisInteractivePlot()

                # Actually do the cross correlation and "copy" it across to this instance
                autocorr = self.auto_correlate_pulses()
                self.ac.stokes      = autocorr.stokes
                self.ac.npulses     = autocorr.npulses
                self.ac.nbins       = autocorr.nbins
                self.ac.first_pulse = autocorr.first_pulse
                self.ac.first_phase = autocorr.first_phase
                self.ac.dpulse      = autocorr.dpulse
                self.ac.dphase_deg  = autocorr.dphase_deg
                self.ac.complex     = autocorr.complex
                self.ac.xlabel      = autocorr.xlabel
                self.ac.ylabel      = autocorr.ylabel
                self.ac.values      = autocorr.values

                # Remove all the subpulses and models
                self.ac.subpulses = Subpulses()
                self.ac.model_fits = {}
                # ... but keep the drift sequence boundaries
                self.ac.drift_sequences = copy.copy(self.drift_sequences)

                # Make it interactive!
                self.ac.start()

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
                    pulse_range = self.model_fits[seq].get_pulse_bounds()
                    pulse_idx_range = self.get_pulse_bin(pulse_range)
                    pulse_idxs = np.arange(pulse_idx_range[0], pulse_idx_range[1] + 1)
                    pulses     = self.get_pulse_from_bin(pulse_idxs)
                    driftrates = self.model_fits[seq].calc_driftrate(pulses)
                    dr_ax.plot(pulses, driftrates, 'k')
                dr_ax.set_xlabel("Pulse number")
                dr_ax.set_ylabel("Drift rate (deg/pulse)")
                dr_fig.show()

            elif event.key == "3":
                P3_fig, P3_ax = plt.subplots()
                for seq in self.model_fits:
                    pulse_range = self.model_fits[seq].get_pulse_bounds()
                    pulse_idx_range = self.get_pulse_bin(pulse_range)
                    pulse_idxs = np.arange(pulse_idx_range[0], pulse_idx_range[1] + 1)
                    pulses     = self.get_pulse_from_bin(pulse_idxs)
                    P3s        = np.abs(self.model_fits[seq].calc_P3(pulses))
                    P3_ax.plot(pulses, P3s, 'k')
                P3_ax.set_xlabel("Pulse number")
                P3_ax.set_ylabel("$P_3/P_1$")
                P3_fig.show()

            elif event.key == "&":
                dr_fig, dr_ax = plt.subplots()
                for seq in self.model_fits:
                    pulse_range = self.model_fits[seq].get_pulse_bounds()
                    pulse_idx_range = self.get_pulse_bin(pulse_range)
                    pulse_idxs = np.arange(pulse_idx_range[0], pulse_idx_range[1] + 1)
                    pulses     = self.get_pulse_from_bin(pulse_idxs)
                    decayrates = self.model_fits[seq].calc_driftrate_decay_rate(pulses)
                    dr_ax.plot(pulses, decayrates, 'k')
                dr_ax.set_xlabel("Pulse number")
                dr_ax.set_ylabel("Driftrate decay rate (/pulse)")
                dr_fig.show()

            elif event.key == "%":
                dr_fig = plt.figure()
                dr_ax = plt.axes(projection='3d')
                p  = [] # Pulse number
                d  = [] # Drift rate
                dd = [] # Derivative of drift rate (w.r.t. pulse number)
                for seq in self.model_fits:
                    p_lo, p_hi = self.model_fits[seq].get_pulse_bounds()
                    p.append(0.5*(p_lo + p_hi))
                    d.append(self.model_fits[seq].calc_driftrate(p[-1]))
                    dd.append(self.model_fits[seq].calc_driftrate_derivative(p[-1]))
                    dr_ax.plot(p, d, dd, 'ro')
                dr_ax.set_xlabel("Pulse number")
                dr_ax.set_ylabel("Drift rate (deg/pulse)")
                dr_ax.set_zlabel("Drift rate derivative (deg/pulse^2)")
                dr_fig.show()

            elif event.key in ["*", "("]:
                # Define what model is going to be plotted here
                # (Use a dummy object to get the necessary parameters)
                dummy = ModelFit()
                if event.key == "*":
                    dummy.model_name = "exponential"
                elif event.key == "(":
                    dummy.model_name = "quadratic"
                nparameters = dummy.get_nparameters()
                param_names = dummy.get_parameter_names(display_type='latex')

                param_fig, param_axs = plt.subplots(nrows=nparameters, ncols=1, sharex=True)

                pmid       = [] # Central pulse number
                perr       = [] # X error bar represents size of drift sequence
                params     = [] # The parameter values
                param_errs = [] # The errors on the parameter values

                for seq in self.model_fits:

                    # Only plot up exponential parameters
                    if self.model_fits[seq].model_name != dummy.model_name:
                        continue

                    # Get the mid pulse and range of the drift sequences
                    p_lo, p_hi = self.model_fits[seq].get_pulse_bounds()
                    pmid.append(0.5*(p_hi + p_lo))
                    perr.append(0.5*(p_hi - p_lo))

                    # Get the parameter values
                    params.append(self.model_fits[seq].parameters)
                    if self.model_fits[seq].pcov is not None:
                        param_errs.append(list(np.sqrt(np.diag(self.model_fits[seq].pcov))))
                    else:
                        param_errs.append(np.zeros((nparameters,)))

                # Convert the params and param_errs list-of-lists to Numpy arrays
                params     = np.array(params)
                param_errs = np.array(param_errs)

                # Plot everything up!
                for i in range(nparameters):
                    param_axs[i].errorbar(pmid, params[:,i], xerr=perr, yerr=param_errs[:,i], fmt='.')
                    param_axs[i].set_ylabel("$" + param_names[i] + "$")
                param_axs[-1].set_xlabel("Pulse number")

                param_fig.show()

            elif event.key == "E":
                self.ax.set_title("Select a drift sequence by clicking on the pulsestack.\nPress enter to confirm, esc to cancel.")
                self.fig.canvas.draw()
                self.mode = "switch_to_exponential_and_solve"

            elif event.key == "m":
                    self.ax.set_title("Select a drift sequence by clicking on the pulsestack.\nPress enter to confirm, esc to cancel.")
                    self.fig.canvas.draw()
                    self.mode = "display_model_details"

        ########################################
        # SPECIALISED KEYS FOR DIFFERENT MODES #
        ########################################

        elif self.mode == "toggle_visibility":
            if event.key == ".":
                # If any kind of subpulse is visible, turn off visibility to all
                if self.subpulses.with_driftbands_plt is not None or self.subpulses.no_driftbands_plt is not None:
                    self.subpulses.clear_plots()
                else:
                    self.subpulses.plot_subpulses(self.ax)
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
                self.maxima_plt.set_data([], [])
                self.subpulses.plot_subpulses(self.ax)

                if self.jsonfile is not None:
                    self.fig.canvas.manager.set_window_title(self.jsonfile + "*")

                self.set_default_mode()
            elif event.key == "escape":
                self.threshold_line.set_data([], [])
                self.maxima_plt.set_data([], [])
                self.subpulses.plot_subpulses(self.ax)
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
                    self.subpulses.plot_subpulses(self.ax)

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
                    self.subpulses.plot_subpulses(self.ax)
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
                        self.model_fits[seq].plot_all_driftbands(self.ax, phlim, pstep=self.dpulse, color='k')

                        self.model_fits[seq+1].clear_all_plots()
                        self.model_fits[seq+1].plot_all_driftbands(self.ax, phlim, pstep=self.dpulse, color='k')

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
                first_pulse_idx, last_pulse_idx = self.drift_sequences.get_bounding_pulse_idxs(self.selected, self.npulses)

                # Set the zoom (only in y-direction)
                ylo = self.get_pulse_from_bin(first_pulse_idx - 0.5)
                yhi = self.get_pulse_from_bin(last_pulse_idx + 0.5)
                self.ax.set_ylim([ylo, yhi])

                self.deselect()
                self.set_default_mode()

            elif event.key == "escape":
                self.deselect()
                self.set_default_mode()

        elif self.mode == "switch_to_quadratic_and_solve":
            if event.key == "enter":
                # Here, self.selected is the drift sequence idx
                if self.selected is None:
                    return

                # Convert to quadratic model
                self.model_fits[self.selected].convert_to_model("quadratic")
                pulse_range = self.model_fits[self.selected].get_pulse_bounds()
                subset = self.subpulses.in_pulse_range(pulse_range, with_valid_driftband=True)

                ph = self.subpulses.get_phases(subset=subset)
                p  = self.subpulses.get_pulses(subset=subset)
                d  = self.subpulses.get_driftbands(subset=subset)
                self.model_fits[self.selected].optimise_fit_to_subpulses(ph, p, d)

                # Update the plot
                if self.onpulse is None:
                    phlim = self.get_phase_from_bin(np.array([0, self.nbins-1]))
                else:
                    phlim = self.onpulse

                self.model_fits[self.selected].plot_all_driftbands(self.ax, phlim, pstep=self.dpulse, color='k')

                # Mark that unsaved changes have been made
                if self.jsonfile is not None:
                    self.fig.canvas.manager.set_window_title(self.jsonfile + "*")

                self.deselect()
                self.set_default_mode()

            elif event.key == "escape":
                self.deselect()
                self.set_default_mode()

        elif self.mode == "switch_to_exponential_and_solve":
            if event.key == "enter":
                # Here, self.selected is the drift sequence idx
                if self.selected is None:
                    return

                # Convert to exponential model
                self.model_fits[self.selected].convert_to_model("exponential")
                pulse_range = self.model_fits[self.selected].get_pulse_bounds()
                subset = self.subpulses.in_pulse_range(pulse_range, with_valid_driftband=True)

                ph = self.subpulses.get_phases(subset=subset)
                p  = self.subpulses.get_pulses(subset=subset)
                d  = self.subpulses.get_driftbands(subset=subset)
                self.model_fits[self.selected].optimise_fit_to_subpulses(ph, p, d)

                # Update the plot
                if self.onpulse is None:
                    phlim = self.get_phase_from_bin(np.array([0, self.nbins-1]))
                else:
                    phlim = self.onpulse

                self.model_fits[self.selected].plot_all_driftbands(self.ax, phlim, pstep=self.dpulse, color='k')

                # Mark that unsaved changes have been made
                if self.jsonfile is not None:
                    self.fig.canvas.manager.set_window_title(self.jsonfile + "*")

                self.deselect()
                self.set_default_mode()

            elif event.key == "escape":
                self.deselect()
                self.set_default_mode()


        elif self.mode == "assign_driftbands":
            if event.key == "enter":
                # Here, self.selected is the drift sequence idx
                if self.selected is None:
                    return

                # Assign each subpulse in sequence to the nearest driftband
                subset = self.subpulses.assign_driftbands_to_subpulses(self.model_fits[self.selected])

                # Replot subpulses to reflect change in status
                self.subpulses.plot_subpulses(self.ax)

                # Mark that unsaved changes have been made
                if self.jsonfile is not None:
                    self.fig.canvas.manager.set_window_title(self.jsonfile + "*")

                self.deselect()
                self.set_default_mode()

            elif event.key == "escape":
                self.deselect()
                self.set_default_mode()

        elif self.mode == "plot_residuals":
            if event.key == "enter":
                # Here, self.selected is the drift sequence idx
                if self.selected is None:
                    return

                # Calculate residuals
                pulse_range = self.model_fits[self.selected].get_pulse_bounds()
                subset = self.subpulses.in_pulse_range(pulse_range, with_valid_driftband=True)

                ph = self.subpulses.get_phases(subset=subset)
                p  = self.subpulses.get_pulses(subset=subset)
                d  = self.subpulses.get_driftbands(subset=subset)

                residuals = self.model_fits[self.selected].calc_residual(p, ph, d)

                # Plot them both as a function of pulse number and phase
                res_fig, res_axs = plt.subplots(nrows=2, ncols=1)

                res_axs[0].axhline(0, linestyle='--', color='k')
                res_axs[0].plot(p, residuals, '.')
                res_axs[0].set_xlabel("Pulse number")
                res_axs[0].set_ylabel("Residual phase (deg)")

                res_axs[1].axhline(0, linestyle='--', color='k')
                res_axs[1].plot(ph, residuals, '.')
                res_axs[1].set_xlabel("Subpulse phase (deg)")
                res_axs[1].set_ylabel("Residual phase (deg)")

                res_fig.show()

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
                    self.candidate_quadratic_model.optimise_fit_to_subpulses(ph, p, d)
                    pulse_idx_range = self.drift_sequences.get_bounding_pulse_idxs(self.drift_sequence_selected, self.npulses)
                    first_pulse, last_pulse = self.get_pulse_from_bin(np.array(pulse_idx_range))
                    self.candidate_quadratic_model.set_pulse_bounds(first_pulse, last_pulse)

                    # Only plot the driftbands in the on pulse region
                    if self.onpulse is None:
                        phlim = self.get_phase_from_bin(np.array([0, self.nbins-1]))
                    else:
                        phlim = self.onpulse

                    # Clear all previous driftbands from the plot and replace them with new ones!
                    self.candidate_quadratic_model.clear_all_plots()
                    self.candidate_quadratic_model.plot_all_driftbands(self.ax, phlim, pstep=self.dpulse, color='w')
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
                    self.model_fits[self.drift_sequence_selected].plot_all_driftbands(self.ax, phlim, pstep=self.dpulse, color='k')

                if self.quadratic_selected_plt is not None:
                    self.quadratic_selected_plt[0].set_data([], [])
                    self.quadratic_selected_plt = None

                self.deselect()
                self.set_default_mode()

        elif self.mode == "display_model_details":

            if event.key == "enter":
                print(self.model_fits[self.selected])
                self.deselect()
                self.set_default_mode()

            elif event.key == "escape":
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
        self.fig, self.ax = plt.subplots()
        self.plot_image(ax=self.ax)

        xlim = self.ax.get_xlim()
        ylim = self.ax.get_ylim()

        self.subpulses.plot_subpulses(self.ax)
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
        self.fig.show()



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
    plt.show()


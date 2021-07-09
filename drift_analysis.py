import sys
import numpy as np
import matplotlib.pyplot as plt
from numpy.polynomial.polynomial import polyfit, polyval
from scipy.interpolate import interp1d
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

    def get_phases(self):
        return self.data[:,0]

    def get_pulses(self):
        return self.data[:,1]

    def get_widths(self):
        return self.data[:,2]

    def get_driftbands(self):
        return self.data[:,3]

    def get_positions(self):
        '''
        Returns an Nx2 numpy array of subpulse positions (phase, pulse)
        '''
        return self.data[:,:2]

    def set_phases(self, phases):
        self.data[:,0] = phases

    def set_pulses(self, pulses):
        self.data[:,1] = pulses

    def set_widths(self, widths):
        self.data[:,2] = widths

    def set_driftbands(self, driftbands):
        self.data[:,3] = driftbands

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

class DriftSequences:
    def __init__(self, ps):
        self.ps         = ps  # Reference to pulsestack
        self.boundaries = []  # Contains indexes of pulse numbers preceding the boundary

    def has_boundary(self, pulse_idx):
        if pulse_idx in self.boundaries:
            return True
        else:
            return False

    def add_boundary(self, pulse_idx):
        if not self.has_boundary(pulse_idx):
            bisect.insort(self.boundaries, pulse_idx)

    def delete_boundaries(self, boundary_idxs):
        self.boundaries = np.delete(self.boundaries, boundary_idxs)

    def get_bounding_pulse_idxs(self, sequence_idx):
        '''
        Returns the first and last pulses indexes in this sequence
        '''
        if sequence_idx < 0:
            sequence_idx = len(self.boundaries) + 1 + sequence_idx

        if sequence_idx == 0:
            first_idx = 0
        else:
            first_idx = self.boundaries[sequence_idx - 1] + 1

        if sequence_idx == len(self.boundaries) + 1:
            last_idx = self.ps.npulses - 1
        else:
            last_idx = self.boundaries[sequence_idx]

        return first_idx, last_idx

    def get_pulses(self, boundary_idxs=None):
        if boundary_idxs is None:
            mid_bins = np.array(self.boundaries) + 0.5
        else:
            mid_bins = np.array(self.boundaries)[boundary_idxs] + 0.5
        return self.ps.get_pulse_from_bin(mid_bins)

class DriftAnalysis(pulsestack.Pulsestack):
    def __init__(self):
        self.fig = None
        self.ax  = None
        self.subpulses = Subpulses()
        self.subpulses_plt = None
        self.subpulses_fmt = 'gx'
        self.maxima_threshold = 0.0
        self.drift_sequences = DriftSequences(super())
        self.dm_boundary_plt = None
        self.jsonfile = None

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
                "pdvfile":             self.pdvfile,
                "stokes":              self.stokes,
                "npulses":             self.npulses,
                "nbins":               self.nbins,
                "first_pulse":         self.first_pulse,
                "first_phase":         self.first_phase,
                "dpulse":              self.dpulse,
                "dphase_deg":          self.dphase_deg,

                "subpulses_pulse":     list(self.subpulses.get_pulses()),
                "subpulses_phase":     list(self.subpulses.get_phases()),
                "subpulses_width":     list(self.subpulses.get_widths()),
                "subpulses_driftband": list(self.subpulses.get_driftbands()),

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

        self.pdvfile          = drift_dict["pdvfile"]
        self.stokes           = drift_dict["stokes"]
        self.npulses          = drift_dict["npulses"]
        self.nbins            = drift_dict["nbins"]
        self.first_pulse      = drift_dict["first_pulse"]
        self.first_phase      = drift_dict["first_phase"]
        self.dpulse           = drift_dict["dpulse"]
        self.dphase_deg       = drift_dict["dphase_deg"]

        subpulses_pulse       = drift_dict["subpulses_pulse"]
        subpulses_phase       = drift_dict["subpulses_phase"]
        subpulses_width       = drift_dict["subpulses_width"]
        subpulses_driftband   = drift_dict["subpulses_driftband"]

        self.subpulses.add_subpulses(subpulses_phase, subpulses_pulse, subpulses_width, subpulses_driftband)

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

        ys = self.drift_sequences.get_pulses()
        if self.dm_boundary_plt is not None:
            segments = np.array([[[xlo, y], [xhi, y]] for y in ys])
            self.dm_boundary_plt.set_segments(segments)
        else:
            self.dm_boundary_plt = self.ax.hlines(ys, xlo, xhi, colors=["k"], linestyles='dashed')

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
        if self.mode == "delete_subpulse":
            idx, dist = self.closest_subpulse(event.x, event.y)

            if dist > 10: # i.e. if mouse click is more than 10 pixels away from the nearest point
                self.selected = None
            else:
                self.selected = idx

            if self.selected is not None:
                if self.selected_plt is None:
                    self.selected_plt, = self.ax.plot([self.subpulses.get_phases()[self.selected]], [self.subpulses.get_pulses()[self.selected]], 'wo')
                else:
                    self.selected_plt.set_data([self.subpulses.get_phases()[self.selected]], [self.subpulses.get_pulses()[self.selected]])
            else:
                if self.selected_plt is not None:
                    self.selected_plt.set_data([], [])
            
            if self.selected_plt is not None:
                self.fig.canvas.draw()

        elif self.mode == "delete_drift_mode_boundary":
            idx, dist = self.closest_drift_mode_boundary(event.y)

            if dist > 10: # i.e. if mouse click is more than 10 pixels away from the nearest point
                self.selected = None
            else:
                self.selected = idx

            if self.selected is not None:
                y = self.drift_sequences.get_pulses([self.selected])
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
            y = self.drift_sequences.ps.get_pulse_from_bin(pulse_idx + 0.5)

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

                # Go back to default mode
                self.set_default_mode()

    def set_default_mode(self):
        self.ax.set_title("Press (capital) 'H' for command list")
        self.fig.canvas.draw()
        self.mode = "default"

    def on_key_press_event(self, event):

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
                print("C     Crop pulsestack to current visible image")
                print(".     Add a subpulse")
                print(">     Delete a subpulse")
                print("P     Plot the profile of the current view")
                print("T     Plot the LRFS of the current view")
                print("/     Add a drift mode boundary")
                print("?     Delete a drift mode boundary")
                print("v     Toggle visibility of plot feature")
                print("D     Set view to selected drift sequence")

            elif event.key == "j":
                self.save_json()

                if self.fig is not None:
                    self.fig.canvas.set_window_title(self.jsonfile)


            elif event.key == "J":
                old_jsonfile = self.jsonfile
                self.jsonfile = None
                self.save_json()
                if self.jsonfile is None:
                    self.jsonfile = old_jsonfile
                if self.fig is not None:
                    self.fig.canvas.set_window_title(self.jsonfile)

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
                self.ax.set_title("Toggle visibility mode. Press escape when finished.\nsubpulses (.), drift mode boundaries (/)")
                self.fig.canvas.draw()
                self.mode = "toggle_visibility"

        elif self.mode == "toggle_visibility":
            if event.key == ".":
                if self.subpulses_plt is not None:
                    self.subpulses_plt.set_data([], [])
                    self.subpulses_plt = None
                else:
                    self.plot_subpulses()
                self.fig.canvas.draw()

            if event.key == "/":
                if self.dm_boundary_plt is not None:
                    self.dm_boundary_plt.set_segments(np.empty((0,2)))
                    self.dm_boundary_plt = None
                else:
                    self.plot_drift_mode_boundaries()
                self.fig.canvas.draw()

            elif event.key == "escape":
                self.set_default_mode()

        elif self.mode == "set_threshold":
            if event.key == "enter":
                self.threshold_line.set_data([], [])
                self.subpulses.delete_all_subpulses()
                self.subpulses.add_subpulses(self.max_locations[1], self.max_locations[0])
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
                    self.fig.canvas.draw()

            elif event.key == "escape":
                self.deselect()
                self.set_default_mode()

        elif self.mode == "delete_drift_mode_boundary":
            if event.key == "enter":
                if self.selected is not None:
                    # Delete the selected boundary from the actual list
                    # Here, "selected" refers to the idx of drift_mode_boundaries[]
                    self.drift_sequences.delete_boundaries([self.selected])

                    # Delete the boundary line from the plot
                    self.plot_drift_mode_boundaries()

                    # Deselect
                    self.deselect()

                    # Redraw the figure
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
                self.fig.canvas.draw()

            elif event.key == "escape":
                self.deselect()
                self.set_default_mode()

        elif self.mode == "add_drift_mode_boundary":
            if event.key == "enter":
                if self.selected is None:
                    return

                # Here, "selected" refers to the pulse_idx of the drift mode boundary
                self.drift_sequences.add_boundary(self.selected)

                xlim = self.ax.get_xlim()
                ylim = self.ax.get_ylim()

                self.plot_drift_mode_boundaries()
                self.deselect()

                self.ax.set_xlim(xlim)
                self.ax.set_ylim(ylim)

                self.fig.canvas.draw()

            elif event.key == "escape":
                self.deselect()
                self.set_default_mode()

    def closest_drift_mode_boundary(self, y):
        dm_boundary_display = self.ax.transData.transform([[0,y] for y in self.drift_sequences.get_pulses()])
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

        self.ax.set_xlim(xlim)
        self.ax.set_ylim(ylim)

        # Set the mode to "default"
        self.set_default_mode()

        # Make it interactive!
        self.cid = self.fig.canvas.mpl_connect('button_press_event', self.on_button_press_event)
        self.cid = self.fig.canvas.mpl_connect('key_press_event', self.on_key_press_event)

        # Set the window title to the json filename
        if self.jsonfile is not None:
            self.fig.canvas.set_window_title(self.jsonfile)
        else:
            self.fig.canvas.set_window_title("[Unsaved pulsestack]")

        # Show the plot
        plt.show()


'''
# Manually flag some outliers
# These indices depend on the exact max_locations already obtained, so any changes to the
# code above this point MAY require this flagging to be redone.
ml = np.array(max_locations)
#for i in range(ml.shape[1]):
#    print("{} {} {}".format(i, ml[0,i], ml[1,i]))
max_locations = np.delete(max_locations, [23, 52, 77, 125, 132, 149, 153, 157, 179, 276, 350, 396, 399, 510], axis=-1)

# Manually set the driftband boundaries.
# A boundary is defined as a line drawn on an array of the same dimensions as cropped_pulsestack.
# The driftband_boundary_points is a list of pairs of points [[x1,y1],[x2,y2]],
# which is easy to fill in by hovering the mouse over a matplotlib plot of cropped_pulsestack.
# These points are later turned into lines (slope,y-intercept) programmatically.
# These points can be quite rough -- it's only important that they "capture" all the points
# into the correct drift band classicifation. It IS important, however, that these boundary
# lines are given in the "right" order.
driftband_boundary_points = np.array([
        [[73,-15], [18,8]],
        [[73,0], [18,23]],
        [[80,15], [20,41]],
        [[84,32], [21,62]],
        [[83,52], [25,84]],
        [[107,68], [22,110]],
        [[109,87], [16,121]],
        [[85,113], [39,121]],
        [[104,123], [27,132]],
        [[59,135], [13,167]],
        [[90,138], [20,192]],
        [[90,179], [18,232]],
        [[86,220], [21,261]],
        [[94,244], [27,283]],
        [[94,270], [27,307]],
        [[44,317], [17,328]],
        [[87,314], [11,340]],
        [[78,330], [22,345]],
        [[71,342], [20,353]],
        [[99,355], [23,370]],
        [[59,374], [23,382]],
        [[76,382], [26,404]],
        [[88,395], [16,437]],
        [[89,420], [6,479]],
        [[88,449], [12,505]],
        [[86,487], [23,534]],
        [[88,531], [74,539]] ])

x1 = driftband_boundary_points[:,0,0]
x2 = driftband_boundary_points[:,1,0]
y1 = driftband_boundary_points[:,0,1]
y2 = driftband_boundary_points[:,1,1]

# The slopes (m) and y-intercepts (c) of the drift band boundary lines
db_m = (y2-y1)/(x2-x1)
db_c = y1 - x1*db_m

assigned_driftbands = []
for max_location in np.transpose(max_locations):
    x = max_location[1]
    y = max_location[0]
    assigned_driftbands.append(np.where(y < db_m*x + db_c)[0][0])

# Ensure that the driftband numbers start at zero
assigned_driftbands = np.array(assigned_driftbands)
assigned_driftbands -= np.min(assigned_driftbands)

# Organise the max_locations into a list of driftbands
ndriftbands = np.max(assigned_driftbands) + 1
# Set up a list of "ndriftbands" empty lists
driftband_points = []
for i in range(ndriftbands):
    driftband_points.append([])
# Go through assigned driftbands and organise the max_locations
for i in range(len(assigned_driftbands)):
    x = max_locations[1][i]
    y = max_locations[0][i]
    driftband_points[assigned_driftbands[i]].append([x, y])
db_xys = [np.transpose(driftband_points[i]) for i in range(len(driftband_points))]
#print(driftband_points)

# Fit lines to each driftband
linear_fits = [polyfit(db_xys[i][0,:], db_xys[i][1,:], 1) for i in range(len(db_xys))]
# One of the driftbands (18) has only one point in it. Set its slope equal to the slope of
# the following driftband.
db = 19
linear_fits[db][1] = linear_fits[db+1][1]
linear_fits[db][0] = driftband_points[db][0][1] - driftband_points[db][0][0]*linear_fits[db][1]
driftrates = dph_deg/np.array(linear_fits)[:,1] # deg per pulse

# Get "pulse-intercept" of each driftband fit with a "fiducial x" (xref)
xref = 55
prefs = np.array([polyval(xref, linear_fits[i]) for i in range(len(linear_fits))])

# Turn these into nominal "P3-phases" by interpolating between driftband number and prefs
P3phase_interp = interp1d(prefs, np.arange(ndriftbands), kind='linear', fill_value="extrapolate")
#print(ml.shape)
nonempty_pulses = np.unique(ml[0,:])
P3phases = P3phase_interp(nonempty_pulses)

# For each driftband, get a P3 and an "inverse driftrate"
# The slope between the two should be the "invariant" P2
P3 = np.diff(prefs)
inv_dr = 1/driftrates[:-1]
# Remove ones near drift sequence boundaries (and other pathological cases)
#print(P3)
#print(inv_dr)
delete_idxs = [7, 18, 19]
P3 = np.delete(P3, delete_idxs)
inv_dr = np.delete(inv_dr, delete_idxs)

P2 = np.dot(P3, P3)/(P3.T @ inv_dr)
'''


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


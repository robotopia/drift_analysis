import numpy as np
import matplotlib.pyplot as plt
from numpy.polynomial.polynomial import polyfit, polyval
from scipy.interpolate import interp1d
import tkinter
import tkinter.filedialog
import pulsestack

class DriftAnalysis(pulsestack.Pulsestack):
    def __init__(self):
        self.max_locations = np.array([])
        self.maxima_plt = None
        self.maxima_fmt = 'gx'

    def get_local_maxima(self, min_value=None):
        is_bigger_than_left  = self.values[:,1:-1] >= self.values[:,:-2]
        is_bigger_than_right = self.values[:,1:-1] >= self.values[:,2:]
        is_local_max = np.logical_and(is_bigger_than_left, is_bigger_than_right)

        if min_value is not None:
            is_local_max = np.logical_and(is_local_max, self.values[:,1:-1] > min_value)

        self.max_locations = np.array(np.where(is_local_max)).astype(float)

        # Add one to phase (bin) locations because of previous splicing
        self.max_locations[1,:] += 1

        # Convert locations to data coordinates (pulse and phase)
        self.max_locations[0,:] = self.max_locations[0,:]*self.dpulse + self.first_pulse
        self.max_locations[1,:] = self.max_locations[1,:]*self.dphase_deg + self.first_phase

    def save_maxima(self, outfile):
        np.savetxt(outfile, np.transpose(self.max_locations), header="phase_(deg) pulse_number")

    def load_maxima(self, infile):
        try:
            dat = np.transpose(np.loadtxt(infile))
        except:
            print("Error opening {} with NumPy loadtxt()".format(infile))
            return

        if dat.shape[0] != 2:
            print("Error opening {}. It must have two columns of data")
            return

        self.max_locations = np.transpose(np.loadtxt(infile))

    def plot_maxima(self, **kwargs):
        if self.maxima_plt is None:
            self.maxima_plt, = self.ax.plot(self.max_locations[1], self.max_locations[0], self.maxima_fmt, **kwargs)
        else:
            self.maxima_plt.set_data(self.max_locations[1,:], self.max_locations[0,:])

class DriftAnalysisInteractivePlot(DriftAnalysis):
    def __init__(self):
        super(DriftAnalysisInteractivePlot, self).__init__()
        self.mode         = "pulsestack"
        self.selected     = None
        self.selected_plt = None

        self.smoothed_ps  = None
        self.show_smooth  = False

    def on_button_press_event(self, event):
        if self.mode == "maxima":
            idx, dist = self.closest_maximum(event.x, event.y)

            if dist > 10: # i.e. if mouse click is more than 10 pixels away from the nearest point
                self.selected = None
            else:
                self.selected = idx

            if self.selected is not None:
                if self.selected_plt is None:
                    self.selected_plt, = self.ax.plot([self.max_locations[1,self.selected]], [self.max_locations[0,self.selected]], 'wo')
                else:
                    self.selected_plt.set_data([self.max_locations[1,self.selected]], [self.max_locations[0,self.selected]])
            else:
                if self.selected_plt is not None:
                    self.selected_plt.set_data([], [])
            
            if self.selected_plt is not None:
                self.selected_plt.figure.canvas.draw()

    def on_key_press_event(self, event):
        '''
        m - toggle mode
        In "maxima" mode:
            d - delete point selected with mouse
            S - save all points to file
            O - load points from file
            c - clear (delete all points)
        In "pulsestack" mode:
            S - toggle smoothed pulsestack
            ^ - find local peaks
        '''
        if event.key == "m":
            if self.mode == "maxima":
                self.mode = "pulsestack"
            elif self.mode == "pulsestack":
                self.mode = "maxima"
            return

        if self.mode == "maxima":
            # 'd' = delete selected point
            if event.key == "d":
                if self.selected is not None:
                    # Delete the selected point from the actual list
                    self.max_locations = np.delete(self.max_locations, self.selected, axis=-1)

                    # Delete the point from the plot
                    self.plot_maxima()
                    self.selected_plt.set_data([], [])

                    # Unselect
                    self.selected = None

                    # Redraw the figure
                    self.selected_plt.figure.canvas.draw()

            # 'S' = save all points from file
            if event.key == "S":
                root = tkinter.Tk()
                root.withdraw()
                filename = tkinter.filedialog.asksaveasfilename(filetypes=(("All files", "*.*"),))
                if filename:
                    self.save_maxima(filename)

            # 'O' = Load points from file
            if event.key == "O":
                root = tkinter.Tk()
                root.withdraw()
                filename = tkinter.filedialog.askopenfilename(filetypes=(("All files", "*.*"),))
                if filename:
                    self.load_maxima(filename)
                    self.plot_maxima()
                    self.maxima_plt.figure.canvas.draw()

            # 'c' = clear (delete all points)
            if event.key == 'c':
                self.max_locations = np.array([])
                if self.maxima_plt is not None:
                    self.maxima_plt.set_data([], [])
                    self.maxima_plt.figure.canvas.draw()

        elif self.mode == "pulsestack":
            # 'S' = toggle smooth pulsestack
            if event.key == "S":
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
                        self.ps_image.figure.canvas.draw()

                else:
                    self.ps_image.set_data(self.values)
                    self.show_smooth = False
                    # Update the colorbar
                    self.cbar.update_normal(self.ps_image)
                    self.ps_image.figure.canvas.draw()

            if event.key == "^":
                root = tkinter.Tk()
                root.withdraw()
                min_value = tkinter.simpledialog.askfloat("Min threshold", "Input minimum threshold for maxima", parent=root)
                if self.show_smooth == True:
                    self.smoothed_ps.get_local_maxima(min_value=min_value)
                    self.max_locations = self.smoothed_ps.max_locations
                else:
                    self.get_local_maxima(min_value=min_value)
                self.plot_maxima()
                self.maxima_plt.figure.canvas.draw()

    def closest_maximum(self, x, y):
        max_locations_display = self.ax.transData.transform(np.transpose(np.flip(self.max_locations, axis=0)))
        dists = np.hypot(x - max_locations_display[:,0], y - max_locations_display[:,1])
        idx = np.argmin(dists)
        return idx, dists[idx]

    def start(self):
        '''
        Start the interactive plot
        '''

        # Make it interactive!
        self.plot_image()
        self.cid = self.ps_image.figure.canvas.mpl_connect('button_press_event', self.on_button_press_event)
        self.cid = self.ps_image.figure.canvas.mpl_connect('key_press_event', self.on_key_press_event)


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

############
# PLOTTING #
############

plt.figure(figsize=(10,40))

# Show where the drift band boundaries are
for p in driftband_boundary_points:
    plt.plot(p[:,0], p[:,1], 'k')

# Show the linear fits to the driftbands
x = np.array([10, 90])
for lfit in linear_fits:
    plt.plot(x, polyval(x, lfit), 'y--')

# Stacked pulses (lines)
#for pulse in pulses:
#    plt.plot(pulsestack[pulse]*5+pulse)

# Profile:
#plt.plot(np.sum(pulsestack, axis=0))

plt.xlabel("Rotation phase (in pixels)")
plt.ylabel("Pulse number")
plt.ylim([-0.5,npulses-0.5])
plt.tight_layout()
plt.savefig("1274143152_linfits.png")

# FIGURE 2
plt.figure(2)
#plt.clf()

# Plot P3 vs inverse driftrate
plt.plot(P3, inv_dr, 'o')
plotP3 = np.array([np.min(P3), np.max(P3)])
ploty = plotP3/P2
plt.plot(plotP3, ploty, 'k--')
plt.xlabel("P_3 (P_1)")
plt.ylabel("1/Driftrate (pulses per deg)")
plt.title("P_2 = {:.1f} deg".format(np.abs(P2)))

# Plot P3phases
#plt.plot(nonempty_pulses, P3phases, '-o')
#plt.xlabel("Pulse number")
#plt.ylabel("P3 phase")

plt.savefig("1274143152_P3_vs_dr.png")

# FIGURE 3
plt.figure(3)

# Plot drift rates against pulse number
plt.plot(prefs, driftrates, 'o')
plt.xlabel("Pulse number of driftband intercept through x={}".format(xref))
plt.ylabel("Drift rate (deg per pulse)")

plt.savefig("1274143152_dr_over_time.png")

# FIGURE 3
plt.figure(4)

# Plot est. P3 against pulse number
plt.plot(prefs, P2/driftrates, 'o')
plt.xlabel("Pulse number of driftband intercept through x={}".format(xref))
plt.ylabel("P_3 (P_1)")

plt.savefig("1274143152_P3_over_time.png")

#plt.show()
'''

if __name__ == '__main__':
    # Load the data
    ps = DriftAnalysisInteractivePlot()
    ps.load_from_pdv('1274143152_J0024-1932.F.pdv', 'I')
    ps.set_fiducial_phase(131.836)

    # Characterise the noise (in an off-pulse region)
    off_pulse_ps = ps.crop(phase_deg_range=[None, -20], inplace=False)
    sigma = np.std(off_pulse_ps.values)

    # Crop to just the central 40 deg
    ps.crop(phase_deg_range=[-20, 20])

    '''
    # Smooth pulses with a Gaussian filter
    kernel_sigma = 1.75 # deg
    smoothed_ps = ps.smooth_with_gaussian(kernel_sigma, inplace=False)

    # Find the local maxima
    smoothed_ps.get_local_maxima(min_value=0.8*sigma)

    '''
    ps.start()
    #smoothed_ps.plot_maxima(fmt='gx', ax=ps.ax)
    plt.show()


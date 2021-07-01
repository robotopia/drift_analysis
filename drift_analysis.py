import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter1d

dat = np.loadtxt('1274143152_J0024-1932.F.pdv')

npulses = int(dat[-1,0] + 1)
nbins   = int(dat[-1,2] + 1)

pulsestack = np.reshape(dat[:,3], (npulses,nbins))

pulses = np.arange(npulses)

cropped_pulsestack = pulsestack[:,320:436]

# Characterise the noise (in an off-pulse region)
sigma = np.std(pulsestack[:,0:320])

# Smooth with a gaussian filter
stdev = 5 # pixels
smoothed_pulsestack = gaussian_filter1d(cropped_pulsestack, stdev, mode='wrap')

# Get ALL the local maxima
lshifted = np.roll(smoothed_pulsestack, -1, axis=-1)
rshifted = np.roll(smoothed_pulsestack,  1, axis=-1)
is_local_max = np.logical_and(smoothed_pulsestack > lshifted, smoothed_pulsestack > rshifted)
is_max_and_bright = np.logical_and(is_local_max, smoothed_pulsestack > 0.8*sigma)
max_locations = np.where(is_max_and_bright)

# Manually flag some outliers
# These indices depend on the exact max_locations already obtained, so any changes to the
# code above this point MAY require this flagging to be redone.
ml = np.array(max_locations)
for i in range(ml.shape[1]):
    print("{} {} {}".format(i, ml[0,i], ml[1,i]))
max_locations = np.delete(max_locations, [23, 52, 77, 125, 132, 179, 350, 396, 399, 510], axis=-1)

# Manually set the driftband boundaries.
# A boundary is defined as a line drawn on an array of the same dimensions as cropped_pulsestack.
# The driftband_boundary_points is a list of pairs of points [[x1,y1],[x2,y2]],
# which is easy to fill in by hovering the mouse over a matplotlib plot of cropped_pulsestack.
# These points are later turned into lines (slope,y-intercept) programmatically.
# These points can be quite rough -- it's only important that they "capture" all the points
# into the correct drift band classicifation
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
        [[52,135], [13,167]],
        [[90,138], [20,192]],
        [[90,179], [18,232]],
        [[86,220], [21,261]],
        [[94,244], [27,283]],
        [[94,270], [27,307]],
        [[44,317], [17,328]],
        [[87,314], [11,340]],
        [[78,330], [22,345]],
        [[99,355], [23,370]],
        [[71,342], [20,353]],
        [[39,374], [23,382]],
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
driftband_boundary_lines = np.array([(y2-y1)/(x2-x1), y1-x1*(y2-y1)/(x2-x1)])

assigned_driftbands = []
for max_location in np.transpose(max_locations):
    x = max_location[1]
    y = max_location[0]
    m = driftband_boundary_lines[0,:]
    c = driftband_boundary_lines[1,:]
    assigned_driftbands.append(np.where(y < m*x + c)[0][0] - 1)

#print(assigned_driftbands)

# Pulsestack (image)
#plt.imshow(pulsestack, aspect='auto', origin='lower', interpolation='none')
plt.imshow(cropped_pulsestack, aspect='auto', origin='lower', interpolation='none')
#plt.imshow(smoothed_pulsestack, aspect='auto', origin='lower', interpolation='none')

# Show where the local maxima are
plt.plot(max_locations[1], max_locations[0], 'rx')

# Show where the drift band boundaries are
for p in driftband_boundary_points:
    plt.plot(p[:,0], p[:,1])

# Stacked pulses (lines)
#for pulse in pulses:
#    plt.plot(pulsestack[pulse]*5+pulse)

# Profile:
#plt.plot(np.sum(pulsestack, axis=0))

plt.show()

import numpy as np

dat = np.loadtxt('1226062160_J0024-1932.F.pdv')

npulses = int(dat[-1,0] + 1)
nbins   = int(dat[-1,2] + 1)

pulsestack = np.reshape(dat[:,3], (npulses,nbins))

pulses = np.arange(npulses)
energy = np.sum(pulsestack[:,190:211], axis=-1)
peak   = np.max(pulsestack, axis=-1)

output = np.array([pulses, energy, peak])

np.savetxt('1226062160_J0024-1932_energy.dat', output.T, fmt='%.6f', header="Pulse number, pulse energy, pulse peak")

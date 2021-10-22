#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr  6 22:53:34 2020

@author: Maher Nasr
"""

from JHVIT import jointHypoVel_T, jointHypoVelPS_T, readEventsFiles
from Disp_results import intersectionEll, insideEllipsoid
import numpy as np
import matplotlib.pyplot as plt
from ttcrpy import tmesh
from JHVIT import MSHReader

phase = 'P'  # or 'PS'

if phase == 'P':
    results = jointHypoVel_T('localisation_P.par')
elif phase == 'PS':
    results = jointHypoVelPS_T('localisation_PS.par')

Hypocenters = results['Hypocenters']
Ucrties = results['Uncertainties']
TrueHypo = readEventsFiles('TrueHypo.dat')

# calculate error
Error_T0 = np.abs(Hypocenters[:, 1] - TrueHypo[:, 1])
Error_X = np.abs(Hypocenters[:, 2] - TrueHypo[:, 2])
Error_Y = np.abs(Hypocenters[:, 3] - TrueHypo[:, 3])
Error_Z = np.abs(Hypocenters[:, 4] - TrueHypo[:, 4])
Error_P = np.sqrt(Error_X**2 + Error_Y**2 + Error_Z**2)


# check if true hypocenter positions are inside confidence ellipsoid

print('origin times')
print('---------------')
for h in np.arange(Hypocenters.shape[0]):
    ΔT0 = np.abs(TrueHypo[h, 1] - Hypocenters[h, 1])
    cfd_intrvl = Ucrties[h][0]
    if cfd_intrvl < ΔT0:
        print('\033[43m' +
            'Event N {0:d}: origin time is outside confidence interval'.format(
                int(h + 1)) + '\033[0m')
    else:
        print('\033[42m' +
            'Event N {0:d}: origin time is inside confidence interval'.format(
                int(h + 1)) + '\033[0m')

print('hypocenter positions')
print('---------------')
for h in np.arange(Hypocenters.shape[0]):

    if insideEllipsoid(TrueHypo[h, 2:] * 1.e3, Hypocenters[h, 2:] * 1.e3,
                       Ucrties[h][1] * 1.e3, Ucrties[h][2] * 1.e3,
                       Ucrties[h][3] * 1.e3) is False:
        print('\033[43m' +
            'Event N {0:d}: hypocenter is outside confidence ellipsoid'.format(
                int(h + 1)) + '\033[0m')
    else:
        print('\033[42m' +
            'Event N {0:d}: hypocenter is inside confidence ellipsoid'.format(
                int(h + 1)) + '\033[0m')

# plot hypocenters and  confidence ellipsoids  on map

Topo_data = np.loadtxt('all_dec.xyz') * 1.e-3
nx = (np.unique(Topo_data[:, 0])).size
ny = (np.unique(Topo_data[:, 1])).size
X = Topo_data[:, 0].reshape([nx, ny])
Y = Topo_data[:, 1].reshape([nx, ny])
Z = Topo_data[:, 2].reshape([nx, ny])

# object cmesh
MESH = MSHReader('Model.msh')
nodes = MESH.readNodes()
cells = MESH.readTetraherdonElements()
Mesh3D = tmesh.Mesh3d(nodes, tetra=cells, method='DSPM', cell_slowness=0,
                      n_threads=1, n_secondary=2, n_tertiary=1,
                      process_vel=1, radius_factor_tertiary=2,
                      translate_grid=1)
for h in range(Hypocenters.shape[0]):
    pts = intersectionEll(Ucrties[h][1], Ucrties[h][2],
                          Ucrties[h][3], Hypocenters[h, 2:],
                          Hypocenters[h, 2:], nbreP=50,
                          meshObj=Mesh3D)
    plt.plot(pts[:, 0], pts[:, 1], '-r', markersize=0.1)
    plt.annotate(str(int(1.e3 * Hypocenters[h, 4])),
                 (Hypocenters[h, 2] - 0.008,
                  Hypocenters[h, 3] + 0.006), fontsize=6, color='g')
plt.plot(Hypocenters[:, 2], Hypocenters[:, 3], '.g', markersize=1)
plt.xlim([305.25, 305.55])
contours = plt.contour(X, Y, (1.e3 * Z).astype(int),
                       30, colors='k', linestyles='-')
plt.clabel(contours, inline=True, fontsize=6, colors='b', fmt='%1.0f')
plt.grid(True)
plt.xlabel('X')
plt.ylabel('Y')
plt.savefig('Fig' + phase + '.pdf', format='pdf', bbox_inches='tight')
plt.show()

# -*- coding: utf-8 -*-
"""
Created on Mon Oct 17 15:29:35 2016

@author: Giroux and Nasr
"""

import numpy as np
import sys

data = np.loadtxt('all_dec.xyz')  # Topo file
data *= 1.e-3
Zinf_lim = 0.050  # lower z
nlcTop = 0.022   # target element size at top model corner
nlcBot = 0.032   # target element size at bottom model corner
# path to Gmesh (to modify)
GmshDir = "/Applications/Gmsh.app/Contents/MacOS/Gmsh"


data[:, 0] -= np.min(data[:, 0])
data[:, 1] -= np.min(data[:, 1])
data[:, 2] -= np.min(data[:, 2]) - Zinf_lim
xmin = np.min(data[:, 0])
xmax = np.max(data[:, 0])
ymin = np.min(data[:, 1])
ymax = np.max(data[:, 1])
zmin = np.min(data[:, 2]) - Zinf_lim
x = np.unique(data[:, 0])
y = np.unique(data[:, 1])

nx = x.size
ny = y.size
f = open('Model.geo', 'w')
ft = open('topo.geo', 'w')
pt_no = 1

f.write("\nlc = {0:6.4f};\n\n".format(nlcTop))
ft.write("\nlc = {0:6.4f};\n\n".format(nlcTop))
for xp in x:
    ind = data[:, 0] == xp
    y = data[ind, 1]
    z = data[ind, 2]

    for n in np.arange(ny):
        f.write(
            "Point({0:d}) = {{{1:8.6f}, {2:9.6f}, {3:6.6f}, lc}};\n".format(
                pt_no,
                xp,
                y[n],
                z[n]))
        ft.write(
            "Point({0:d}) = {{{1:8.6f}, {2:9.6f}, {3:6.6f}, lc}};\n".format(
                pt_no, xp, y[n], z[n]))
        pt_no += 1
f.write("\nlc = {0:6.4f};\n\n".format(nlcBot))
f.write(
    "Point({0:d}) = {{{1:8.6f}, {2:9.6f}, {3:6.6f}, lc}};\n".format(
        pt_no, xmin, ymin, zmin))
pt_no += 1
f.write(
    "Point({0:d}) = {{{1:8.6f}, {2:9.6f}, {3:6.6f}, lc}};\n".format(
        pt_no, xmin, ymax, zmin))
pt_no += 1
f.write(
    "Point({0:d}) = {{{1:8.6f}, {2:9.6f}, {3:6.6f}, lc}};\n".format(
        pt_no, xmax, ymin, zmin))
pt_no += 1
f.write(
    "Point({0:d}) = {{{1:8.6f}, {2:9.6f}, {3:6.6f}, lc}};\n\n".format(
        pt_no, xmax, ymax, zmin))
pt_no += 1

li_no = 1
pt_no = 1

for xp in x:
    f.write("BSpline({0:d}) = {{{1:d}".format(li_no, pt_no))
    ft.write("BSpline({0:d}) = {{{1:d}".format(li_no, pt_no))
    pt_no += 1
    for n in np.arange(1, ny):
        f.write(", {0:d}".format(pt_no))
        ft.write(", {0:d}".format(pt_no))
        pt_no += 1
    f.write("};\n")
    ft.write("};\n")
    li_no += 1

for n in np.arange(nx - 1):
    f.write("Line({0:d}) = {{{1:d}, {2:d}}};\n".format(
        li_no, 1 + n * ny, 1 + (n + 1) * ny))
    ft.write("Line({0:d}) = {{{1:d}, {2:d}}};\n".format(
        li_no, 1 + n * ny, 1 + (n + 1) * ny))
    li_no += 1
    f.write(
        "Line({0:d}) = {{{1:d}, {2:d}}};\n".format(
            li_no, (n + 1) * ny, (n + 2) * ny))
    ft.write(
        "Line({0:d}) = {{{1:d}, {2:d}}};\n".format(
            li_no, (n + 1) * ny, (n + 2) * ny))
    li_no += 1

f.write("Line({0:d}) = {{{1:d}, {2:d}}};\n".format(li_no, 1, 1 + nx * ny))
li_no += 1
f.write("Line({0:d}) = {{{1:d}, {2:d}}};\n".format(li_no, ny, 2 + nx * ny))
li_no += 1
f.write("Line({0:d}) = {{{1:d}, {2:d}}};\n".format(
    li_no, 1 + (nx - 1) * ny, 3 + nx * ny))
li_no += 1
f.write(
    "Line({0:d}) = {{{1:d}, {2:d}}};\n".format(
        li_no, nx * ny, 4 + nx * ny))
li_no += 1

f.write(
    "Line({0:d}) = {{{1:d}, {2:d}}};\n".format(
        li_no, 1 + nx * ny, 2 + nx * ny))
li_no += 1
f.write(
    "Line({0:d}) = {{{1:d}, {2:d}}};\n".format(
        li_no, 2 + nx * ny, 4 + nx * ny))
li_no += 1
f.write(
    "Line({0:d}) = {{{1:d}, {2:d}}};\n".format(
        li_no, 4 + nx * ny, 3 + nx * ny))
li_no += 1
f.write(
    "Line({0:d}) = {{{1:d}, {2:d}}};\n".format(
        li_no, 3 + nx * ny, 1 + nx * ny))
li_no += 1

ll_no = li_no
for n in np.arange(nx - 1):
    f.write("Line Loop({0:d}) = {{{1:d}, {2:d}, {3:d}, {4:d}}};\n".format(
        ll_no, n + 1, nx + (n + 1) * 2, -(n + 2), -(nx + 1 + n * 2)))
    ft.write("Line Loop({0:d}) = {{{1:d}, {2:d}, {3:d}, {4:d}}};\n".format(
        ll_no, n + 1, nx + (n + 1) * 2, -(n + 2), -(nx + 1 + n * 2)))
    ll_no += 1


f.write("Line Loop({0:d}) = {{".format(ll_no))
for n in np.arange(nx - 1):
    f.write("{0:d}, ".format(nx + 1 + n * 2))
f.write("{0:d}, {1:d}, {2:d}}};\n".format(nx + 2 * (nx - 1) +
                                          3, nx + 2 * (nx - 1) + 8,
                                          -(nx + 2 * (nx - 1) + 1)))
ll_no += 1

f.write("Line Loop({0:d}) = {{".format(ll_no))
for n in np.arange(nx - 1):
    f.write("{0:d}, ".format(nx + 2 + n * 2))
f.write("{0:d}, {1:d}, {2:d}}};\n".format(nx + 2 * (nx - 1) +
                                          4, -(nx + 2 * (nx - 1) + 6),
                                          -(nx + 2 * (nx - 1) + 2)))
ll_no += 1

f.write("Line Loop({0:d}) = {{{1:d}, {2:d}, {3:d}, {4:d}}};\n".format(
    ll_no, 1, nx + 2 * (nx - 1) + 2, -(li_no - 4), -(li_no - 8)))
ll_no += 1

f.write("Line Loop({0:d}) = {{{1:d}, {2:d}, {3:d}, {4:d}}};\n".format(
    ll_no, nx, nx + 2 * (nx - 1) + 4, li_no - 2, -(li_no - 6)))
ll_no += 1

f.write("Line Loop({0:d}) = {{{1:d}, {2:d}, {3:d}, {4:d}}};\n".format(
    ll_no, li_no - 4, li_no - 3, li_no - 2, li_no - 1))
ll_no += 1


su_no = 1
for n in np.arange(nx - 1):
    f.write("Ruled Surface({0:d}) = {{{1:d}}};\n".format(su_no, li_no + n))
    ft.write("Ruled Surface({0:d}) = {{{1:d}}};\n".format(su_no, li_no + n))
    su_no += 1
f.write("Plane Surface({0:d}) = {{{1:d}}};\n".format(su_no, ll_no - 5))
su_no += 1
f.write("Plane Surface({0:d}) = {{{1:d}}};\n".format(su_no, ll_no - 4))
su_no += 1
f.write("Plane Surface({0:d}) = {{{1:d}}};\n".format(su_no, ll_no - 3))
su_no += 1
f.write("Plane Surface({0:d}) = {{{1:d}}};\n".format(su_no, ll_no - 2))
su_no += 1
f.write("Plane Surface({0:d}) = {{{1:d}}};\n".format(su_no, ll_no - 1))
su_no += 1

sl_no = su_no

f.write("Surface Loop({0:d}) = {{1".format(sl_no))
for n in np.arange(1, su_no - 1):
    f.write(", {0:d}".format(n + 1))
f.write("};\n")
f.write("Volume(1) = {0:d};\n".format(sl_no))
f.write("Physical Volume(\"Roc\") = {1};\n")
f.close()
ft.close()

try:
    from subprocess import Popen
    Popen([GmshDir, "Model.geo", "-3", "-optimize"])
    Popen([GmshDir, "topo.geo", "-3", "-optimize"])
except BaseException:
    print("Unexpected error:", sys.exc_info()[0])
    raise

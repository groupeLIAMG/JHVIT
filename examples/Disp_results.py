#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Apr  3 03:40:04 2020

@author: Maher Nasr
"""

import numpy as np


def insideEllipsoid(P, center, axis1, axis2, axis3):
    # https://en.wikipedia.org/wiki/Ellipsoid : Parametric representation
    P = P.reshape([-1, 1])
    center = center.reshape([-1, 1])
    PI = P - center
    axis1 = axis1.reshape([-1, 1])
    axis2 = axis2.reshape([-1, 1])
    axis3 = axis3.reshape([-1, 1])
    det1 = np.linalg.det(np.hstack([PI, axis2, axis3]))
    det2 = np.linalg.det(np.hstack([axis1, PI, axis3]))
    det3 = np.linalg.det(np.hstack([axis1, axis2, PI]))
    det4 = np.linalg.det(np.hstack([axis1, axis2, axis3]))
    return (det1**2 + det2**2 + det3**2 - det4**2) < 1.e-6


def plotEllipsoid(axis1, axis2, axis3, center, nbreP):
    """
    plot ellipsoid contour defined by its 3 axes (vector)
    """
    nbreP = int(nbreP)
    teta = np.random.uniform(-np.pi * 0.5, np.pi * 0.5, nbreP)
    phi = np.random.uniform(0., np.pi * 2, nbreP)
    X = center[0] + np.cos(teta) * np.cos(phi) * axis1[0] + \
        np.cos(teta) * np.sin(phi) * axis2[0] + np.sin(teta) * axis3[0]
    Y = center[1] + np.cos(teta) * np.cos(phi) * axis1[1] + \
        np.cos(teta) * np.sin(phi) * axis2[1] + np.sin(teta) * axis3[1]
    Z = center[2] + np.cos(teta) * np.cos(phi) * axis1[2] + \
        np.cos(teta) * np.sin(phi) * axis2[2] + np.sin(teta) * axis3[2]
    return X, Y, Z


def intersectionEll(axis1, axis2, axis3, center, points,
                    dirct=2, nbreP=1.e3, meshObj=None):
    """
    find ellipse of intersection between ellipsoid defined
    by its 3 axes (vector) and the plan of direction dir
    """
    c = points[dirct] - center[dirct]
    nbreP = int(nbreP)
    teta = np.hstack(
        (np.linspace(
            0,
            np.pi,
            nbreP // 2),
            np.linspace(
            2 * np.pi,
            np.pi,
            nbreP // 2)))
    alpha = np.arctan(axis2[dirct] / axis1[dirct])
    arg = np.cos(alpha) * (c - axis3[dirct] *
                           np.sin(teta)) / (axis1[dirct] * np.cos(teta))
    ind = abs(arg) <= 1.
    arg = arg[ind]
    n_el = arg.shape[0]
    teta_el = teta[ind]
    phi_el = np.zeros([n_el, ])
    phi_el[:n_el // 2] = alpha - np.arccos(arg[:n_el // 2])
    phi_el[n_el // 2:] = alpha + np.arccos(arg[n_el // 2:])
    X_el = center[0] + np.cos(teta_el) * np.cos(phi_el) * axis1[0] + np.cos(
        teta_el) * np.sin(phi_el) * axis2[0] + np.sin(teta_el) * axis3[0]
    Y_el = center[1] + np.cos(teta_el) * np.cos(phi_el) * axis1[1] + np.cos(
        teta_el) * np.sin(phi_el) * axis2[1] + np.sin(teta_el) * axis3[1]
    Z_el = center[2] + np.cos(teta_el) * np.cos(phi_el) * axis1[2] + np.cos(
        teta_el) * np.sin(phi_el) * axis2[2] + np.sin(teta_el) * axis3[2]
    points = np.column_stack((X_el, Y_el, Z_el))
    if meshObj is not None:
        for i in range(points.shape[0]):
            if meshObj.is_outside(points[i].reshape([1, -1])):
                P2 = points[i, :]
                for _ in range(1000):
                    P2 = 0.1 * center + 0.9 * P2
                    if meshObj.is_outside(P2.reshape([1, -1])) is False:
                        points[i, :] = P2
                        break
    return points

#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""

Join Hypocenter-Velocity Inversion on Tetrahedral meshes (JHVIT).

6 functions can be called and run in this package:

     1- jntHypoVel_T : Joint hypocenter-velocity inversion of P wave data,
        parametrized via the velocity model.

     2- jntHyposlow_T : Joint hypocenter-velocity inversion of P wave data,
        parametrized via the slowness model.

     3- jntHypoVelPS_T : Joint hypocenter-velocity inversion of P- and S-wave data,
        parametrized via the velocity models.

     4- jntHyposlowPS_T : Joint hypocenter-velocity inversion of P- and S-wave data,
        parametrized via the slowness models.

     5-jointHypoVel_T : Joint hypocenter-velocity inversion of P wave data.
       Input data and inversion parameters are downloaded automatically
       from external text files.

     6-jointHypoVelPS_T : Joint hypocenter-velocity inversion of P- and S-wave data.
       Input data and inversion parameters are downloaded automatically
       from external text files.

Notes:
     - The package ttcrpy must be installed in order to perform the raytracing step.
       This package can be downloaded from: https://ttcrpy.readthedocs.io/en/latest/
     - To prevent bugs, it would be better to use python 3.7

Created on Sat Sep 14 2019
@author: maher nasr
"""

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spl
import scipy.stats as scps
import re
import sys
import copy
from mesh import MSHReader
from ttcrpy import tmesh
from multiprocessing import Pool, cpu_count, current_process, Manager
import multiprocessing as mp
from collections import OrderedDict
try:
    import vtk
    from vtk.util.numpy_support import numpy_to_vtk
except BaseException:
    print('VTK module not found, saving velocity model in vtk form is disabled')

def msh2vtk(nodes, cells, velocity, outputFilename, fieldname="Velocity"):
    """
     Generate a vtk file to store the velocity model.

     Parameters
     ----------
     nodes : np.ndarray, shape (nnodes, 3)
          Node coordinates.
     cells : np.ndarray of int, shape (number of cells, 4)
          Indices of nodes forming each cell.
     velocity : np.ndarray, shape (nnodes, 1)
          Velocity model.
     outputFilename : string
          The output vtk filename.
     fieldname : string, optional
          The saved field title. The default is "Velocity".

     Returns
     -------
     float
          return 0.0 if no bugs occur.

     """
    ugrid = vtk.vtkUnstructuredGrid()
    tPts = vtk.vtkPoints()
    tPts.SetNumberOfPoints(nodes.shape[0])
    for n in range(nodes.shape[0]):
        tPts.InsertPoint(n, nodes[n, 0], nodes[n, 1], nodes[n, 2])
    ugrid.SetPoints(tPts)

    VtkVelocity = numpy_to_vtk(velocity, deep=0, array_type=vtk.VTK_DOUBLE)
    VtkVelocity.SetName(fieldname)
    ugrid.GetPointData().SetScalars(VtkVelocity)
    Tetra = vtk.vtkTetra()
    for n in np.arange(cells.shape[0]):
        Tetra.GetPointIds().SetId(0, cells[n, 0])
        Tetra.GetPointIds().SetId(1, cells[n, 1])
        Tetra.GetPointIds().SetId(2, cells[n, 2])
        Tetra.GetPointIds().SetId(3, cells[n, 3])
        ugrid.InsertNextCell(Tetra.GetCellType(), Tetra.GetPointIds())
    gWriter = vtk.vtkUnstructuredGridWriter()
    gWriter.SetFileName(outputFilename)
    gWriter.SetInputData(ugrid)
    gWriter.SetFileTypeToBinary()
    gWriter.Update()
    return 0.0


def check_hypo_indomain(Hypo_new, P_Dimension, Mesh=None):
    """
     Check if the new hypocenter is still inside the domain and
     project it onto the domain surface otherwise.

     Parameters
     ----------
     Hypo_new : np.ndarray, shape (3, ) or (3,1)
          The updated hypocenter coordinates.
     P_Dimension : np.ndarray, shape (6, )
          Domain borders: the maximum and minimum of its 3 dimensions.
     Mesh : instance of the class tmesh, optional
          The domain discretization. The default is None.

     Returns
     -------
     Hypo_new : np.ndarray, shape (3, )
          The input Hypo_new or its projections on the domain surface.
     outside : boolean
          True if Hypo_new was outside the domain.

     """
    outside = False
    Hypo_new = Hypo_new.reshape([1, -1])
    if Hypo_new[0, 0] < P_Dimension[0]:
        Hypo_new[0, 0] = P_Dimension[0]
        outside = True
    if Hypo_new[0, 0] > P_Dimension[1]:
        Hypo_new[0, 0] = P_Dimension[1]
        outside = True
    if Hypo_new[0, 1] < P_Dimension[2]:
        Hypo_new[0, 1] = P_Dimension[2]
        outside = True
    if Hypo_new[0, 1] > P_Dimension[3]:
        Hypo_new[0, 1] = P_Dimension[3]
        outside = True
    if Hypo_new[0, 2] < P_Dimension[4]:
        Hypo_new[0, 2] = P_Dimension[4]
        outside = True
    if Hypo_new[0, 2] > P_Dimension[5]:
        Hypo_new[0, 2] = P_Dimension[5]
        outside = True
    if Mesh:
        if Mesh.is_outside(Hypo_new):
            outside = True
            Hypout = copy.copy(Hypo_new)
            Hypin = np.array([[Hypo_new[0, 0], Hypo_new[0, 1], P_Dimension[4]]])
            distance = np.sqrt(np.sum((Hypin - Hypout)**2))
            while distance > 1.e-5:
                Hmiddle = 0.5 * Hypout + 0.5 * Hypin
                if Mesh.is_outside(Hmiddle):
                    Hypout = Hmiddle
                else:
                    Hypin = Hmiddle
                distance = np.sqrt(np.sum((Hypout - Hypin)**2))
            Hypo_new = Hypin
    return Hypo_new.reshape([-1, ]), outside

class Parameters:

    def __init__(self, maxit, maxit_hypo, conv_hypo, Vlim, VpVslim, dmax,
                 lagrangians, max_sc, invert_vel=True, invert_VsVp=False,
                 hypo_2step=False, use_sc=True, save_vel=False, uncrtants=False,
                 confdce_lev=0.95, verbose=False):
        """
         Parameters
         ----------
         maxit : int
              Maximum number of iterations.
         maxit_hypo : int
              Maximum number of iterations to update hypocenter coordinates.
         conv_hypo : float
              Convergence criterion.
         Vlim : tuple of 3 or 6 floats
              Vlmin holds the maximum and the minimum values of P- and S-wave
              velocity models and the slopes of the penalty functions,
              example Vlim = (Vpmin, Vpmax, PAp, Vsmin, Vsmax, PAs).
         VpVslim : tuple of 3 floats
              Upper and lower limits of Vp/Vs ratio and
              the slope of the corresponding Vp/Vs penalty function.
         dmax : tuple of four floats
              It holds the maximum admissible corrections for the velocity models
              (dVp_max and dVs_max), the origin time (dt_max) and
              the hypocenter coordinates (dx_max).
         lagrangians : tuple of 6 floats
              Penalty and constraint weights: λ (smoothing constraint weight),
              γ (penalty constraint weight), α (weight of velocity data point const-
              raint), wzK (vertical smoothing weight), γ_vpvs (penalty constraint
              weight of Vp/Vs ratio), stig (weight of the constraint used to impose
              statistical moments on Vp/Vs model).
         invert_vel : boolean, optional
              Perform velocity inversion if True. The default is True.
         invert_VsVp : boolean, optional
              Find Vp/Vs ratio model rather than S wave model. The default is False.
         hypo_2step : boolean, optional
              Relocate hypocenter events in 2 steps. The default is False.
         use_sc : boolean, optional
              Use static corrections. The default is 'True'.
         save_vel : string, optional
              Save intermediate velocity models or the final model.
              The default is False.
         uncrtants : boolean, optional
              Calculate the uncertainty of the hypocenter parameters.
              The default is False.
         confdce_lev : float, optional
              The confidence coefficient to calculate the uncertainty.
              The default is 0.95.
         verbose : boolean, optional
              Print information messages about inversion progression.
              The default is False.

         Returns
         -------
         None.

        """
        self.maxit = maxit
        self.maxit_hypo = maxit_hypo
        self.conv_hypo = conv_hypo
        self.Vpmin = Vlim[0]
        self.Vpmax = Vlim[1]
        self.PAp = Vlim[2]
        if len(Vlim) > 3:
            self.Vsmin = Vlim[3]
            self.Vsmax = Vlim[4]
            self.PAs = Vlim[5]
        self.VpVsmin = VpVslim[0]
        self.VpVsmax = VpVslim[1]
        self.Pvpvs = VpVslim[2]
        self.dVp_max = dmax[0]
        self.dx_max = dmax[1]
        self.dt_max = dmax[2]
        if len(dmax) > 3:
            self.dVs_max = dmax[3]
        self.λ = lagrangians[0]
        self.γ = lagrangians[1]
        self.γ_vpvs = lagrangians[2]
        self.α = lagrangians[3]
        self.stig = lagrangians[4]
        self.wzK = lagrangians[5]
        self.invert_vel = invert_vel
        self.invert_VpVs = invert_VsVp
        self.hypo_2step = hypo_2step
        self.use_sc = use_sc
        self.max_sc = max_sc
        self.p = confdce_lev
        self.uncertainty = uncrtants
        self.verbose = verbose
        self.saveVel = save_vel

    def __str__(self):
        """
          Encapsulate the attributes of the class Parameters in a string.

          Returns
          -------
          output : string
               Attributes of the class Parameters written in string.

          """
        output = "-------------------------\n"
        output += "\nParameters of Inversion :\n"
        output += "\n-------------------------\n"
        output += "\nMaximum number of iterations : {0:d}\n".format(self.maxit)
        output += "\nMaximum number of iterations to get hypocenters"
        output += ": {0:d}\n".format(self.maxit_hypo)
        output += "\nVp minimum : {0:4.2f} km/s\n".format(self.Vpmin)
        output += "\nVp maximum : {0:4.2f} km/s\n".format(self.Vpmax)
        if self.Vsmin:
            output += "\nVs minimum : {0:4.2f} km/s\n".format(self.Vsmin)
        if self.Vsmax:
            output += "\nVs maximum : {0:4.2f} km/s\n".format(self.Vsmax)
        if self.VpVsmin:
            output += "\nVpVs minimum : {0:4.2f} km/s\n".format(self.VpVsmin)
        if self.VpVsmax:
            output += "\nVpVs maximum : {0:4.2f} km/s\n".format(self.VpVsmax)
        output += "\nSlope of the penalty function (P wave) : {0:3f}\n".format(
            self.PAp)
        if self.PAs:
            output += "\nSlope of the penalty function (S wave) : {0:3f}\n".format(
                self.PAs)
        if self.Pvpvs:
            output += "\nSlope of the penalty function"
            output += "(VpVs ratio wave) : {0:3f}\n".format(self.Pvpvs)
        output += "\nMaximum time perturbation by step : {0:4.3f} s\n".format(
            self.dt_max)
        output += "\nMaximum distance perturbation by step : {0:4.3f} km\n".format(
            self.dx_max)
        output += "\nMaximum P wave velocity correction by step"
        output += " : {0:4.3f} km/s\n".format(self.dVp_max)
        if self.dVs_max:
            output += "\nMaximum S wave velocity correction by step"
            output += " : {0:4.3f} km/s\n".format(self.dVs_max)
        output += "\nLagrangians parameters : λ = {0:1.1e}\n".format(self.λ)
        output += "                       : γ = {0:1.1e}\n".format(self.γ)
        if self.γ_vpvs:
            output += "                       : γ VpVs ratio = {0:1.1e}\n".format(
                self.γ_vpvs)
        output += "                       : α = {0:1.1e}\n".format(self.α)
        output += "                       : wzK factor = {0:4.2f}\n".format(
            self.wzK)
        if self.stig:
            output += "                       : stats. moment. penalty"
            output += "coef. = {0:1.1e}\n".format(self.stig)
        output += "\nOther parameters : Inverse Velocity = {0}\n".format(
            self.invert_vel)
        output += "\n                 : Use Vs/Vp instead of Vs = {0}\n".format(
            self.invert_VpVs)
        output += "\n                 : Use static correction = {0}\n".format(
            self.use_sc)
        output += "\n                 : Hyp. parameter Uncertainty estimation = "
        output += "{0}\n".format(self.uncertainty)
        if self.uncertainty:
            output += "\n                   with a confidence level of"
            output += " {0:3.2f}\n".format(self.p)
        if self.saveVel == 'last':
            output += "\n                 : Save intermediate velocity models = "
            output += "last iteration only\n"
        elif self.saveVel == 'all':
            output += "\n                 : Save intermediate velocity models = "
            output += "all iterations\n"
        else:
            output += "\n                 : Save intermediate velocity models = "
            output += "False\n"
        output += "\n                 : Relocate hypoctenters using 2 steps = "
        output += "{0}\n".format(self.hypo_2step)
        output += "\n                 : convergence criterion = {0:3.4f}\n".format(
            self.conv_hypo)
        if self.use_sc:
            output += "\n                 : Maximum static correction = "
            output += "{0:3.2f}\n".format(self.max_sc)
        return output


class fileReader:
    def __init__(self, filename):
        """
         Parameters
         ----------
         filename : string
              List of data files and other inversion parameters.

         Returns
         -------
         None.

         """
        try:
            open(filename, 'r')
        except IOError:
            print("Could not read file:", filename)
            sys.exit()
        self.filename = filename
        assert(self.readParameter('base name')), 'invalid base name'
        assert(self.readParameter('mesh file')), 'invalid mesh file'
        assert(self.readParameter('rcvfile')), 'invalid rcv file'
        assert(self.readParameter('Velocity')), 'invalid Velocity file'
        assert(self.readParameter('Time calibration')
               ), 'invalid calibration data file'

    def readParameter(self, parameter, dtype=None):
        """
        Read the data filename or the inversion parameter value specified by
        the argument parameter.

         Parameters
         ----------
         parameter : string
              Filename or inversion parameter to read.
         dtype : data type, optional
              Explicit data type of the filename or the parameter read.
              The default is None.

         Returns
         -------
         param : string/int/float
              File or inversion parameter.

         """
        try:
            f = open(self.filename, 'r')
            for line in f:
                if line.startswith(parameter):
                    position = line.find(':')
                    param = line[position + 1:]
                    param = param.rstrip("\n\r")
                    if dtype is None:
                        break
                    if dtype == int:
                        param = int(param)
                    elif dtype == float:
                        param = float(param)
                    elif dtype == bool:
                        if param == 'true' or param == 'True' or param == '1':
                            param = True
                        elif param == 'false' or param == 'False' or param == '0':
                            param = False
                    else:
                        print(" non recognized format")
                    break
            return param
        except OSError as err:
            print("OS error: {0}".format(err))
        except ValueError:
            print("Could not convert data to a float for " + parameter + "\n")
        except NameError as NErr:
            print(
                parameter +
                " is not indicated or has bad value:{0}".format(NErr))
        except BaseException:
            print("Unexpected error:", sys.exc_info()[0])
            raise
        finally:
            f.close()

    def saveVel(self):
        """
         Method to read the specified option for saving the velocity model(s).

         Returns
         -------
         bool/string
              Save or not the velocity model(s) and for which iteration.

         """
        try:
            f = open(self.filename, 'r')
            for line in f:
                if line.startswith('Save Velocity'):
                    position = line.find(':')
                    if position > 0:
                        sv = line[position + 1:].strip()
                        break
            f.close()
            if sv == 'last' or sv == 'Last':
                return 'last'
            elif sv == 'all' or sv == 'All':
                return 'all'
            elif sv == 'false' or sv == 'False' or sv == '0':
                return False
            else:
                print('bad option to save velocity: default value will be used')
                return False
        except OSError as err:
            print("OS error: {0}".format(err))
        except NameError as NErr:
            print("save velocity is not indicated :{0}".format(NErr))
        except BaseException:
            print("Unexpected error:", sys.exc_info()[0])
            raise

    def getIversionParam(self):
        """
         Read the inversion parameters and
         store them in an object of the class Parameters.

         Returns
         -------
         Params : instance of the class Parameters
              Inversion parameters and options.

         """
        maxit = self.readParameter('number of iterations', int)
        maxit_hypo = self.readParameter('num. iters. to get hypo.', int)
        conv_hypo = self.readParameter('convergence Criterion', float)

        Vpmin = self.readParameter('Vpmin', float)
        Vpmax = self.readParameter('Vpmax', float)
        PAp = self.readParameter('PAp', float)
        if PAp is None or PAp < 0:
            print('PAp : default value will be considered\n')
            PAp = 1.  # default value
        Vsmin = self.readParameter('Vsmin', float)
        Vsmax = self.readParameter('Vsmax', float)
        PAs = self.readParameter('PAs', float)
        if PAs is None or PAs < 0:
            print('PAs : default value will be considered\n')
            PAs = 1.  # default value
        VpVsmax = self.readParameter('VpVs_max', float)
        if VpVsmax is None or VpVsmax < 0:
            print('default value will be considered (5)\n')
            VpVsmax = 5.  # default value
        VpVsmin = self.readParameter('VpVs_min', float)
        if VpVsmin is None or VpVsmin < 0:
            print('default value will be considered (1.5)\n')
            VpVsmin = 1.5  # default value
        Pvpvs = self.readParameter('Pvpvs', float)
        if Pvpvs is None or Pvpvs < 0:
            print('default value will be considered\n')
            Pvpvs = 1.  # default value
        dVp_max = self.readParameter('dVp max', float)
        dVs_max = self.readParameter('dVs max', float)
        dx_max = self.readParameter('dx max', float)
        dt_max = self.readParameter('dt max', float)
        Alpha = self.readParameter('alpha', float)
        Lambda = self.readParameter('lambda', float)
        Gamma = self.readParameter('Gamma', float)
        Gamma_ps = self.readParameter('Gamma_vpvs', float)

        stigma = self.readParameter('stigma', float)
        if stigma is None or stigma < 0:
            stigma = 0.  # default value
        VerSmooth = self.readParameter('vertical smoothing', float)

        InverVel = self.readParameter('inverse velocity', bool)
        InverseRatio = self.readParameter('inverse Vs/Vp', bool)
        Hyp2stp = self.readParameter('reloc.hypo.in 2 steps', bool)
        Sc = self.readParameter('use static corrections', bool)
        if Sc:
            Sc_max = self.readParameter('maximum stat. correction', float)
        else:
            Sc_max = 0.
        uncrtants = self.readParameter('uncertainty estm.', bool)
        if uncrtants:
            confdce_lev = self.readParameter('confidence level', float)
        else:
            confdce_lev = np.NAN
        Verb = self.readParameter('Verbose ', bool)
        saveVel = self.saveVel()
        Params = Parameters(maxit, maxit_hypo, conv_hypo,
                            (Vpmin, Vpmax, PAp, Vsmin, Vsmax, PAs),
                            (VpVsmin, VpVsmax, Pvpvs),
                            (dVp_max, dx_max, dt_max, dVs_max),
                            (Lambda, Gamma, Gamma_ps, Alpha, stigma,
                             VerSmooth), Sc_max, InverVel, InverseRatio,
                            Hyp2stp, Sc, saveVel, uncrtants, confdce_lev, Verb)
        return Params


class RCVReader:
    def __init__(self, p_rcvfile):
        """
         Parameters
         ----------
         p_rcvfile : string
              File holding receiver coordinates.

         Returns
         -------
         None.

         """
        self.rcv_file = p_rcvfile
        assert(self.__ChekFormat()), 'invalid format for rcv file'

    def getNumberOfStation(self):
        """
         Return the number of receivers.

         Returns
         -------
         Nstations : int
              Receiver number.

         """
        try:
            fin = open(self.rcv_file, 'r')
            Nstations = int(fin.readline())
            fin.close()
            return Nstations
        except OSError as err:
            print("OS error: {0}".format(err))
        except ValueError:
            print("Could not convert data to an integer for the station number.")
        except BaseException:
            print("Unexpected error:", sys.exc_info()[0])
            raise

    def getStation(self):
        """
         Return coordinates of receivers.

         Returns
         -------
         coordonates : np.ndarray, shape(receiver number,3)
              Receiver coordinates.

         """
        try:
            fin = open(self.rcv_file, 'r')
            Nsta = int(fin.readline())
            coordonates = np.zeros([Nsta, 3])
            for n in range(Nsta):
                line = fin.readline()
                Coord = re.split(r' ', line)
                coordonates[n, 0] = float(Coord[0])
                coordonates[n, 1] = float(Coord[2])
                coordonates[n, 2] = float(Coord[4])
            fin.close()
            return coordonates
        except OSError as err:
            print("OS error: {0}".format(err))
        except ValueError:
            print("Could not convert data to a float in rcvfile.")
        except BaseException:
            print("Unexpected error:", sys.exc_info()[0])
            raise

    def __ChekFormat(self):
        try:
            fin = open(self.rcv_file)
            n = 0
            for line in fin:
                if n == 0:
                    Nsta = int(line)
                    num_lines = sum(1 for line in fin)
                    if(num_lines != Nsta):
                        fin.close()
                        return False
                if n > 0:
                    Coord = re.split(r' ', line)
                    if len(Coord) != 5:
                        fin.close()
                        return False
                n += 1
            fin.close()
            return True
        except OSError as err:
            print("OS error: {0}".format(err))
        except ValueError:
            print("Could not convert data to a float in rcvfile.")
        except BaseException:
            print("Unexpected error:", sys.exc_info()[0])
            raise


def readEventsFiles(time_file, waveType=False):
    """
     Read a list of seismic events and corresponding data from a text file.

     Parameters
     ----------
     time_file : string
          Event data filename.
     waveType : bool
          True if the seismic phase of each event is identified.
          The default is False.

     Returns
     -------
     data : np.ndarray or a list of two np.ndarrays
          Event arrival time data

     """
    if (time_file == ""):
        if not waveType:
            return (np.array([]))
        elif waveType:
            return (np.array([]), np.array([]))
    try:
        fin = open(time_file, 'r')
        lstart = 0
        for line in fin:
            lstart += 1
            if line.startswith('Ev_idn'):
                break
        if not waveType:
            data = np.loadtxt(time_file, skiprows=lstart, ndmin=2)
        elif waveType:
            data = np.loadtxt(fname=time_file, skiprows=2,
                              dtype='S15', ndmin=2)
            ind = np.where(data[:, -1] == b'P')[0]
            dataP = data[ind, :-1].astype(float)

            ind = np.where(data[:, -1] == b'S')[0]
            dataS = data[ind, :-1].astype(float)
            data = (dataP, dataS)
        return data
    except OSError as err:
        print("OS error: {0}".format(err))
    except ValueError:
        print("Could not convert data to a float in  " + time_file + "  file.")
    except BaseException:
        print("Unexpected error:", sys.exc_info()[0])
        raise


def readVelpoints(vlpfile):
    """
     Read known velocity points from a text file.

     Parameters
     ----------
     vlpfile : string
          Name of the file containing the known velocity points.

     Returns
     -------
     data : np.ndarray, shape (number of points , 3)
          Data corresponding to the known velocity points.

     """
    if (vlpfile == ""):
        return (np.array([]))
    try:
        fin = open(vlpfile, 'r')
        lstart = 0
        for line in fin:
            lstart += 1
            if line.startswith('Pt_id'):
                break
        data = np.loadtxt(vlpfile, skiprows=lstart)
        return data
    except OSError as err:
        print("OS error: {0}".format(err))
    except ValueError:
        print("Could not convert data to a float in  " + vlpfile + "  file.")
    except BaseException:
        print("Unexpected error:", sys.exc_info()[0])
    raise


def _hypo_relocation(ev, evID, hypo, data, rcv, sc, convergence, par):
    """
     Location of a single hypocenter event using P arrival time data.

     Parameters
     ----------
     ev : int
          Event index in the array evID.
     evID : np.ndarray, shape (number of events ,)
          Event indices.
     hypo : np.ndarray, shape (number of events ,5)
          Current hypocenter coordinates and origin time for each event.
     data : np.ndarray, shape (arrival times number,3)
          Arrival times for all events.
     rcv : np.ndarray, shape (receiver number,3)
          Coordinates of receivers.
     sc : np.ndarray, shape (receiver number or 0 ,1)
          Static correction values.
     convergence : boolean list, shape (event number)
          Convergence state of each event.
     par : instance of the class Parameters
          The inversion parameters.

     Returns
     -------
     Hypocenter : np.ndarray, shape (5,)
          Updated origin time and coordinates of event evID[ev].

     """
    indh = np.where(hypo[:, 0] == evID[ev])[0]
    if par.verbose:
        print("\nEven N {0:d} is relacated in the ".format(
            int(hypo[ev, 0])) + current_process().name + '\n')
        sys.stdout.flush()
    indr = np.where(data[:, 0] == evID[ev])[0]
    rcv_ev = rcv[data[indr, 2].astype(int) - 1, :]
    if par.use_sc:
        sc_ev = sc[data[indr, 2].astype(int) - 1]
    else:
        sc_ev = 0.
    nst = indr.size
    Hypocenter = hypo[indh[0]].copy()
    if par.hypo_2step:
        print("\nEven N {0:d}: Update longitude and latitude\n".format(
            int(hypo[ev, 0])))
        sys.stdout.flush()

        T0 = np.kron(hypo[indh, 1], np.ones([nst, 1]))
        for It in range(par.maxit_hypo):
            Tx = np.kron(Hypocenter[2:], np.ones([nst, 1]))
            src = np.hstack((ev*np.ones([nst, 1]), T0 + sc_ev, Tx))
            tcal, rays = Mesh3D.raytrace(source=src, rcv=rcv_ev, slowness=None,
                                         aggregate_src=False, compute_L=False,
                                         return_rays=True)
            slow_0 = Mesh3D.get_s0(src)
            Hi = np.ones((nst, 2))
            for nr in range(nst):
                rayi = rays[nr]
                if rayi.shape[0] == 1:
                    print('\033[43m' + '\nWarning: raypath failed to converge'
                          ' for even N {0:d} :({1:5.4f},{2:5.4f},{3:5.4f}) and '
                          'receiver N {4:d} :({5:5.4f},{6:5.4f},{7:5.4f})\n'.format(
                               int(data[indr[nr], 0]),
                               Tx[nr, 0], Tx[nr, 1], Tx[nr, 2],
                               int(data[indr[nr], 2]), rcv_ev[nr, 0],
                               rcv_ev[nr, 1], rcv_ev[nr, 2]) + '\033[0m')
                    sys.stdout.flush()
                    continue
                slw0 = slow_0[nr]
                dx = rayi[1, 0] - Hypocenter[2]
                dy = rayi[1, 1] - Hypocenter[3]
                dz = rayi[1, 2] - Hypocenter[4]
                ds = np.sqrt(dx * dx + dy * dy + dz * dz)
                Hi[nr, 0] = -dx * slw0 / ds
                Hi[nr, 1] = -dy * slw0 / ds
            convrays = np.where(tcal != 0)[0]
            res = data[indr, 1] - tcal
            if convrays.size < nst:
                res = res[convrays]
                Hi = Hi[convrays, :]
            deltaH = np.linalg.lstsq(Hi, res, rcond=1.e-6)[0]
            if not np.all(np.isfinite(deltaH)):
                try:
                    U, S, VVh = np.linalg.svd(Hi.T.dot(Hi) + 1e-9 * np.eye(2))
                    VV = VVh.T
                    deltaH = np.dot(VV, np.dot(U.T, Hi.T.dot(res)) / S)
                except np.linalg.linalg.LinAlgError:
                    print('\nEvent could not be relocated (iteration no ' +
                          str(It) + '), skipping')
                    sys.stdout.flush()
                    break
            indH = np.abs(deltaH) > par.dx_max
            deltaH[indH] = par.dx_max * np.sign(deltaH[indH])
            updatedHypo = np.hstack((Hypocenter[2:4] + deltaH, Hypocenter[-1]))
            updatedHypo, _ = check_hypo_indomain(updatedHypo, Dimensions,
                                                 Mesh3D)
            Hypocenter[2:] = updatedHypo
            if np.all(np.abs(deltaH[1:]) < par.conv_hypo):
                break
    if par.verbose:
        print("\nEven N {0:d}: Update all parameters\n".format(int(hypo[ev, 0])))
        sys.stdout.flush()
    for It in range(par.maxit_hypo):
        Tx = np.kron(Hypocenter[2:], np.ones([nst, 1]))
        T0 = np.kron(Hypocenter[1], np.ones([nst, 1]))
        src = np.hstack((ev*np.ones([nst, 1]), T0 + sc_ev, Tx))

        tcal, rays = Mesh3D.raytrace(source=src, rcv=rcv_ev, slowness=None,
                                     aggregate_src=False, compute_L=False,
                                     return_rays=True)
        slow_0 = Mesh3D.get_s0(src)
        Hi = np.ones([nst, 4])
        for nr in range(nst):
            rayi = rays[nr]
            if rayi.shape[0] == 1:
                print('\033[43m' + '\nWarning: raypath failed to converge '
                      'for even N {0:d} :({1:5.4f},{2:5.4f},{3:5.4f}) and '
                      'receiver N {4:d} :({5:5.4f},{6:5.4f},{7:5.4f})\n'.format(
                           int(data[indr[nr], 0]), Tx[nr, 0],
                           Tx[nr, 1], Tx[nr, 2], int(data[indr[nr], 2]),
                           rcv_ev[nr, 0], rcv_ev[nr, 1], rcv_ev[nr, 2]) + '\033[0m')
                sys.stdout.flush()
                continue
            slw0 = slow_0[nr]
            dx = rayi[1, 0] - Hypocenter[2]
            dy = rayi[1, 1] - Hypocenter[3]
            dz = rayi[1, 2] - Hypocenter[4]
            ds = np.sqrt(dx * dx + dy * dy + dz * dz)
            Hi[nr, 1] = -dx * slw0 / ds
            Hi[nr, 2] = -dy * slw0 / ds
            Hi[nr, 3] = -dz * slw0 / ds
        convrays = np.where(tcal != 0)[0]
        res = data[indr, 1] - tcal
        if convrays.size < nst:
            res = res[convrays]
            Hi = Hi[convrays, :]
        deltaH = np.linalg.lstsq(Hi, res, rcond=1.e-6)[0]
        if not np.all(np.isfinite(deltaH)):
            try:
                U, S, VVh = np.linalg.svd(Hi.T.dot(Hi) + 1e-9 * np.eye(4))
                VV = VVh.T
                deltaH = np.dot(VV, np.dot(U.T, Hi.T.dot(res)) / S)
            except np.linalg.linalg.LinAlgError:
                print('\nEvent cannot be relocated (iteration no ' +
                      str(It) + '), skipping')
                sys.stdout.flush()
                break
        if np.abs(deltaH[0]) > par.dt_max:
            deltaH[0] = par.dt_max * np.sign(deltaH[0])
        if np.linalg.norm(deltaH[1:]) > par.dx_max:
            deltaH[1:] *= par.dx_max / np.linalg.norm(deltaH[1:])
        updatedHypo = Hypocenter[2:] + deltaH[1:]
        updatedHypo, outside = check_hypo_indomain(updatedHypo,
                                                   Dimensions, Mesh3D)
        Hypocenter[1:] = np.hstack((Hypocenter[1] + deltaH[0], updatedHypo))
        if outside and It == par.maxit_hypo - 1:
            print('\nEvent N {0:d} cannot be relocated inside the domain\n'.format(
                int(hypo[ev, 0])))
            convergence[ev] = 'out'
            return Hypocenter
        if np.all(np.abs(deltaH[1:]) < par.conv_hypo):
            convergence[ev] = True
            if par.verbose:
                print('\033[42m' + '\nEven N {0:d} has converged at {1:d}'
                      ' iteration(s)\n'.format(int(hypo[ev, 0]), It + 1) + '\n'
                      + '\033[0m')
                sys.stdout.flush()
            break
    else:
        if par.verbose:
            print('\nEven N {0:d} : maximum number of iterations'
                  ' was reached\n'.format(int(hypo[ev, 0])) + '\n')
            sys.stdout.flush()
    return Hypocenter


def _hypo_relocationPS(ev, evID, hypo, data, rcv, sc, convergence, slow, par):
    """
     Relocate a single hypocenter event using P- and S-wave arrival times.

     Parameters
     ----------
     ev : int
          Event index in the array evID.
     evID : np.ndarray, shape (event number ,)
          Event indices.
     hypo : np.ndarray, shape (event number ,5)
          Current hypocenter coordinates and origin times for each event.
     data : tuple of two np.ndarrays
          Arrival times of P- and S-waves.
     rcv : np.ndarray, shape (receiver number,3)
          Coordinates of receivers.
     sc : tuple of two np.ndarrays (shape(receiver number or 0,1))
          Static correction values of P- and S-waves.
     convergence : boolean list, shape (event number)
          The convergence state of each event.
     slow : tuple of two np.ndarrays (shape(nnodes,1))
          P and S slowness models.
     par : instance of the class Parameters
          The inversion parameters.

     Returns
     -------
     Hypocenter : np.ndarray, shape (5,)
          Updated origin time and coordinates of event evID[ev].

     """
    (slowP, slowS) = slow
    (scp, scs) = sc
    (dataP, dataS) = data
    indh = np.where(hypo[:, 0] == evID[ev])[0]
    if par.verbose:
        print("Even N {0:d} is relacated in the ".format(
            int(hypo[ev, 0])) + current_process().name + '\n')
        sys.stdout.flush()
    indrp = np.where(dataP[:, 0] == evID[ev])[0]
    rcv_evP = rcv[dataP[indrp, 2].astype(int) - 1, :]
    nstP = indrp.size
    indrs = np.where(dataS[:, 0] == evID[ev])[0]
    rcv_evS = rcv[dataS[indrs, 2].astype(int) - 1, :]
    nstS = indrs.size
    Hypocenter = hypo[indh[0]].copy()
    if par.use_sc:
        scp_ev = scp[dataP[indrp, 2].astype(int) - 1]
        scs_ev = scs[dataS[indrs, 2].astype(int) - 1]
    else:
        scp_ev = np.zeros([nstP, 1])
        scs_ev = np.zeros([nstS, 1])
    if par.hypo_2step:
        if par.verbose:
            print("\nEven N {0:d}: Update longitude and latitude\n".format(
                int(hypo[ev, 0])))
            sys.stdout.flush()
        for It in range(par.maxit_hypo):
            Txp = np.kron(Hypocenter[1:], np.ones([nstP, 1]))
            Txp[:, 0] += scp_ev[:, 0]
            srcP = np.hstack((ev*np.ones([nstP, 1]), Txp))
            tcalp, raysP = Mesh3D.raytrace(source=srcP, rcv=rcv_evP, slowness=slowP,
                                           aggregate_src=False, compute_L=False,
                                           return_rays=True)
            slowP_0 = Mesh3D.get_s0(srcP)

            Txs = np.kron(Hypocenter[1:], np.ones([nstS, 1]))
            Txs[:, 0] += scs_ev[:, 0]
            srcS = np.hstack((ev*np.ones([nstS, 1]), Txs))
            tcals, raysS = Mesh3D.raytrace(source=srcS, rcv=rcv_evS, slowness=slowS,
                                           aggregate_src=False, compute_L=False,
                                           return_rays=True)
            slowS_0 = Mesh3D.get_s0(srcS)
            Hi = np.ones((nstP + nstS, 2))
            for nr in range(nstP):
                rayi = raysP[nr]
                if rayi.shape[0] == 1:
                    if par.verbose:
                        print('\033[43m' +
                              '\nWarning: raypath failed to converge for even '
                              'N {0:d} :({1:5.4f},{2:5.4f},{3:5.4f})and receiver '
                              'N {4:d} :({5:5.4f},{6:5.4f},{7:5.4f})\n'.format(
                                   int(dataP[indrp[nr], 0]),
                                   Txp[nr, 1], Txp[nr, 2], Txp[nr, 3],
                                   int(dataP[indrp[nr], 2]),
                                   rcv_evP[nr, 0], rcv_evP[nr, 1],
                                   rcv_evP[nr, 2]) +
                              '\033[0m')
                        sys.stdout.flush()
                    continue
                slw0 = slowP_0[nr]
                dx = rayi[1, 0] - Hypocenter[2]
                dy = rayi[1, 1] - Hypocenter[3]
                dz = rayi[1, 2] - Hypocenter[4]
                ds = np.sqrt(dx * dx + dy * dy + dz * dz)
                Hi[nr, 0] = -dx * slw0 / ds
                Hi[nr, 1] = -dy * slw0 / ds
            for nr in range(nstS):
                rayi = raysS[nr]
                if rayi.shape[0] == 1:
                    if par.verbose:
                        print('\033[43m' +
                              '\nWarning: raypath failed to converge for even '
                              'N {0:d} :({1:5.4f},{2:5.4f},{3:5.4f}) and receiver '
                              'N {4:d} :({5:5.4f},{6:5.4f},{7:5.4f})\n'.format(
                                   int(dataS[indrs[nr], 0]),
                                   Txs[nr, 1], Txs[nr, 2],
                                   Txs[nr, 3], int(dataS[indrs[nr], 2]),
                                   rcv_evS[nr, 0], rcv_evS[nr, 1],
                                   rcv_evS[nr, 2]) +
                              '\033[0m')
                        sys.stdout.flush()
                    continue
                slw0 = slowS_0[nr]
                dx = rayi[1, 0] - Hypocenter[2]
                dy = rayi[1, 1] - Hypocenter[3]
                dz = rayi[1, 2] - Hypocenter[4]
                ds = np.sqrt(dx * dx + dy * dy + dz * dz)
                Hi[nr + nstP, 0] = -dx * slw0 / ds
                Hi[nr + nstP, 1] = -dy * slw0 / ds
            tcal = np.hstack((tcalp, tcals))
            res = np.hstack((dataP[indrp, 1], dataS[indrs, 1])) - tcal
            convrays = np.where(tcal != 0)[0]
            if convrays.size < (nstP + nstS):
                res = res[convrays]
                Hi = Hi[convrays, :]
            deltaH = np.linalg.lstsq(Hi, res, rcond=1.e-6)[0]
            if not np.all(np.isfinite(deltaH)):
                try:
                    U, S, VVh = np.linalg.svd(Hi.T.dot(Hi) + 1e-9 * np.eye(2))
                    VV = VVh.T
                    deltaH = np.dot(VV, np.dot(U.T, Hi.T.dot(res)) / S)
                except np.linalg.linalg.LinAlgError:
                    if par.verbose:
                        print('\nEvent could not be relocated (iteration no ' +
                              str(It) + '), skipping')
                        sys.stdout.flush()
                    break
            indH = np.abs(deltaH) > par.dx_max
            deltaH[indH] = par.dx_max * np.sign(deltaH[indH])
            updatedHypo = np.hstack((Hypocenter[2:4] + deltaH, Hypocenter[-1]))
            updatedHypo, _ = check_hypo_indomain(updatedHypo, Dimensions,
                                                 Mesh3D)
            Hypocenter[2:] = updatedHypo
            if np.all(np.abs(deltaH) < par.conv_hypo):
                break
    if par.verbose:
        print("\nEven N {0:d}: Update all parameters\n".format(int(hypo[ev, 0])))
        sys.stdout.flush()
    for It in range(par.maxit_hypo):
        Txp = np.kron(Hypocenter[1:], np.ones([nstP, 1]))
        Txp[:, 0] += scp_ev[:, 0]
        srcP = np.hstack((ev*np.ones([nstP, 1]), Txp))
        tcalp, raysP = Mesh3D.raytrace(source=srcP, rcv=rcv_evP, slowness=slowP,
                                       aggregate_src=False, compute_L=False,
                                       return_rays=True)
        slowP_0 = Mesh3D.get_s0(srcP)
        Txs = np.kron(Hypocenter[1:], np.ones([nstS, 1]))
        Txs[:, 0] += scs_ev[:, 0]
        srcS = np.hstack((ev*np.ones([nstS, 1]), Txs))
        tcals, raysS = Mesh3D.raytrace(source=srcS, rcv=rcv_evS, slowness=slowS,
                                       aggregate_src=False, compute_L=False,
                                       return_rays=True)
        slowS_0 = Mesh3D.get_s0(srcS)
        Hi = np.ones((nstP + nstS, 4))
        for nr in range(nstP):
            rayi = raysP[nr]
            if rayi.shape[0] == 1:
                if par.verbose:
                    print('\033[43m' + '\nWarning: raypath failed to converge for '
                          'even N {0:d} :({1:5.4f},{2:5.4f},{3:5.4f}) and receiver '
                          'N {4:d} :({5:5.4f},{6:5.4f},{7:5.4f})\n'.format(
                           int(dataP[indrp[nr], 0]), Txp[nr, 1],
                           Txp[nr, 2], Txp[nr, 3], int(dataP[indrp[nr], 2]),
                           rcv_evP[nr, 0], rcv_evP[nr, 1],
                           rcv_evP[nr, 2]) + '\033[0m')
                    sys.stdout.flush()
                continue
            slw0 = slowP_0[nr]
            dx = rayi[2, 0] - Hypocenter[2]
            dy = rayi[2, 1] - Hypocenter[3]
            dz = rayi[2, 2] - Hypocenter[4]
            ds = np.sqrt(dx * dx + dy * dy + dz * dz)
            Hi[nr, 1] = -dx * slw0 / ds
            Hi[nr, 2] = -dy * slw0 / ds
            Hi[nr, 3] = -dz * slw0 / ds
        for nr in range(nstS):
            rayi = raysS[nr]
            if rayi.shape[0] == 1:
                if par.verbose:
                    print('\033[43m' + '\nWarning: raypath failed to converge for '
                          'even N {0:d} :({1:5.4f},{2:5.4f},{3:5.4f}) and receiver '
                          'N {4:d} :({5:5.4f},{6:5.4f},{7:5.4f})\n'.format(
                               int(dataS[indrs[nr], 0]), Txs[nr, 1],
                               Txs[nr, 2], Txs[nr, 3], int(dataS[indrs[nr], 2]),
                               rcv_evS[nr, 0], rcv_evS[nr, 1],
                               rcv_evS[nr, 2]) + '\033[0m')
                    sys.stdout.flush()
                continue
            slw0 = slowS_0[nr]
            dx = rayi[1, 0] - Hypocenter[2]
            dy = rayi[1, 1] - Hypocenter[3]
            dz = rayi[1, 2] - Hypocenter[4]
            ds = np.sqrt(dx * dx + dy * dy + dz * dz)
            Hi[nr + nstP, 1] = -dx * slw0 / ds
            Hi[nr + nstP, 2] = -dy * slw0 / ds
            Hi[nr + nstP, 3] = -dz * slw0 / ds
        tcal = np.hstack((tcalp, tcals))
        res = np.hstack((dataP[indrp, 1], dataS[indrs, 1])) - tcal
        convrays = np.where(tcal != 0)[0]
        if convrays.size < (nstP + nstS):
            res = res[convrays]
            Hi = Hi[convrays, :]
        deltaH = np.linalg.lstsq(Hi, res, rcond=1.e-6)[0]
        if not np.all(np.isfinite(deltaH)):
            try:
                U, S, VVh = np.linalg.svd(Hi.T.dot(Hi) + 1e-9 * np.eye(4))
                VV = VVh.T
                deltaH = np.dot(VV, np.dot(U.T, Hi.T.dot(res)) / S)
            except np.linalg.linalg.LinAlgError:
                if par.verbose:
                    print('\nEvent could not be relocated (iteration no ' +
                          str(It) + '), skipping\n')
                    sys.stdout.flush()
                break
        if np.abs(deltaH[0]) > par.dt_max:
            deltaH[0] = par.dt_max * np.sign(deltaH[0])
        if np.linalg.norm(deltaH[1:]) > par.dx_max:
            deltaH[1:] *= par.dx_max / np.linalg.norm(deltaH[1:])
        updatedHypo = Hypocenter[2:] + deltaH[1:]
        updatedHypo, outside = check_hypo_indomain(updatedHypo,
                                                   Dimensions, Mesh3D)
        Hypocenter[1:] = np.hstack((Hypocenter[1] + deltaH[0], updatedHypo))
        if outside and It == par.maxit_hypo - 1:
            if par.verbose:
                print('\nEvent N {0:d} could not be relocated inside '
                      'the domain\n'.format(int(hypo[ev, 0])))
                sys.stdout.flush()
            convergence[ev] = 'out'
            return Hypocenter
        if np.all(np.abs(deltaH[1:]) < par.conv_hypo):
            convergence[ev] = True
            if par.verbose:
                print('\033[42m' + '\nEven N {0:d} has converged at '
                      ' iteration {1:d}\n'.format(int(hypo[ev, 0]), It + 1) +
                      '\n' + '\033[0m')
                sys.stdout.flush()
            break
    else:
        if par.verbose:
            print('\nEven N {0:d} : maximum number of iterations was'
                  ' reached'.format(int(hypo[ev, 0])) + '\n')
            sys.stdout.flush()
    return Hypocenter


def _uncertaintyEstimat(ev, evID, hypo, data, rcv, sc, slow, par, varData=None):
    """
     Estimate origin time uncertainty and confidence ellipsoid.

     Parameters
     ----------
     ev : int
          Event index in the array evID.
     evID : np.ndarray, shape (event number,)
          Event indices.
     hypo : np.ndarray, shape (event number,5)
          Estimated hypocenter coordinates and origin time.
     data : np.ndarray, shape (arrival time number,3) or
            tuple if both P and S waves are used.
          Arrival times of seismic events.
     rcv : np.ndarray, shape (receiver number ,3)
          coordinates of receivers.
     sc : np.ndarray, shape (receiver number or 0 ,1) or
          tuple if both P and S waves are used.
          Static correction values.
     slow : np.ndarray or tuple, shape(nnodes,1)
          P or P and S slowness models.
     par : instance of the class Parameters
          The inversion parameters.
     varData : list of two lists
          Number of arrival times and the sum of residuals needed to
          compute the noise variance. See Block's Thesis, 1991 (P. 63)
          The default is None.

     Returns
     -------
     to_confInterv : float
          Origin time uncertainty interval.
     axis1 : np.ndarray, shape(3,)
          Coordinates of the 1st confidence ellipsoid axis (vector).
     axis2 : np.ndarray, shape(3,)
          Coordinates of the 2nd confidence ellipsoid axis (vector).
     axis3 : np.ndarray, shape(3,)
          Coordinates of the 3rd confidence ellipsoid axis (vector).

     """
    if par.verbose:
        print("Uncertainty estimation for the Even N {0:d}".format(
            int(hypo[ev, 0])) + '\n')
        sys.stdout.flush()
    indh = np.where(hypo[:, 0] == evID[ev])[0]
    if len(slow) == 2:
        (slowP, slowS) = slow
        (dataP, dataS) = data
        (scp, scs) = sc
        indrp = np.where(dataP[:, 0] == evID[ev])[0]
        rcv_evP = rcv[dataP[indrp, 2].astype(int) - 1, :]
        nstP = indrp.size
        T0p = np.kron(hypo[indh, 1], np.ones([nstP, 1]))
        indrs = np.where(dataS[:, 0] == evID[ev])[0]
        rcv_evS = rcv[dataS[indrs, 2].astype(int) - 1, :]
        nstS = indrs.size
        T0s = np.kron(hypo[indh, 1], np.ones([nstS, 1]))
        Txp = np.kron(hypo[indh, 2:], np.ones([nstP, 1]))
        Txs = np.kron(hypo[indh, 2:], np.ones([nstS, 1]))
        if par.use_sc:
            scp_ev = scp[dataP[indrp, 2].astype(int) - 1, :]
            scs_ev = scs[dataS[indrs, 2].astype(int) - 1, :]
        else:
            scp_ev = np.zeros([nstP, 1])
            scs_ev = np.zeros([nstS, 1])
        srcp = np.hstack((ev*np.ones([nstP, 1]), T0p + scp_ev, Txp))
        srcs = np.hstack((ev*np.ones([nstS, 1]), T0s + scs_ev, Txs))
        tcalp, raysP = Mesh3D.raytrace(source=srcp, rcv=rcv_evP, slowness=slowP,
                                       aggregate_src=False, compute_L=False,
                                       return_rays=True)
        tcals, raysS = Mesh3D.raytrace(source=srcs, rcv=rcv_evS, slowness=slowS,
                                       aggregate_src=False, compute_L=False,
                                       return_rays=True)
        slowP_0 = Mesh3D.get_s0(srcp)
        slowS_0 = Mesh3D.get_s0(srcs)
        Hi = np.ones((nstP + nstS, 4))
        for nr in range(nstP):
            rayi = raysP[nr]
            if rayi.shape[0] == 1:
                continue
            slw0 = slowP_0[nr]
            dx = rayi[1, 0] - hypo[indh, 2]
            dy = rayi[1, 1] - hypo[indh, 3]
            dz = rayi[1, 2] - hypo[indh, 4]
            ds = np.sqrt(dx * dx + dy * dy + dz * dz)
            Hi[nr, 1] = -dx * slw0 / ds
            Hi[nr, 2] = -dy * slw0 / ds
            Hi[nr, 3] = -dz * slw0 / ds
        for nr in range(nstS):
            rayi = raysS[nr]
            if rayi.shape[0] == 1:
                continue
            slw0 = slowS_0[nr]
            dx = rayi[1, 0] - hypo[indh, 2]
            dy = rayi[1, 1] - hypo[indh, 3]
            dz = rayi[1, 2] - hypo[indh, 4]
            ds = np.sqrt(dx * dx + dy * dy + dz * dz)
            Hi[nr, 1] = -dx * slw0 / ds
            Hi[nr, 2] = -dy * slw0 / ds
            Hi[nr, 3] = -dz * slw0 / ds
        tcal = np.hstack((tcalp, tcals))
        res = np.hstack((dataP[indrp, 1], dataS[indrs, 1])) - tcal
        convrays = np.where(tcal != 0)[0]
        if convrays.size < (nstP + nstS):
            res = res[convrays]
            Hi = Hi[convrays, :]
    elif len(slow) == 1:
        indr = np.where(data[0][:, 0] == evID[ev])[0]
        rcv_ev = rcv[data[0][indr, 2].astype(int) - 1, :]
        if par.use_sc:
            sc_ev = sc[data[0][indr, 2].astype(int) - 1]
        else:
            sc_ev = 0.
        nst = indr.size
        T0 = np.kron(hypo[indh, 1], np.ones([nst, 1]))
        Tx = np.kron(hypo[indh, 2:], np.ones([nst, 1]))
        src = np.hstack((ev*np.ones([nst, 1]), T0+sc_ev, Tx))

        tcal, rays = Mesh3D.raytrace(source=src, rcv=rcv_ev, slowness=slow[0],
                                     aggregate_src=False, compute_L=False,
                                     return_rays=True)
        slow_0 = Mesh3D.get_s0(src)
        Hi = np.ones([nst, 4])
        for nr in range(nst):
            rayi = rays[nr]
            if rayi.shape[0] == 1:  # unconverged ray
                continue
            slw0 = slow_0[nr]
            dx = rayi[1, 0] - hypo[indh, 2]
            dy = rayi[1, 1] - hypo[indh, 3]
            dz = rayi[1, 2] - hypo[indh, 4]
            ds = np.sqrt(dx * dx + dy * dy + dz * dz)
            Hi[nr, 1] = -dx * slw0 / ds
            Hi[nr, 2] = -dy * slw0 / ds
            Hi[nr, 3] = -dz * slw0 / ds
        convrays = np.where(tcal != 0)[0]
        res = data[0][indr, 1] - tcal
        if convrays.size < nst:
            res = res[convrays]
            Hi = Hi[convrays, :]
    N = res.shape[0]
    try:
        Q = np.linalg.inv(Hi.T @ Hi)
    except np.linalg.linalg.LinAlgError:
        if par.verbose:
            print("ill-conditioned Jacobian matrix")
            sys.stdout.flush()
        U, S, V = np.linalg.svd(Hi.T @ Hi)
        Q = V.T @ np.diag(1./(S + 1.e-9)) @ U.T
    eigenVals, eigenVec = np.linalg.eig(Q[:3, :3])
    ind = np.argsort(eigenVals)
    if varData:
        s2 = 1
        varData[0] += [np.sum(res**2)]
        varData[1] += [N]
    else:
        s2 = np.sum(res**2) / (N - 4)
    alpha = 1 - par.p
    coef = scps.t.ppf(1 - alpha / 2., N - 4)
    axis1 = np.sqrt(eigenVals[ind[2]] * s2) * coef * eigenVec[:, ind[2]]
    axis2 = np.sqrt(eigenVals[ind[1]] * s2) * coef * eigenVec[:, ind[1]]
    axis3 = np.sqrt(eigenVals[ind[0]] * s2) * coef * eigenVec[:, ind[0]]
    to_confInterv = np.sqrt(Q[-1, -1] * s2) * coef
    return to_confInterv, axis1, axis2, axis3


def jntHypoVel_T(data, caldata, Vinit, cells, nodes, rcv, Hypo0,
                 par, threads=1, vPoints=np.array([]), basename='Vel'):
    """
    Joint hypocenter-velicoty inversion from P wave arrival time data
    parametrized using the velocity model.

    Parameters
    ----------
     data : np.ndarray, shape(arrival time number, 3)
          Arrival times and corresponding receivers for each event..
     caldata : np.ndarray, shape(number of calibration shots, 3)
          Calibration shot data.
     Vinit : np.ndarray, shape(nnodes,1) or (1,1)
          Initial velocity model.
     cells : np.ndarray of int, shape (cell number, 4)
          Indices of nodes forming the cells.
     nodes : np.ndarray, shape (nnodes, 3)
          Node coordinates.
     rcv : np.ndarray, shape (receiver number,3)
          Coordinates of receivers.
     Hypo0 : np.ndarray, shape(event number, 5)
          First guesses of the hypocenter coordinates (must be all diffirent).
     par : instance of the class Parameters
          The inversion parameters.
     threads : int, optional
          Thread number. The default is 1.
     vPoints :  np.ndarray, shape(point number,4), optional
          Known velocity points. The default is np.array([]).
     basename : string, optional
          The filename used to save the output file. The default is 'Vel'.

     Returns
     -------
     output : python dictionary
          It contains the estimated hypocenter coordinates and their origin times,
          static correction values, velocity model, convergence states,
          parameter uncertainty and residual norm in each iteration.

     """

    if par.verbose:
        print(par)
        print('inversion involves the velocity model\n')
        sys.stdout.flush()
    if par.use_sc:
        nstation = rcv.shape[0]
    else:
        nstation = 0
    Static_Corr = np.zeros([nstation, 1])
    nnodes = nodes.shape[0]
    # observed traveltimes
    if data.shape[0] > 0:
        evID = np.unique(data[:, 0]).astype(int)
        tObserved = data[:, 1]
        numberOfEvents = evID.size
    else:
        tObserved = np.array([])
        numberOfEvents = 0
    rcvData = np.zeros([data.shape[0], 3])
    for ev in range(numberOfEvents):
        indr = np.where(data[:, 0] == evID[ev])[0]
        rcvData[indr] = rcv[data[indr, 2].astype(int) - 1, :]
    # calibration data
    if caldata.shape[0] > 0:
        calID = np.unique(caldata[:, 0])
        ncal = calID.size
        time_calibration = caldata[:, 1]
        TxCalib = np.zeros((caldata.shape[0], 5))
        TxCalib[:, 2:] = caldata[:, 3:]
        TxCalib[:, 0] = caldata[:, 0]
        rcvCalib = np.zeros([caldata.shape[0], 3])
        if par.use_sc:
            Msc_cal = []
        for nc in range(ncal):
            indr = np.where(caldata[:, 0] == calID[nc])[0]
            rcvCalib[indr] = rcv[caldata[indr, 2].astype(int) - 1, :]
            if par.use_sc:
                Msc_cal.append(sp.csr_matrix(
                     (np.ones([indr.size, ]),
                      (range(indr.size), caldata[indr, 2]-1)),
                     shape=(indr.size, nstation)))
    else:
        ncal = 0
        time_calibration = np.array([])
    # initial velocity model
    if Vinit.size == 1:
        Velocity = Vinit * np.ones([nnodes, 1])
        Slowness = 1. / Velocity
    elif Vinit.size == nnodes:
        Velocity = Vinit
        Slowness = 1. / Velocity
    else:
        print("invalid Velocity Model\n")
        sys.stdout.flush()
        return 0
    # used threads
    nThreadsSystem = cpu_count()
    nThreads = np.min((threads, nThreadsSystem))

    global Mesh3D, Dimensions
    Mesh3D = tmesh.Mesh3d(nodes, tetra=cells, method='DSPM', cell_slowness=0,
                          n_threads=nThreads, n_secondary=2, n_tertiary=1,
                          process_vel=1, radius_factor_tertiary=2,
                          translate_grid=1)
    Mesh3D.set_slowness(Slowness)
    Dimensions = np.empty(6)
    Dimensions[0] = min(nodes[:, 0])
    Dimensions[1] = max(nodes[:, 0])
    Dimensions[2] = min(nodes[:, 1])
    Dimensions[3] = max(nodes[:, 1])
    Dimensions[4] = min(nodes[:, 2])
    Dimensions[5] = max(nodes[:, 2])
    # Hypocenter
    if numberOfEvents > 0 and Hypo0.shape[0] != numberOfEvents:
        print("invalid Hypocenters0 format\n")
        sys.stdout.flush()
        return 0
    else:
        Hypocenters = Hypo0.copy()

    ResidueNorm = np.zeros([par.maxit])
    if par.invert_vel:
        if par.use_sc:
            U = sp.bsr_matrix(
                np.vstack((np.zeros([nnodes, 1]), np.ones([nstation, 1]))))
            nbre_param = nnodes + nstation
            if par.max_sc > 0. and par.max_sc < 1.:
                N = sp.bsr_matrix(
                      np.hstack((np.zeros([nstation, nnodes]), np.eye(nstation))))
                NtN = (1. / par.max_sc**2) * N.T.dot(N)
        else:
            U = sp.csr_matrix(np.zeros([nnodes, 1]))
            nbre_param = nnodes
        # build matrix D
        if vPoints.size > 0:
            if par.verbose:
                print('\nBuilding velocity data point matrix D\n')
                sys.stdout.flush()
            D = Mesh3D.compute_D(vPoints[:, 2:])
            D = sp.hstack((D, sp.csr_matrix((D.shape[0], nstation)))).tocsr()
            DtD = D.T @ D
            nD = spl.norm(DtD)
        # Build regularization matrix
        if par.verbose:
            print('\n...Building regularization matrix K\n')
            sys.stdout.flush()
        kx, ky, kz = Mesh3D.compute_K(order=2, taylor_order=2,
                                      weighting=1, squared=0,
                                      s0inside=0, additional_points=3)
        KX = sp.hstack((kx, sp.csr_matrix((nnodes, nstation))))
        KX_Square = KX.transpose().dot(KX)
        KY = sp.hstack((ky, sp.csr_matrix((nnodes, nstation))))
        KY_Square = KY.transpose().dot(KY)
        KZ = sp.hstack((kz, sp.csr_matrix((nnodes, nstation))))
        KZ_Square = KZ.transpose().dot(KZ)
        KtK = KX_Square + KY_Square + par.wzK * KZ_Square
        nK = spl.norm(KtK)
    if nThreads == 1:
        hypo_convergence = list(np.zeros(numberOfEvents, dtype=bool))
    else:
        manager = Manager()
        hypo_convergence = manager.list(np.zeros(numberOfEvents, dtype=bool))
    for i in range(par.maxit):
        if par.verbose:
            print("Iteration N : {0:d}\n".format(i + 1))
            sys.stdout.flush()
        if par.invert_vel:
            if par.verbose:
                print('Iteration {0:d} - Updating velocity model\n'.format(i + 1))
                print("Updating penalty vector\n")
                sys.stdout.flush()
            # Build vector C
            cx = kx.dot(Velocity)
            cy = ky.dot(Velocity)
            cz = kz.dot(Velocity)
            # build matrix P and dP
            indVmin = np.where(Velocity < par.Vpmin)[0]
            indVmax = np.where(Velocity > par.Vpmax)[0]
            indPinality = np.hstack([indVmin, indVmax])
            dPinality_V = np.hstack(
                [-par.PAp * np.ones(indVmin.size), par.PAp * np.ones(indVmax.size)])
            pinality_V = np.vstack(
                [par.PAp * (par.Vpmin - Velocity[indVmin]), par.PAp *
                 (Velocity[indVmax] - par.Vpmax)])
            d_Pinality = sp.csr_matrix(
                (dPinality_V, (indPinality, indPinality)), shape=(
                    nnodes, nbre_param))
            Pinality = sp.csr_matrix(
                 (pinality_V.reshape([-1, ]),
                  (indPinality, np.zeros([indPinality.shape[0]]))),
                 shape=(nnodes, 1))
            if par.verbose:
                print('Penalties applied at {0:d} nodes\n'.format(
                        dPinality_V.size))
                print('...Start Raytracing\n')
                sys.stdout.flush()

            if numberOfEvents > 0:
                sources = np.empty((data.shape[0], 5))
                if par.use_sc:
                    sc_data = np.empty((data.shape[0], ))
                for ev in np.arange(numberOfEvents):
                    indr = np.where(data[:, 0] == evID[ev])[0]
                    indh = np.where(Hypocenters[:, 0] == evID[ev])[0]
                    sources[indr, :] = Hypocenters[indh, :]
                    if par.use_sc:
                        sc_data[indr] = Static_Corr[data[indr, 2].astype(int)
                                                    - 1, 0]
                if par.use_sc:
                    sources[:, 1] += sc_data
                    tt, rays, M0 = Mesh3D.raytrace(source=sources,
                                                   rcv=rcvData, slowness=None,
                                                   aggregate_src=False,
                                                   compute_L=True, return_rays=True)
                else:
                    tt, rays, M0 = Mesh3D.raytrace(source=sources,
                                                   rcv=rcvData, slowness=None,
                                                   aggregate_src=False,
                                                   compute_L=True, return_rays=True)
                v0 = 1. / Mesh3D.get_s0(sources)
                if par.verbose:
                    inconverged = np.where(tt == 0)[0]
                    for icr in inconverged:
                        print('\033[43m' +
                              '\nWarning: raypath failed to converge for even '
                              'N {0:d} :({1:5.4f},{2:5.4f},{3:5.4f}) and receiver '
                              'N {4:d} :({5:5.4f},{6:5.4f},{7:5.4f})\n'.format(
                                   int(data[icr, 0]), sources[icr, 2],
                                   sources[icr, 3], sources[icr, 4],
                                   int(data[icr, 2]), rcvData[icr, 0],
                                   rcvData[icr, 1], rcvData[icr, 2])
                              + '\033[0m')
                        print('\033[43m' + 'ray will be temporary removed' +
                              '\033[0m')
                        sys.stdout.flush()
            else:
                tt = np.array([])

            if ncal > 0:
                if par.use_sc:
                    TxCalib[:, 1] = Static_Corr[caldata[:, 2].astype(int) - 1, 0]
                    tt_Calib, Mcalib = Mesh3D.raytrace(
                         source=TxCalib, rcv=rcvCalib, slowness=None,
                         aggregate_src=False, compute_L=True, return_rays=False)
                else:
                    tt_Calib, Mcalib = Mesh3D.raytrace(
                         source=TxCalib, rcv=rcvCalib, slowness=None,
                         aggregate_src=False, compute_L=True, return_rays=False)

                if par.verbose:
                    inconverged = np.where(tt_Calib == 0)[0]
                    for icr in inconverged:
                        print('\033[43m' +
                              '\nWarning: raypath failed to converge '
                              'for calibration shot N '
                              '{0:d} :({1:5.4f},{2:5.4f},{3:5.4f}) and receiver'
                              ' N {4:d} :({5:5.4f},{6:5.4f},{7:5.4f})\n'.format(
                                   int(caldata[icr, 0]), TxCalib[icr, 2],
                                   TxCalib[icr, 3], TxCalib[icr, 4],
                                   int(caldata[icr, 2]), rcvCalib[icr, 0],
                                   rcvCalib[icr, 1], rcvCalib[icr, 2])
                              + '\033[0m')
                        print('\033[43m' + 'ray will be temporary removed' +
                              '\033[0m')
                        sys.stdout.flush()
            else:
                tt_Calib = np.array([])

            Resid = tObserved - tt
            convrayData = np.where(tt != 0)[0]
            convrayClib = np.where(tt_Calib != 0)[0]
            if Resid.size == 0:
                Residue = time_calibration[convrayClib] - tt_Calib[convrayClib]
            else:
                Residue = np.hstack(
                     (np.zeros([np.count_nonzero(tt) - 4 * numberOfEvents]),
                      time_calibration[convrayClib] - tt_Calib[convrayClib]))
            ResidueNorm[i] = np.linalg.norm(np.hstack(
                (Resid[convrayData], time_calibration[convrayClib] -
                 tt_Calib[convrayClib])))
            if par.verbose:
                print('...Building matrix M\n')
                sys.stdout.flush()
            M = sp.csr_matrix((0, nbre_param))
            ir = 0
            for even in range(numberOfEvents):
                indh = np.where(Hypocenters[:, 0] == evID[even])[0]
                indr = np.where(data[:, 0] == evID[even])[0]
                Mi = M0[even]
                nst_ev = Mi.shape[0]
                Hi = np.ones([indr.size, 4])
                for nr in range(indr.size):
                    rayi = rays[indr[nr]]
                    if rayi.shape[0] == 1:
                        continue
                    vel0 = v0[indr[nr]]
                    dx = rayi[1, 0] - Hypocenters[indh[0], 2]
                    dy = rayi[1, 1] - Hypocenters[indh[0], 3]
                    dz = rayi[1, 2] - Hypocenters[indh[0], 4]
                    ds = np.sqrt(dx * dx + dy * dy + dz * dz)
                    Hi[nr, 1] = -dx / (vel0 * ds)
                    Hi[nr, 2] = -dy / (vel0 * ds)
                    Hi[nr, 3] = -dz / (vel0 * ds)
                convrays = np.where(tt[indr] != 0)[0]
                if convrays.shape[0] < nst_ev:
                    Hi = Hi[convrays, :]
                    nst_ev = convrays.size
                Q, _ = np.linalg.qr(Hi, mode='complete')
                Ti = sp.csr_matrix(Q[:, 4:])
                Ti = Ti.T
                if par.use_sc:
                    Lsc = sp.csr_matrix((np.ones(nst_ev,),
                                         (range(nst_ev),
                                          data[indr[convrays], 2] - 1)),
                                        shape=(nst_ev, nstation))
                    Mi = sp.hstack((Mi, Lsc))
                Mi = sp.csr_matrix(Ti @ Mi)
                M = sp.vstack([M, Mi])
                Residue[ir:ir + (nst_ev - 4)] = Ti.dot(Resid[indr[convrays]])
                ir += nst_ev - 4
            for evCal in range(len(Mcalib)):
                Mi = Mcalib[evCal]
                if par.use_sc:
                    indrCal = np.where(caldata[:, 0] == calID[evCal])[0]
                    convraysCal = np.where(tt_Calib[indrCal] != 0)[0]
                    Mi = sp.hstack((Mi, Msc_cal[evCal][convraysCal]))
                M = sp.vstack([M, Mi])
            if par.verbose:
                print('Assembling matrices and solving system\n')
                sys.stdout.flush()
            S = np.sum(Static_Corr)
            term1 = (M.T).dot(M)
            nM = spl.norm(term1[:nnodes, :nnodes])
            term2 = (d_Pinality.T).dot(d_Pinality)
            nP = spl.norm(term2)
            term3 = U.dot(U.T)
            λ = par.λ * nM / nK
            if nP != 0:
                γ = par.γ * nM / nP
            else:
                γ = par.γ
            A = term1 + λ * KtK + γ * term2 + term3
            if par.use_sc and par.max_sc > 0. and par.max_sc < 1.:
                A += NtN
            term1 = (M.T).dot(Residue)
            term1 = term1.reshape([-1, 1])
            term2 = (KX.T).dot(cx) + (KY.T).dot(cy) + par.wzK * (KZ.T).dot(cz)
            term3 = (d_Pinality.T).dot(Pinality)
            term4 = U.dot(S)
            b = term1 - λ * term2 - γ * term3 - term4
            if vPoints.size > 0:
                α = par.α * nM / nD
                A += α * DtD
                b += α * D.T @ (vPoints[:, 1].reshape(-1, 1) -
                                D[:, :nnodes] @ Velocity)
            x = spl.minres(A, b, tol=1.e-8)
            deltam = x[0].reshape(-1, 1)
            # update velocity vector and static correction
            dVmax = np.max(abs(deltam[:nnodes]))
            if dVmax > par.dVp_max:
                deltam[:nnodes] *= par.dVp_max / dVmax
            if par.use_sc and par.max_sc > 0. and par.max_sc < 1.:
                sc_mean = np.mean(abs(deltam[nnodes:]))
                if sc_mean > par.max_sc * np.mean(abs(Residue)):
                    deltam[nnodes:] *= par.max_sc * np.mean(abs(Residue)) / sc_mean
            Velocity += np.matrix(deltam[:nnodes])
            Slowness = 1. / Velocity
            Static_Corr += deltam[nnodes:]
            if par.saveVel == 'all':
                if par.verbose:
                    print('...Saving Velocity models\n')
                    sys.stdout.flush()
                try:
                    msh2vtk(nodes, cells, Velocity, basename +
                            'it{0}.vtk'.format(i + 1))
                except ImportError:
                    print('vtk module is not installed\n')
                    sys.stdout.flush()
            elif par.saveVel == 'last' and i == par.maxit - 1:
                try:
                    msh2vtk(nodes, cells, Velocity, basename + '.vtk')
                except ImportError:
                    print('vtk module is not installed\n')
                    sys.stdout.flush()
                #######################################
                    # relocate Hypocenters
                #######################################
        Mesh3D.set_slowness(Slowness)
        if numberOfEvents > 0:
            print("\nIteration N {0:d} : Relocation of events\n".format(i + 1))
            sys.stdout.flush()
            if nThreads == 1:
                for ev in range(numberOfEvents):
                    Hypocenters[ev, :] = _hypo_relocation(
                        ev, evID, Hypocenters, data, rcv,
                        Static_Corr, hypo_convergence, par)
            else:
                p = mp.get_context("fork").Pool(processes=nThreads)
                updatedHypo = p.starmap(_hypo_relocation,
                                        [(int(ev), evID, Hypocenters, data,
                                          rcv, Static_Corr, hypo_convergence,
                                          par)for ev in range(numberOfEvents)])
                p.close()  # pool won't take any new tasks
                p.join()
                Hypocenters = np.array([updatedHypo])[0]
            # Calculate the hypocenter parameter uncertainty
    uncertnty = []
    if par.uncertainty and numberOfEvents > 0:
        print("\nUncertainty evaluation\n")
        sys.stdout.flush()
        # estimate data variance
        if nThreads == 1:
            varData = [[], []]
            for ev in range(numberOfEvents):
                uncertnty.append(
                    _uncertaintyEstimat(ev, evID, Hypocenters, (data,), rcv,
                                        Static_Corr, (Slowness,), par, varData))
        else:
            varData = manager.list([[], []])
            with Pool(processes=nThreads) as p:
                uncertnty = p.starmap(
                     _uncertaintyEstimat,
                     [(int(ev), evID, Hypocenters, (data, ),
                       rcv, Static_Corr, (Slowness, ), par,
                       varData) for ev in range(numberOfEvents)])
                p.close()  # pool won't take any new tasks
                p.join()
        sgmData = np.sqrt(np.sum(varData[0]) /
                          (np.sum(varData[1]) - 4 *
                           numberOfEvents -
                           Static_Corr.size))
        for ic in range(numberOfEvents):
            uncertnty[ic] = tuple([sgmData * x for x in uncertnty[ic]])
    output = OrderedDict()
    output['Hypocenters'] = Hypocenters
    output['Convergence'] = list(hypo_convergence)
    output['Uncertainties'] = uncertnty
    output['Velocity'] = Velocity
    output['Sts_Corrections'] = Static_Corr
    output['Residual_norm'] = ResidueNorm

    return output


def jntHyposlow_T(data, caldata, Vinit, cells, nodes, rcv, Hypo0,
                  par, threads=1, vPoints=np.array([]), basename='Slowness'):
    """
    Joint hypocenter-velicoty inversion from P wave arrival time data
    parametrized using the slowness model.

    Parameters
    ----------
     data : np.ndarray, shape(arrival time number, 3)
          Arrival times and corresponding receivers for each event.
     caldata : np.ndarray, shape(number of calibration shots, 6)
          Calibration shot data.
     Vinit : np.ndarray, shape(nnodes,1) or (1,1)
          Initial velocity model.
     Cells : np.ndarray of int, shape (cell number, 4)
          Indices of nodes forming the cells.
     nodes : np.ndarray, shape (nnodes, 3)
          Node coordinates.
     rcv : np.ndarray, shape (receiver number,3)
          Coordinates of receivers.
     Hypo0 : np.ndarray, shape(event number, 5)
          First guesses of the hypocenter coordinates (must be all diffirent).
     par : instance of the class Parameters
          The inversion parameters.
     threads : int, optional
          Thread number. The default is 1.
     vPoints : np.ndarray, shape(point number,4), optional
          Known velocity points. The default is np.array([]).
     basename : string, optional
          The filename used to save the output files. The default is 'Slowness'.

    Returns
    -------
     output : python dictionary
          It contains the estimated hypocenter coordinates and their origin times,
          static correction values, velocity model, convergence states,
          parameter uncertainty and residual norm in each iteration.


    """
    if par.verbose:
        print(par)
        print('inversion involves the slowness model\n')
        sys.stdout.flush()
    if par.use_sc:
        nstation = rcv.shape[0]
    else:
        nstation = 0
    Static_Corr = np.zeros([nstation, 1])
    nnodes = nodes.shape[0]
    # observed traveltimes
    if data.shape[0] > 0:
        evID = np.unique(data[:, 0]).astype(int)
        tObserved = data[:, 1]
        numberOfEvents = evID.size
    else:
        tObserved = np.array([])
        numberOfEvents = 0
    rcvData = np.zeros([data.shape[0], 3])
    for ev in range(numberOfEvents):
        indr = np.where(data[:, 0] == evID[ev])[0]
        rcvData[indr] = rcv[data[indr, 2].astype(int) - 1, :]

    # get calibration data
    if caldata.shape[0] > 0:
        calID = np.unique(caldata[:, 0])
        ncal = calID.size
        time_calibration = caldata[:, 1]
        TxCalib = np.zeros((caldata.shape[0], 5))
        TxCalib[:, 2:] = caldata[:, 3:]
        TxCalib[:, 0] = caldata[:, 0]
        rcvCalib = np.zeros([caldata.shape[0], 3])
        if par.use_sc:
            Msc_cal = []
        for nc in range(ncal):
            indr = np.where(caldata[:, 0] == calID[nc])[0]
            rcvCalib[indr] = rcv[caldata[indr, 2].astype(int) - 1, :]
            if par.use_sc:
                Msc_cal.append(sp.csr_matrix(
                     (np.ones([indr.size, ]),
                      (range(indr.size), caldata[indr, 2]-1)),
                     shape=(indr.size, nstation)))
    else:
        ncal = 0
        time_calibration = np.array([])
    # initial velocity model
    if Vinit.size == 1:
        Slowness = 1. / (Vinit * np.ones([nnodes, 1]))
    elif Vinit.size == nnodes:
        Slowness = 1. / Vinit
    else:
        print("invalid Velocity Model")
        sys.stdout.flush()
        return 0

    # Hypocenter
    if numberOfEvents > 0 and Hypo0.shape[0] != numberOfEvents:
        print("invalid Hypocenters0 format\n")
        sys.stdout.flush()
        return 0
    else:
        Hypocenters = Hypo0.copy()
    # number of threads
    nThreadsSystem = cpu_count()
    nThreads = np.min((threads, nThreadsSystem))

    global Mesh3D, Dimensions
    # build mesh object
    Mesh3D = tmesh.Mesh3d(nodes, tetra=cells, method='DSPM', cell_slowness=0,
                          n_threads=nThreads, n_secondary=2, n_tertiary=1,
                          radius_factor_tertiary=2, translate_grid=1)
    Mesh3D.set_slowness(Slowness)

    Dimensions = np.empty(6)
    Dimensions[0] = min(nodes[:, 0])
    Dimensions[1] = max(nodes[:, 0])
    Dimensions[2] = min(nodes[:, 1])
    Dimensions[3] = max(nodes[:, 1])
    Dimensions[4] = min(nodes[:, 2])
    Dimensions[5] = max(nodes[:, 2])
    ResidueNorm = np.zeros([par.maxit])
    if par.invert_vel:
        if par.use_sc:
            U = sp.bsr_matrix(np.vstack((np.zeros([nnodes, 1]),
                                         np.ones([nstation, 1]))))
            nbre_param = nnodes + nstation
            if par.max_sc > 0. and par.max_sc < 1.:
                N = sp.bsr_matrix(
                     np.hstack((np.zeros([nstation, nnodes]), np.eye(nstation))))
                NtN = (1. / par.max_sc**2) * N.T.dot(N)
        else:
            U = sp.csr_matrix(np.zeros([nnodes, 1]))
            nbre_param = nnodes
        # build matrix D
        if vPoints.size > 0:
            if par.verbose:
                print('\nBuilding velocity data point matrix D\n')
                sys.stdout.flush()
            D = Mesh3D.compute_D(vPoints[:, 2:])
            D = sp.hstack((D, sp.csr_matrix((D.shape[0], nstation)))).tocsr()
            DtD = D.T @ D
            nD = spl.norm(DtD)
        # Build regularization matrix
        if par.verbose:
            print('\n...Building regularization matrix K\n')
            sys.stdout.flush()
        kx, ky, kz = Mesh3D.compute_K(order=2, taylor_order=2,
                                      weighting=1, squared=0,
                                      s0inside=0, additional_points=3)
        KX = sp.hstack((kx, sp.csr_matrix((nnodes, nstation))))
        KX_Square = KX.transpose().dot(KX)
        KY = sp.hstack((ky, sp.csr_matrix((nnodes, nstation))))
        KY_Square = KY.transpose().dot(KY)
        KZ = sp.hstack((kz, sp.csr_matrix((nnodes, nstation))))
        KZ_Square = KZ.transpose().dot(KZ)
        KtK = KX_Square + KY_Square + par.wzK * KZ_Square
        nK = spl.norm(KtK)

    if nThreads == 1:
        hypo_convergence = list(np.zeros(numberOfEvents, dtype=bool))
    else:
        manager = Manager()
        hypo_convergence = manager.list(np.zeros(numberOfEvents, dtype=bool))
    for i in range(par.maxit):
        if par.verbose:
            print("\nIteration N : {0:d}\n".format(i + 1))
            sys.stdout.flush()
        if par.invert_vel:
            if par.verbose:
                print(
                    '\nIteration {0:d} - Updating velocity model\n'.format(i + 1))
                print("\nUpdating penalty vector\n")
                sys.stdout.flush()
            # Build vector C
            cx = kx.dot(Slowness)
            cy = ky.dot(Slowness)
            cz = kz.dot(Slowness)
            # build matrix P and dP
            indSmin = np.where(Slowness < 1. / par.Vpmax)[0]
            indSmax = np.where(Slowness > 1. / par.Vpmin)[0]
            indPinality = np.hstack([indSmin, indSmax])
            dPinality_V = np.hstack(
                [-par.PAp * np.ones(indSmin.size), par.PAp * np.ones(indSmax.size)])
            pinality_V = np.vstack([par.PAp *
                                    (1. / par.Vpmax -
                                     Slowness[indSmin]), par.PAp *
                                    (Slowness[indSmax] -
                                        1. / par.Vpmin)])
            d_Pinality = sp.csr_matrix(
                (dPinality_V, (indPinality, indPinality)), shape=(
                    nnodes, nbre_param))
            Pinality = sp.csr_matrix((
                 pinality_V.reshape([-1, ]),
                 (indPinality, np.zeros([indPinality.shape[0]]))),
                 shape=(nnodes, 1))
            if par.verbose:
                print('\nPenalties applied at {0:d} nodes\n'.format(
                        dPinality_V.size))
                print('...Start Raytracing\n')
                sys.stdout.flush()

            if numberOfEvents > 0:
                sources = np.empty((data.shape[0], 5))
                if par.use_sc:
                    sc_data = np.empty((data.shape[0], ))
                for ev in np.arange(numberOfEvents):
                    indr = np.where(data[:, 0] == evID[ev])[0]
                    indh = np.where(Hypocenters[:, 0] == evID[ev])[0]
                    sources[indr, :] = Hypocenters[indh, :]
                    if par.use_sc:
                        sc_data[indr] = Static_Corr[data[indr, 2].astype(int)
                                                    - 1, 0]
                if par.use_sc:
                    sources[:, 1] += sc_data
                    tt, rays, M0 = Mesh3D.raytrace(source=sources,
                                                   rcv=rcvData, slowness=None,
                                                   aggregate_src=False,
                                                   compute_L=True, return_rays=True)
                else:
                    tt, rays, M0 = Mesh3D.raytrace(source=sources,
                                                   rcv=rcvData, slowness=None,
                                                   aggregate_src=False,
                                                   compute_L=True, return_rays=True)
                slow_0 = Mesh3D.get_s0(sources)

                if par.verbose:
                    inconverged = np.where(tt == 0)[0]
                    for icr in inconverged:
                        print('\033[43m' +
                              '\nWarning: raypath failed to converge for even '
                              'N {0:d} :({1:5.4f},{2:5.4f},{3:5.4f}) and receiver'
                              ' N {4:d} :({5:5.4f},{6:5.4f},{7:5.4f})\n'.format(
                                   int(data[icr, 0]), sources[icr, 2],
                                   sources[icr, 3], sources[icr, 4],
                                   int(data[icr, 2]), rcvData[icr, 0],
                                   rcvData[icr, 1], rcvData[icr, 2]) +
                              '\033[0m')
                        print('\033[43m' + 'ray will be temporary removed' +
                              '\033[0m')
                        sys.stdout.flush()
            else:
                tt = np.array([])

            if ncal > 0:
                if par.use_sc:
                    # add static corrections for each station
                    TxCalib[:, 1] = Static_Corr[caldata[:, 2].astype(int) - 1, 0]
                    tt_Calib, Mcalib = Mesh3D.raytrace(
                         source=TxCalib, rcv=rcvCalib, slowness=None,
                         aggregate_src=False, compute_L=True, return_rays=False)
                else:
                    tt_Calib,  Mcalib = Mesh3D.raytrace(
                         source=TxCalib, rcv=rcvCalib, slowness=None,
                         aggregate_src=False, compute_L=True, return_rays=False)
                if par.verbose:
                    inconverged = np.where(tt_Calib == 0)[0]
                    for icr in inconverged:
                        print('\033[43m' +
                              '\nWarning: raypath failed to converge'
                              'for calibration shot N '
                              '{0:d} :({1:5.4f},{2:5.4f},{3:5.4f}) and receiver'
                              ' N {4:d} :({5:5.4f},{6:5.4f},{7:5.4f})\n'.format(
                                   int(caldata[icr, 0]), TxCalib[icr, 2],
                                   TxCalib[icr, 3], TxCalib[icr, 4],
                                   int(caldata[icr, 2]), rcvCalib[icr, 0],
                                   rcvCalib[icr, 1], rcvCalib[icr, 2]) +
                              '\033[0m')
                        print('\033[43m' + 'ray will be temporary removed' +
                              '\033[0m')
                        sys.stdout.flush()
            else:
                tt_Calib = np.array([])
            Resid = tObserved - tt
            convrayData = np.where(tt != 0)[0]
            convrayClib = np.where(tt_Calib != 0)[0]
            if Resid.size == 0:
                Residue = time_calibration[convrayClib] - tt_Calib[convrayClib]
            else:
                Residue = np.hstack((np.zeros([np.count_nonzero(tt) -
                                               4 * numberOfEvents]),
                                     time_calibration[convrayClib]
                                     - tt_Calib[convrayClib]))
            ResidueNorm[i] = np.linalg.norm(np.hstack(
                (Resid[convrayData], time_calibration[convrayClib] -
                 tt_Calib[convrayClib])))
            if par.verbose:
                print('\n...Building matrix M\n')
                sys.stdout.flush()
            M = sp.csr_matrix((0, nbre_param))
            ir = 0
            for even in range(numberOfEvents):
                indh = np.where(Hypocenters[:, 0] == evID[even])[0]
                indr = np.where(data[:, 0] == evID[even])[0]
                Mi = M0[even]
                nst_ev = Mi.shape[0]
                Hi = np.ones([indr.size, 4])
                for nr in range(indr.size):
                    rayi = rays[indr[nr]]
                    if rayi.shape[0] == 1:
                        continue
                    slw0 = slow_0[indr[nr]]
                    dx = rayi[1, 0] - Hypocenters[indh[0], 2]
                    dy = rayi[1, 1] - Hypocenters[indh[0], 3]
                    dz = rayi[1, 2] - Hypocenters[indh[0], 4]
                    ds = np.sqrt(dx * dx + dy * dy + dz * dz)
                    Hi[nr, 1] = -slw0 * dx / ds
                    Hi[nr, 2] = -slw0 * dy / ds
                    Hi[nr, 3] = -slw0 * dz / ds
                convrays = np.where(tt[indr] != 0)[0]
                if convrays.shape[0] < indr.size:
                    Hi = Hi[convrays, :]
                Q, _ = np.linalg.qr(Hi, mode='complete')
                Ti = sp.csr_matrix(Q[:, 4:])
                Ti = Ti.T
                if par.use_sc:
                    Lsc = sp.csr_matrix((np.ones(nst_ev,),
                                         (range(nst_ev),
                                          data[indr[convrays], 2] - 1)),
                                        shape=(nst_ev, nstation))
                    Mi = sp.hstack((Mi, Lsc))
                Mi = sp.csr_matrix(Ti @ Mi)
                M = sp.vstack([M, Mi])
                Residue[ir:ir + (nst_ev - 4)] = Ti.dot(Resid[indr[convrays]])
                ir += nst_ev - 4
            for evCal in range(len(Mcalib)):
                Mi = Mcalib[evCal]
                if par.use_sc:
                    indrCal = np.where(caldata[:, 0] == calID[evCal])[0]
                    convraysCal = np.where(tt_Calib[indrCal] != 0)[0]
                    Mi = sp.hstack((Mi, Msc_cal[evCal][convraysCal]))
                M = sp.vstack([M, Mi])
            if par.verbose:
                print('Assembling matrices and solving system\n')
                sys.stdout.flush()
            S = np.sum(Static_Corr)
            term1 = (M.T).dot(M)
            nM = spl.norm(term1[:nnodes, :nnodes])
            term2 = (d_Pinality.T).dot(d_Pinality)
            nP = spl.norm(term2)
            term3 = U.dot(U.T)
            λ = par.λ * nM / nK
            if nP != 0:
                γ = par.γ * nM / nP
            else:
                γ = par.γ
            A = term1 + λ * KtK + γ * term2 + term3
            if par.use_sc and par.max_sc > 0. and par.max_sc < 1.:
                A += NtN
            term1 = (M.T).dot(Residue)
            term1 = term1.reshape([-1, 1])
            term2 = (KX.T).dot(cx) + (KY.T).dot(cy) + par.wzK * (KZ.T).dot(cz)
            term3 = (d_Pinality.T).dot(Pinality)
            term4 = U.dot(S)
            b = term1 - λ * term2 - γ * term3 - term4
            if vPoints.size > 0:
                α = par.α * nM / nD
                A += α * DtD
                b += α * D.T @ (1. / (vPoints[:, 1].reshape(-1, 1)) -
                                D[:, :nnodes] @ Slowness)
            x = spl.minres(A, b, tol=1.e-8)
            deltam = x[0].reshape(-1, 1)
            # update velocity vector and static correction
            deltaV_max = np.max(
                abs(1. / (Slowness + deltam[:nnodes]) - 1. / Slowness))
            if deltaV_max > par.dVp_max:
                print('\n...Rescale P slowness vector\n')
                sys.stdout.flush()
                L1 = np.max(deltam[:nnodes] / (-par.dVp_max *
                                               (Slowness**2) /
                                               (1 + par.dVp_max * Slowness)))
                L2 = np.max(deltam[:nnodes] / (par.dVp_max *
                                               (Slowness**2) /
                                               (1 - par.dVp_max * Slowness)))
                deltam[:nnodes] /= np.max([L1, L2])
                print('P wave: maximum ds = {0:4.3f}, '
                      'maximum dV = {1:4.3f}\n'.format(max(abs(
                           deltam[:nnodes]))[0], np.max(
                                abs(1. / (Slowness + deltam[:nnodes])
                                    - 1. / Slowness))))
                sys.stdout.flush()
            if par.use_sc and par.max_sc > 0. and par.max_sc < 1.:
                sc_mean = np.mean(abs(deltam[nnodes:]))
                if sc_mean > par.max_sc * np.mean(abs(Residue)):
                    deltam[nnodes:] *= par.max_sc * np.mean(abs(Residue)) / sc_mean
            Slowness += np.matrix(deltam[:nnodes])
            Mesh3D.set_slowness(Slowness)
            Static_Corr += deltam[nnodes:]
            if par.saveVel == 'all':
                if par.verbose:
                    print('...Saving Velocity models')
                try:
                    msh2vtk(nodes, cells, 1. / Slowness, basename +
                            'it{0}.vtk'.format(i + 1))
                except ImportError:
                    print('vtk module is not installed or encouters problems')
            elif par.saveVel == 'last' and i == par.maxit - 1:
                try:
                    msh2vtk(nodes, cells, 1. / Slowness, basename + '.vtk')
                except ImportError:
                    print('vtk module is not installed or encouters problems')

                #######################################
                    # relocate Hypocenters
                #######################################
        if numberOfEvents > 0:
            print("\nIteration N {0:d} : Relocation of events".format(
                i + 1) + '\n')
            sys.stdout.flush()
            if nThreads == 1:
                for ev in range(numberOfEvents):
                    Hypocenters[ev, :] = _hypo_relocation(
                        ev, evID, Hypocenters, data, rcv, Static_Corr,
                        hypo_convergence, par)
            else:
                with Pool(processes=nThreads) as p:
                    updatedHypo = p.starmap(_hypo_relocation,
                                            [(int(ev), evID, Hypocenters, data,
                                              rcv, Static_Corr, hypo_convergence,
                                              par) for ev in range(numberOfEvents)])
                    p.close()  # pool won't take any new tasks
                    p.join()
                Hypocenters = np.array([updatedHypo])[0]
            # Calculate the hypocenter parameter uncertainty
    uncertnty = []
    if par.uncertainty and numberOfEvents > 0:
        print("\nUncertainty evaluation\n")
        sys.stdout.flush()
        # estimate data variance
        if nThreads == 1:
            varData = [[], []]
            for ev in range(numberOfEvents):
                uncertnty.append(_uncertaintyEstimat(ev, evID, Hypocenters,
                                                     (data,), rcv, Static_Corr,
                                                     (Slowness,), par, varData))
        else:
            varData = manager.list([[], []])
            with Pool(processes=nThreads) as p:
                uncertnty = p.starmap(_uncertaintyEstimat,
                                      [(int(ev), evID, Hypocenters, (data,),
                                        rcv, Static_Corr, (Slowness,), par,
                                        varData) for ev in range(numberOfEvents)])
                p.close()  # pool won't take any new tasks
                p.join()

        sgmData = np.sqrt(np.sum(varData[0]) /
                          (np.sum(varData[1]) - 4 * numberOfEvents -
                           Static_Corr.size))
        for ic in range(numberOfEvents):
            uncertnty[ic] = tuple([sgmData * x for x in uncertnty[ic]])
    output = OrderedDict()
    output['Hypocenters'] = Hypocenters
    output['Convergence'] = list(hypo_convergence)
    output['Uncertainties'] = uncertnty
    output['Velocity'] = 1. / Slowness
    output['Sts_Corrections'] = Static_Corr
    output['Residual_norm'] = ResidueNorm
    return output


def jntHypoVelPS_T(obsData, calibdata, Vinit, cells, nodes, rcv, Hypo0,
                   par, threads=1, vPnts=(np.array([]), np.array([])),
                   basename='Vel'):
    """

     Joint hypocenter-velocity inversion from P- and S-wave arrival time data
     parametrized using the velocity models.

     Parameters
     ----------
     obsData : tuple of two np.ndarrays (shape(observed data number, 3))
          Observed arrival time data of P- and S-waves.
     calibdata : tuple of two np.ndarrays (shape (number of calibration shots, 5))
          Calibration data of P- and S-waves.
     Vinit : tuple of np.ndarrays (shape (nnodes, 1) or (1,1))
          Initial velocity models of P- and S-waves.
     cells : np.ndarray of int, shape (cell number, 4)
          Indices of nodes forming the cells.
     nodes : np.ndarray, shape (nnodes, 3)
          Node coordinates.
     rcv : np.ndarray, shape (receiver number,3)
          Coordinates of receivers.
     Hypo0 : np.ndarray, shape(event number, 5)
          First guesses of the hypocenter coordinates (must be all diffirent).
     par : instance of the class Parameters
          The inversion parameters.
     threads : int, optional
          Thread number. The default is 1.
     vPnts : tuple of two np.ndarrays, optional
          Known velocity points of P- and S-waves.
          The default is (np.array([]), np.array([])).
     basename : string, optional
          The filename used to save the output files. The default is 'Vel'.

     Raises
     ------
     ValueError
          If the Vs/Vp ratio is inverted instead of Vs model and some
          known velocity points are given for the S wave and not for the P wave.

     Returns
     -------
     output : python dictionary
          It contains the estimated hypocenter coordinates and their origin times,
          static correction values, velocity models of P- and S-waves,
          hypocenter convergence states, parameter uncertainty and
          residual norm in each iteration.

     """

    if par.verbose:
        print(par)
        print('inversion involves the velocity model\n')
        sys.stdout.flush()
    if par.use_sc:
        nstation = rcv.shape[0]
    else:
        nstation = 0
    scP = np.zeros([nstation, 1])
    scS = np.zeros([nstation, 1])
    nnodes = nodes.shape[0]
    # observed traveltimes
    dataP, dataS = obsData
    data = np.vstack([dataP, dataS])
    if data.size > 0:
        evID = np.unique(data[:, 0])
        tObserved = data[:, 1]
        numberOfEvents = evID.size
    else:
        tObserved = np.array([])
        numberOfEvents = 0
    rcvData_P = np.zeros([dataP.shape[0], 3])
    rcvData_S = np.zeros([dataS.shape[0], 3])
    for ev in range(numberOfEvents):
        indr = np.where(dataP[:, 0] == evID[ev])[0]
        rcvData_P[indr] = rcv[dataP[indr, 2].astype(int) - 1, :]
        indr = np.where(dataS[:, 0] == evID[ev])[0]
        rcvData_S[indr] = rcv[dataS[indr, 2].astype(int) - 1, :]
    # calibration data
    caldataP, caldataS = calibdata
    if caldataP.size * caldataS.size > 0:
        caldata = np.vstack([caldataP, caldataS])
        calID = np.unique(caldata[:, 0])
        ncal = calID.size
        nttcalp = caldataP.shape[0]
        nttcals = caldataS.shape[0]
        time_calibration = caldata[:, 1]
        TxCalibP = np.zeros((caldataP.shape[0], 5))
        TxCalibP[:, 0] = caldataP[:, 0]
        TxCalibP[:, 2:] = caldataP[:, 3:]
        TxCalibS = np.zeros((caldataS.shape[0], 5))
        TxCalibS[:, 0] = caldataS[:, 0]
        TxCalibS[:, 2:] = caldataS[:, 3:]
        rcvCalibP = np.zeros([nttcalp, 3])
        rcvCalibS = np.zeros([nttcals, 3])
        if par.use_sc:
            Mscp_cal = []
            Mscs_cal = []
        for nc in range(ncal):
            indr = np.where(caldataP[:, 0] == calID[nc])[0]
            rcvCalibP[indr] = rcv[caldataP[indr, 2].astype(int) - 1, :]
            if par.use_sc:
                Mscp_cal.append(sp.csr_matrix((np.ones([indr.size, ]),
                                               (range(indr.size), caldataP[indr, 2]
                                                - 1)), shape=(indr.size, nstation)))
            indr = np.where(caldataS[:, 0] == calID[nc])[0]
            rcvCalibS[indr] = rcv[caldataS[indr, 2].astype(int) - 1, :]
            if par.use_sc:
                Mscs_cal.append(sp.csr_matrix((np.ones([indr.size, ]),
                                               (range(indr.size), caldataS[indr, 2]
                                                - 1)), shape=(indr.size, nstation)))
    else:
        ncal = 0
        time_calibration = np.array([])
    # set number of threads
    nThreadsSystem = cpu_count()
    nThreads = np.min((threads, nThreadsSystem))
    global Mesh3D, Dimensions
    Mesh3D = tmesh.Mesh3d(nodes, tetra=cells, method='DSPM', cell_slowness=0,
                          n_threads=nThreads, n_secondary=2, n_tertiary=1,
                          process_vel=True, radius_factor_tertiary=2,
                          translate_grid=1)
    # initial velocity models for P and S waves
    Vpint, Vsint = Vinit
    if Vpint.size == 1:
        Velp = Vpint * np.ones([nnodes, 1])
        SlowP = 1. / Velp
    elif Vpint.size == nnodes:
        Velp = Vpint
        SlowP = 1. / Velp
    else:
        print("invalid P Velocity model\n")
        sys.stdout.flush()
        return 0
    if Vsint.size == 1:
        Vels = Vsint * np.ones([nnodes, 1])
        SlowS = 1. / Vels
    elif Vsint.size == nnodes:
        Vels = Vsint
        SlowS = 1. / Vels
    else:
        print("invalid S Velocity model\n")
        sys.stdout.flush()
        return 0
    if par.invert_VpVs:
        VsVp = Vels / Velp
        Velocity = np.vstack((Velp, VsVp))
    else:
        Velocity = np.vstack((Velp, Vels))
    # initial parameters Hyocenters0 and origin times
    if numberOfEvents > 0 and Hypo0.shape[0] != numberOfEvents:
        print("invalid Hypocenters0 format\n")
        sys.stdout.flush()
        return 0
    else:
        Hypocenters = Hypo0.copy()

    Dimensions = np.empty(6)
    Dimensions[0] = min(nodes[:, 0])
    Dimensions[1] = max(nodes[:, 0])
    Dimensions[2] = min(nodes[:, 1])
    Dimensions[3] = max(nodes[:, 1])
    Dimensions[4] = min(nodes[:, 2])
    Dimensions[5] = max(nodes[:, 2])
    if par.invert_vel:
        if par.use_sc:
            U = sp.hstack((sp.csr_matrix(np.vstack(
                 (np.zeros([2 * nnodes, 1]), np.ones([nstation, 1]),
                  np.zeros([nstation, 1])))), sp.csr_matrix(np.vstack(
                       (np.zeros([2 * nnodes + nstation, 1]),
                        np.ones([nstation, 1]))))))
            nbre_param = 2 * (nnodes + nstation)
            if par.max_sc > 0. and par.max_sc < 1.:
                N = sp.bsr_matrix(np.hstack(
                    (np.zeros([2 * nstation, 2 * nnodes]), np.eye(2 * nstation))))
                NtN = (1. / par.max_sc**2) * N.T.dot(N)
        else:
            U = sp.csr_matrix(np.zeros([2 * nnodes, 2]))
            nbre_param = 2 * nnodes
        # calculate statistical moments of VpVs ratio
        if par.stig != 0.:
            momnts = np.zeros([4, ])
            if par.invert_VpVs:
                Ratio = caldataP[:, 1] / caldataS[:, 1]  # Ratio=Vs/Vp
            else:
                Ratio = caldataS[:, 1] / caldataP[:, 1]  # Ratio=Vp/Vs
            for m in np.arange(4):
                if m == 0:
                    momnts[m] = Ratio.mean() * nnodes
                else:
                    momnts[m] = scps.moment(Ratio, m + 1) * nnodes

        # build matrix D
        vPoints_p, vPoints_s = vPnts
        if vPoints_p.shape[0] > 0 or vPoints_s.shape[0] > 0:
            if par.invert_VpVs:
                for i in np.arange(vPoints_s.shape[0]):
                    dist = np.sqrt(np.sum((vPoints_p[:, 2:] -
                                           vPoints_s[i, 2:])**2, axis=1))
                    indp = np.where(dist < 1.e-5)[0]
                    if indp.size > 0:
                        vPoints_s[i, 1] /= vPoints_p[indp, 1]  # Vs/Vp
                    else:
                        raise ValueError('Missing Vp data point for Vs data '
                                         'at ({0:f}, {1:f}, {2:f})'.format(
                                              vPoints_s[i, 2], vPoints_s[i, 3],
                                              vPoints_s[i, 4]))
                        sys.stdout.flush()
                vPoints = np.vstack((vPoints_p, vPoints_s))
                if par.verbose:
                    print('Building velocity data point matrix D\n')
                    sys.stdout.flush()
                Ds = Mesh3D.compute_D(vPoints_s[:, 2:])
                D = sp.block_diag((Ds, Ds)).tocsr()
                D = sp.hstack((D, sp.csr_matrix((D.shape[0], 2*nstation)))).tocsr()
                DtD = D.T @ D
                nD = spl.norm(DtD)
            else:
                vPoints = np.vstack((vPoints_p, vPoints_s))
                Dp = Mesh3D.compute_D(vPoints_p[:, 2:])
                Ds = Mesh3D.compute_D(vPoints_s[:, 2:])
                D = sp.block_diag((Dp, Ds)).tocsr()
                D = sp.hstack((D, sp.csr_matrix((D.shape[0], 2*nstation)))).tocsr()
                DtD = D.T @ D
                nD = spl.norm(DtD)
        else:
            vPoints = np.array([])
        # Build regularization matrix
        if par.verbose:
            print('\n...Building regularization matrix K\n')
            sys.stdout.flush()
        kx, ky, kz = Mesh3D.compute_K(order=2, taylor_order=2,
                                      weighting=1, squared=0,
                                      s0inside=0, additional_points=3)
        kx = sp.block_diag((kx, kx))
        ky = sp.block_diag((ky, ky))
        kz = sp.block_diag((kz, kz))
        KX = sp.hstack((kx, sp.csr_matrix((2 * nnodes, 2 * nstation))))
        KX_Square = KX.transpose() @ KX
        KY = sp.hstack((ky, sp.csr_matrix((2 * nnodes, 2 * nstation))))
        KY_Square = KY.transpose() @ KY
        KZ = sp.hstack((kz, sp.csr_matrix((2 * nnodes, 2 * nstation))))
        KZ_Square = KZ.transpose() @ KZ
        KtK = KX_Square + KY_Square + par.wzK * KZ_Square
        nK = spl.norm(KtK)
    if par.invert_VpVs:
        VsVpmax = 1. / par.VpVsmin
        VsVpmin = 1. / par.VpVsmax
    if nThreads == 1:
        hypo_convergence = list(np.zeros(numberOfEvents, dtype=bool))
    else:
        manager = Manager()
        hypo_convergence = manager.list(np.zeros(numberOfEvents, dtype=bool))
    ResidueNorm = np.zeros([par.maxit])
    for i in range(par.maxit):
        if par.verbose:
            print("Iteration N : {0:d}\n".format(i + 1))
            sys.stdout.flush()
        if par.invert_vel:
            if par.verbose:
                print('\nIteration {0:d} - Updating velocity model\n'.format(i + 1))
                print("Updating penalty vector\n")
                sys.stdout.flush()
            # Build vector C
            cx = kx.dot(Velocity)
            cy = ky.dot(Velocity)
            cz = kz.dot(Velocity)
            # build matrix P and dP
            indVpmin = np.where(Velocity[:nnodes] < par.Vpmin)[0]
            indVpmax = np.where(Velocity[:nnodes] > par.Vpmax)[0]
            if par.invert_VpVs:
                indVsVpmin = np.where(Velocity[nnodes:] < VsVpmin)[0] + nnodes
                indVsVpmax = np.where(Velocity[nnodes:] > VsVpmax)[0] + nnodes

                pinality_V = np.vstack([par.PAp * (par.Vpmin - Velocity[indVpmin]),
                                        par.PAp * (Velocity[indVpmax] - par.Vpmax),
                                        par.Pvpvs * (VsVpmin -
                                                     Velocity[indVsVpmin]),
                                        par.Pvpvs * (Velocity[indVsVpmax] -
                                                     VsVpmax)])

                dPinality_V = np.hstack([-par.PAp * np.ones(indVpmin.size),
                                         par.PAp * np.ones(indVpmax.size),
                                         -par.Pvpvs * np.ones(indVsVpmin.size),
                                         par.Pvpvs * np.ones(indVsVpmax.size)])
                indPinality = np.hstack(
                    [indVpmin, indVpmax, indVsVpmin, indVsVpmax])
            else:
                indVsmin = np.where(Velocity[nnodes:] < par.Vsmin)[0] + nnodes
                indVsmax = np.where(Velocity[nnodes:] > par.Vsmax)[0] + nnodes

                pinality_V = np.vstack([par.PAp *
                                        (par.Vpmin -
                                         Velocity[indVpmin]), par.PAp *
                                        (Velocity[indVpmax] -
                                         par.Vpmax), par.PAs *
                                        (par.Vsmin -
                                            Velocity[indVsmin]), par.PAs *
                                        (Velocity[indVsmax] -
                                            par.Vsmax)])

                dPinality_V = np.hstack([-par.PAp * np.ones(indVpmin.size),
                                         par.PAp * np.ones(indVpmax.size),
                                         -par.PAs * np.ones(indVsmin.size),
                                         par.PAs * np.ones(indVsmax.size)])
                indPinality = np.hstack(
                    [indVpmin, indVpmax, indVsmin, indVsmax])
                if par.VpVsmin and par.VpVsmax:
                    indvpvs_min = np.where(Velp / Vels <= par.VpVsmin)[0]
                    indvpvs_max = np.where(Velp / Vels >= par.VpVsmax)[0]
                    if par.verbose and indvpvs_max.size > 0:
                        print('\n{0:d} nodes have Vp/Vs ratio higher than the '
                              'upper VpVs limit\n'.format(indvpvs_max.size))
                        sys.stdout.flush()
                    if par.verbose and indvpvs_min.size > 0:
                        print('\n{0:d} nodes have Vp/Vs ratio lower than the lower '
                              'VpVs limit\n'.format(indvpvs_min.size))
                        sys.stdout.flush()
                    indPnltvpvs = np.hstack([indvpvs_min, indvpvs_max])
                    no = 2  # order or pinality function
                    pinlt_vpvs = np.vstack([par.Pvpvs *
                                            (par.VpVsmin - Velp[indvpvs_min] /
                                             Vels[indvpvs_min])**no,
                                            par.Pvpvs * (Velp[indvpvs_max] /
                                                         Vels[indvpvs_max] -
                                                         par.VpVsmax)**no])

                    PinltVpVs = sp.csr_matrix((pinlt_vpvs.reshape(
                        [-1, ]), (indPnltvpvs, np.zeros([indPnltvpvs.shape[0]]))),
                         shape=(nnodes, 1))
                    dPinltVpVsind = (np.hstack([indvpvs_min, indvpvs_max,
                                                indvpvs_min, indvpvs_max]),
                                     np.hstack([indvpvs_min, indvpvs_max,
                                                indvpvs_min + nnodes,
                                                indvpvs_max + nnodes]))

                    dPinltVpVs_V = np.vstack(
                         (-par.Pvpvs / Vels[indvpvs_min] * no *
                          (par.VpVsmin - Velp[indvpvs_min] /
                           Vels[indvpvs_min])**(no - 1), par.Pvpvs /
                          Vels[indvpvs_max] * no *
                          (Velp[indvpvs_max] / Vels[indvpvs_max] -
                           par.VpVsmax)**(no - 1),
                          par.Pvpvs * Velp[indvpvs_min] /
                          (Vels[indvpvs_min]**2) * no *
                          (par.VpVsmin - Velp[indvpvs_min] /
                           Vels[indvpvs_min])**(no - 1),
                          -par.Pvpvs * Velp[indvpvs_max] /
                          (Vels[indvpvs_max]**2) * no *
                          (Velp[indvpvs_max] / Vels[indvpvs_max] -
                           par.VpVsmax)**(no - 1)))

                    dPinltVpVs = sp.csr_matrix((dPinltVpVs_V.reshape(
                        [-1, ]), dPinltVpVsind), shape=(nnodes, nbre_param))
            d_Pinality = sp.csr_matrix(
                (dPinality_V, (indPinality, indPinality)),
                shape=(2 * nnodes, nbre_param))
            Pinality = sp.csr_matrix((pinality_V.reshape(
                [-1, ]), (indPinality, np.zeros([indPinality.shape[0]]))),
                 shape=(2 * nnodes, 1))
            if par.verbose:
                print('P wave Penalties were applied at {0:d} nodes\n'.format(
                        indVpmin.shape[0] + indVpmax.shape[0]))
                sys.stdout.flush()
                if par.invert_VpVs:
                    print(
                        'Vs/Vp ratio Penalties were applied '
                        'at {0:d} nodes\n'.format(indVsVpmin.shape[0] +
                                                  indVsVpmax.shape[0]))
                    sys.stdout.flush()
                else:
                    print('S wave Penalties were applied at {0:d} nodes\n'.format(
                            indVsmin.shape[0] + indVsmax.shape[0]))
                    sys.stdout.flush()
                print('...Start Raytracing\n')
                sys.stdout.flush()
            if numberOfEvents > 0:
                sourcesp = np.empty((dataP.shape[0], 5))
                if par.use_sc:
                    scp_data = np.zeros((dataP.shape[0], 1))
                for ev in np.arange(numberOfEvents):
                    indrp = np.where(dataP[:, 0] == evID[ev])[0]
                    indh = np.where(Hypocenters[:, 0] == evID[ev])[0]
                    sourcesp[indrp, :] = Hypocenters[indh, :]
                    if par.use_sc:
                        scp_data[indrp, :] = scP[dataP[indrp, 2].astype(int) - 1]
                if par.use_sc:
                    sourcesp[:, 1] += scp_data[:, 0]
                    ttp, raysp, M0p = Mesh3D.raytrace(
                         source=sourcesp, rcv=rcvData_P,
                         slowness=SlowP, aggregate_src=False,
                         compute_L=True, return_rays=True)

                else:
                    ttp, raysp, M0p = Mesh3D.raytrace(
                         source=sourcesp, rcv=rcvData_P,
                         slowness=SlowP, aggregate_src=False,
                         compute_L=True, return_rays=True)
                if par.verbose:
                    inconverged = np.where(ttp == 0)[0]
                    for icr in inconverged:
                        print('\033[43m' +
                              '\nWarning: raypath failed to converge for even '
                              'N {0:d} :({1:5.4f},{2:5.4f},{3:5.4f}) and receiver '
                              'N {4:d} :({5:5.4f},{6:5.4f},{7:5.4f})\n'.format(
                                   int(dataP[icr, 0]), sourcesp[icr, 2],
                                   sourcesp[icr, 3], sourcesp[icr, 4],
                                   int(dataP[icr, 2]), rcvData_P[icr, 0],
                                   rcvData_P[icr, 1], rcvData_P[icr, 2]) +
                              '\033[0m')
                        print('\033[43m' + 'ray will be temporary removed' +
                              '\033[0m')
                        sys.stdout.flush()
                v0p = 1. / Mesh3D.get_s0(sourcesp)

                sourcesS = np.empty((dataS.shape[0], 5))
                if par.use_sc:
                    scs_data = np.zeros((dataS.shape[0], 1))
                for ev in np.arange(numberOfEvents):
                    indrs = np.where(dataS[:, 0] == evID[ev])[0]
                    indh = np.where(Hypocenters[:, 0] == evID[ev])[0]
                    sourcesS[indrs, :] = Hypocenters[indh, :]
                    if par.use_sc:
                        scs_data[indrs, :] = scS[dataS[indrs, 2].astype(int) - 1]
                if par.use_sc:
                    sourcesS[:, 1] += scs_data[:, 0]
                    tts, rayss, M0s = Mesh3D.raytrace(
                         source=sourcesS, rcv=rcvData_S,
                         slowness=SlowS, aggregate_src=False,
                         compute_L=True, return_rays=True)
                else:
                    tts, rayss, M0s = Mesh3D.raytrace(
                         source=sourcesS, rcv=rcvData_S,
                         slowness=SlowS, aggregate_src=False,
                         compute_L=True, return_rays=True)
                if par.verbose:
                    inconverged = np.where(tts == 0)[0]
                    for icr in inconverged:
                        print('\033[43m' +
                              '\nWarning: raypath failed to converge for even '
                              'N {0:d} :({1:5.4f},{2:5.4f},{3:5.4f}) and receiver '
                              'N {4:d} :({5:5.4f},{6:5.4f},{7:5.4f})\n'.format(
                                   int(dataS[icr, 0]), sourcesS[icr, 0],
                                   sourcesS[icr, 1], sourcesS[icr, 2],
                                   int(dataS[icr, 2]), rcvData_S[icr, 0],
                                   rcvData_S[icr, 1], rcvData_S[icr, 2]) +
                              '\033[0m')
                        print('\033[43m' + 'ray will be temporary removed' +
                              '\033[0m')
                        sys.stdout.flush()
                v0s = 1. / Mesh3D.get_s0(sourcesS)
                tt = np.hstack((ttp, tts))
                v0 = np.hstack((v0p, v0s))
                rays = raysp + rayss
            else:
                tt = np.array([])
            if nttcalp > 0:
                if par.use_sc:
                    scp_cal = scP[caldataP[:, 2].astype(int) - 1, 0]
                    TxCalibP[:, 1] = scp_cal
                    tt_Calibp, Mcalibp = Mesh3D.raytrace(
                         source=TxCalibP, rcv=rcvCalibP,
                         slowness=SlowP, aggregate_src=False,
                         compute_L=True, return_rays=False)
                else:
                    tt_Calibp, Mcalibp = Mesh3D.raytrace(
                         source=TxCalibP, rcv=rcvCalibP,
                         slowness=SlowP, aggregate_src=False,
                         compute_L=True, return_rays=False)
                if par.verbose:
                    inconverged = np.where(tt_Calibp == 0)[0]
                    for icr in inconverged:
                        print('\033[43m' + '\nWarning: raypath failed to converge '
                              'for calibration shot N {0:d} :({1:5.4f},{2:5.4f},'
                              '{3:5.4f}) and receiver N {4:d} :({5:5.4f},{6:5.4f},'
                              '{7:5.4f})\n'.format(
                                   int(caldataP[icr, 0]), TxCalibP[icr, 2],
                                   TxCalibP[icr, 3], TxCalibP[icr, 4],
                                   int(caldataP[icr, 2]), rcvCalibP[icr, 0],
                                   rcvCalibP[icr, 1], rcvCalibP[icr, 2]) +
                              '\033[0m')
                        print('\033[43m' + 'ray will be temporary removed' +
                              '\033[0m')
                        sys.stdout.flush()
            else:
                tt_Calibp = np.array([])
            if nttcals > 0:
                if par.use_sc:
                    scs_cal = scS[caldataS[:, 2].astype(int) - 1, 0]
                    TxCalibS[:, 1] = scs_cal
                    tt_Calibs, Mcalibs = Mesh3D.raytrace(
                         source=TxCalibS, rcv=rcvCalibS,
                         slowness=SlowS, aggregate_src=False,
                         compute_L=True, return_rays=False)
                else:
                    tt_Calibs, Mcalibs = Mesh3D.raytrace(
                         source=TxCalibS, rcv=rcvCalibS,
                         slowness=SlowS, aggregate_src=False,
                         compute_L=True, return_rays=False)
                if par.verbose:
                    inconverged = np.where(tt_Calibs == 0)[0]
                    for icr in inconverged:
                        print('\033[43m' + '\nWarning: raypath failed to converge '
                              'for calibration shot N {0:d} :({1:5.4f},{2:5.4f},'
                              '{3:5.4f}) and receiver N {4:d} :({5:5.4f},{6:5.4f},'
                              '{7:5.4f})\n'.format(
                                   int(caldataS[icr, 0]), TxCalibS[icr, 2],
                                   TxCalibS[icr, 3], TxCalibS[icr, 4],
                                   int(caldataS[icr, 2]), rcvCalibS[icr, 0],
                                   rcvCalibS[icr, 1], rcvCalibS[icr, 2]) +
                              '\033[0m')
                        print('\033[43m' + 'ray will be temporary removed' +
                              '\033[0m')
                        sys.stdout.flush()
            else:
                tt_Calibs = np.array([])
            tt_Calib = np.hstack((tt_Calibp, tt_Calibs))
            Resid = tObserved - tt
            convrayData = np.where(tt != 0)[0]
            convrayClib = np.where(tt_Calib != 0)[0]
            if Resid.size == 0:
                Residue = time_calibration[convrayClib] - tt_Calib[convrayClib]
            else:
                Residue = np.hstack(
                     (np.zeros([np.count_nonzero(tt) - 4 * numberOfEvents]),
                      time_calibration[convrayClib] - tt_Calib[convrayClib]))
            ResidueNorm[i] = np.linalg.norm(np.hstack(
                (Resid[convrayData], time_calibration[convrayClib] -
                 tt_Calib[convrayClib])))
            if par.verbose:
                print('...Building matrix M\n')
                sys.stdout.flush()
            M = sp.csr_matrix((0, nbre_param))
            ir = 0
            for even in range(numberOfEvents):
                Mpi = M0p[even]
                nst_ev = Mpi.shape[0]
                if par.use_sc:
                    indrp = np.where(dataP[:, 0] == evID[even])[0]
                    convrays = np.where(ttp[indrp] != 0)[0]
                    Lscp = sp.csr_matrix(
                         (np.ones(nst_ev,),
                          (range(nst_ev), dataP[indrp[convrays], 2] - 1)),
                         shape=(nst_ev, nstation))
                else:
                    Lscp = sp.csr_matrix((nst_ev, 0))
                Mpi = sp.hstack((Mpi, sp.csr_matrix((nst_ev, nnodes)),
                                 Lscp, sp.csr_matrix((nst_ev, nstation)))).tocsr()

                Msi = M0s[even]
                nst_ev = Msi.shape[0]
                if par.use_sc:
                    indrs = np.where(dataS[:, 0] == evID[even])[0]
                    convrays = np.where(tts[indrs] != 0)[0]
                    Lscs = sp.csr_matrix(
                         (np.ones(nst_ev,),
                          (range(nst_ev), dataS[indrs[convrays], 2] - 1)),
                         shape=(nst_ev, nstation))
                else:
                    Lscs = sp.csr_matrix((nst_ev, 0))
                if par.invert_VpVs:
                    dTsdVp = Msi.multiply(np.tile(VsVp, nst_ev).T)
                    dTsdVsVp = Msi.multiply(np.tile(Velp, nst_ev).T)
                    Msi = sp.hstack((dTsdVp, dTsdVsVp, sp.csr_matrix(
                        (nst_ev, nstation)), Lscs)).tocsr()
                else:
                    Msi = sp.hstack((sp.csr_matrix((nst_ev, nnodes)), Msi,
                                     sp.csr_matrix((nst_ev, nstation)),
                                     Lscs)).tocsr()
                Mi = sp.vstack((Mpi, Msi))
                indh = np.where(Hypocenters[:, 0] == evID[even])[0]
                indr = np.where(data[:, 0] == evID[even])[0]
                nst_ev = indr.size
                Hi = np.ones([nst_ev, 4])
                for nr in range(nst_ev):
                    rayi = rays[indr[nr]]
                    if rayi.shape[0] == 1:  # ray does not converge
                        continue
                    vel0 = v0[indr[nr]]
                    dx = rayi[1, 0] - Hypocenters[indh[0], 2]
                    dy = rayi[1, 1] - Hypocenters[indh[0], 3]
                    dz = rayi[1, 2] - Hypocenters[indh[0], 4]
                    ds = np.sqrt(dx * dx + dy * dy + dz * dz)
                    Hi[nr, 1] = -dx / (vel0 * ds)
                    Hi[nr, 2] = -dy / (vel0 * ds)
                    Hi[nr, 3] = -dz / (vel0 * ds)
                convrays = np.where(tt[indr] != 0)[0]
                if convrays.shape[0] < nst_ev:
                    Hi = Hi[convrays, :]
                    nst_ev = convrays.size
                if Hi.shape[0] < 4:
                    print('\n Wraning : even contains less than 4 rays')
                Q, _ = np.linalg.qr(Hi, mode='complete')
                Ti = sp.csr_matrix(Q[:, 4:])
                Ti = Ti.T
                Mi = sp.csr_matrix(Ti @ Mi)
                M = sp.vstack([M, Mi])
                Residue[ir:ir + (nst_ev - 4)] = Ti.dot(Resid[indr[convrays]])
                ir += nst_ev - 4
            for evCalp in range(len(Mcalibp)):
                Mpi = Mcalibp[evCalp]
                nst_evcal = Mpi.shape[0]
                if par.use_sc:
                    indrCalp = np.where(caldataP[:, 0] == calID[evCalp])[0]
                    convraysp = np.where(tt_Calibp[indrCalp] != 0)[0]
                    Mpi = sp.hstack((
                         Mpi, sp.csr_matrix((nst_evcal, nnodes)),
                         Mscp_cal[evCalp][convraysp],
                         sp.csr_matrix((nst_evcal, nstation)))).tocsr()
                else:
                    Mpi = sp.hstack(
                        (Mpi, sp.csr_matrix((nst_evcal, nnodes)))).tocsr()
                M = sp.vstack([M, Mpi])
            for evCals in range(len(Mcalibs)):
                Msi = Mcalibs[evCals]
                nst_evcal = Msi.shape[0]
                if par.invert_VpVs:
                    dTsdVp = Msi.multiply(np.tile(VsVp, nst_evcal).T)
                    dTsdVsVp = Msi.multiply(np.tile(Velp, nst_evcal).T)
                    if par.use_sc:
                        indrCals = np.where(caldataS[:, 0] == calID[evCals])[0]
                        convrayss = np.where(tt_Calibs[indrCals] != 0)[0]
                        Msi = sp.hstack((dTsdVp, dTsdVsVp,
                                         sp.csr_matrix((nst_evcal, nstation)),
                                         Mscs_cal[evCals][convrayss])).tocsr()
                    else:
                        Msi = Msi = sp.hstack((dTsdVp, dTsdVsVp)).tocsr()
                else:
                    if par.use_sc:
                        indrCals = np.where(caldataS[:, 0] == calID[evCals])[0]
                        convrayss = np.where(tt_Calibs[indrCals] != 0)[0]
                        Msi = sp.hstack((sp.csr_matrix((nst_evcal, nnodes)),
                                         Msi, sp.csr_matrix((nst_evcal, nstation)),
                                         Mscs_cal[evCals][convrayss])).tocsr()
                    else:
                        Msi = sp.hstack((sp.csr_matrix((nst_evcal, nnodes)),
                                         Msi)).tocsr()
                M = sp.vstack((M, Msi))
            if par.stig != 0.:
                dPr = sp.lil_matrix((4, nbre_param))
                Pr = np.zeros([4, 1])
                if par.invert_VpVs:
                    Gamma_mean = VsVp.mean()
                    for m in np.arange(4):
                        dPr[m, nnodes:2 * nnodes] = (
                             (m + 1) * (VsVp - Gamma_mean)**m).reshape([-1, ])
                        if m == 0:
                            Pr[m, 0] = momnts[m] - np.sum(VsVp)
                        else:
                            Pr[m, 0] = momnts[m] - np.sum((VsVp - Gamma_mean)**(m + 1))
                else:
                    Gamma_mean = (Velp / Vels).mean()
                    for m in np.arange(4):
                        dPr[m, :nnodes] = ((m + 1) *
                                           (Velp / Vels - Gamma_mean)**m
                                           / Vels).reshape([-1, ])
                        dPr[m, nnodes:2 * nnodes] = (-(m + 1) * (
                            Velp / Vels - Gamma_mean)**m * Velp /
                             (Vels**2)).reshape([-1, ])
                        if m == 0:
                            Pr[m, 0] = momnts[m] - np.sum(Velp / Vels)
                        else:
                            Pr[m, 0] = momnts[m] - np.sum((Velp / Vels - Gamma_mean)**(m + 1))
                dPr2 = dPr.T @ dPr
            if par.verbose:
                print('Assembling matrices and solving system\n')
                sys.stdout.flush()
            S = np.array([np.sum(scP), np.sum(scS)]).reshape([-1, 1])
            term1 = (M.T).dot(M)
            nM = spl.norm(term1[:2 * nnodes, :2 * nnodes])
            λ = par.λ * nM / nK

            term2 = (d_Pinality.T).dot(d_Pinality)
            nP = spl.norm(term2)
            if nP != 0:
                γ = par.γ * nM / nP
            else:
                γ = par.γ
            term3 = U.dot(U.T)
            A = term1 + λ * KtK + γ * term2 + term3
            if par.use_sc and par.max_sc > 0. and par.max_sc < 1.:
                A += NtN
            term1 = (M.T).dot(Residue.reshape(-1, 1))
            term2 = KX.T @ cx + KY.T @ cy + par.wzK * KZ.T @ cz
            term3 = d_Pinality.T @ Pinality
            term4 = U.dot(S)
            b = term1 - λ * term2 - γ * term3 - term4
            if par.stig != 0.:
                if spl.norm(dPr2) != 0:
                    stg = par.stig * nM / spl.norm(dPr2)
                    A += stg * dPr2
                    b += stg * dPr.T @ Pr
            if vPoints.size > 0:
                α = par.α * nM / nD
                A += α * DtD
                b += α * D.T @ (vPoints[:, 1].reshape(-1, 1) -
                                D[:, :2 * nnodes] @ Velocity)
            if not par.invert_VpVs and par.VpVsmin and par.VpVsmax:
                dPinltVpVs2 = dPinltVpVs.T @ dPinltVpVs
                if spl.norm(dPinltVpVs2) != 0:
                    γvpvs = par.γ_vpvs * nM / spl.norm(dPinltVpVs2)
                    A += γvpvs * dPinltVpVs2
                    b -= γvpvs * dPinltVpVs.T @ PinltVpVs
            x = spl.minres(A, b, tol=1.e-8)
            deltam = x[0].reshape(-1, 1)
            # update velocity vector and static correction
            deltaVp_max = np.max(abs(deltam[:nnodes]))
            if deltaVp_max > par.dVp_max:
                print('...Rescale P velocity\n')
                sys.stdout.flush()
                deltam[:nnodes] *= par.dVp_max / deltaVp_max
            if par.invert_VpVs:
                deltaVs_max = np.max(
                     abs((deltam[:nnodes] + Velp) *
                         (Velocity[nnodes:2 * nnodes] +
                          deltam[nnodes:2 * nnodes]) - Vels))
                if deltaVs_max > par.dVs_max:
                    print('...Rescale VsVp\n')
                    sys.stdout.flush()
                    L1 = np.max((deltam[nnodes:2 * nnodes] /
                                 ((-par.dVs_max + Vels) /
                                  (deltam[:nnodes] + Velp) -
                                  Velocity[nnodes:2 * nnodes])))
                    L2 = np.max((deltam[nnodes:2 * nnodes] /
                                 ((par.dVs_max + Vels) / (deltam[:nnodes] + Velp)
                                  - Velocity[nnodes:2 * nnodes])))
                    deltam[nnodes:2 * nnodes] /= np.max([L1, L2])
            else:
                deltaVs_max = np.max(abs(deltam[nnodes:2 * nnodes]))
                if deltaVs_max > par.dVs_max:
                    print('...Rescale S velocity\n')
                    sys.stdout.flush()
                    deltam[nnodes:2 * nnodes] *= par.dVs_max / deltaVs_max
            Velocity += np.matrix(deltam[:2 * nnodes])
            Velp = Velocity[:nnodes]
            if par.invert_VpVs:
                VsVp = Velocity[nnodes:2 * nnodes]
                Vels = VsVp * Velp
            else:
                Vels = Velocity[nnodes:2 * nnodes]
            if par.use_sc and par.max_sc > 0. and par.max_sc < 1.:
                sc_mean = np.mean(abs(deltam[2 * nnodes:]))
                if sc_mean > par.max_sc * np.mean(abs(Residue)):
                    deltam[2 * nnodes:] *= par.max_sc * np.mean(
                         abs(Residue)) / sc_mean
            SlowP = 1. / Velp
            SlowS = 1. / Vels
            scP += deltam[2 * nnodes:2 * nnodes + nstation]
            scS += deltam[2 * nnodes + nstation:]
            if par.saveVel == 'all':
                if par.verbose:
                    print(
                        '...Saving Velocity models of interation N: {0:d}\n'.format(
                            i + 1))
                    sys.stdout.flush()
                try:
                    msh2vtk(nodes, cells, Velp, basename +
                            '_Vp_it{0}.vtk'.format(i + 1))
                except ImportError:
                    print('cannot save P wave velocity model in format vtk')
                    sys.stdout.flush()
                try:
                    msh2vtk(nodes, cells, Vels, basename +
                            '_Vs_it{0}.vtk'.format(i + 1))
                except ImportError:
                    print('cannot save S wave velocity model in format vtk')
                    sys.stdout.flush()
                if par.invert_VpVs:
                    try:
                        msh2vtk(nodes, cells, VsVp, basename +
                                '_VsVp Ratio_it{0}.vtk'.format(i + 1))
                    except ImportError:
                        print('cannot save Vs/Vp ration model in format vtk')
                        sys.stdout.flush()
            elif par.saveVel == 'last' and i == par.maxit - 1:
                if par.verbose:
                    print('...Saving Velocity models of the last iteration\n')
                    sys.stdout.flush()
                try:
                    msh2vtk(nodes, cells, Velp, basename + '_Vp_final.vtk')
                except ImportError:
                    print('cannot save the final P wave '
                          'velocity model in format vtk')
                    sys.stdout.flush()
                try:
                    msh2vtk(nodes, cells, Vels, basename + '_Vs_final.vtk')
                except ImportError:
                    print('cannot save the final S wave '
                          'velocity model in format vtk')
                    sys.stdout.flush()
                if par.invert_VpVs:
                    try:
                        msh2vtk(nodes, cells, VsVp, basename +
                                '_VsVp Ratio_final.vtk')
                    except ImportError:
                        print('cannot save the final Vs/Vp model in format vtk')
                        sys.stdout.flush()
                #######################################
                        # relocate Hypocenters
                #######################################
        updatedHypo = []
        if numberOfEvents > 0:
            print(
                "\nIteration N {0:d} : Relocation of events".format(
                    i + 1) + '\n')
            sys.stdout.flush()
            if nThreads == 1:
                for ev in range(numberOfEvents):
                    updatedHypo.append(
                        _hypo_relocationPS(
                             ev, evID, Hypocenters, (dataP, dataS), rcv,
                             (scP, scS), hypo_convergence,
                             (SlowP, SlowS), par))
            else:
                with Pool(processes=nThreads) as p:
                    updatedHypo = p.starmap(
                         _hypo_relocationPS,
                         [(int(ev), evID, Hypocenters, (dataP, dataS),
                           rcv, (scP, scS), hypo_convergence, (SlowP, SlowS),
                           par) for ev in range(numberOfEvents)])
                    p.close()  # pool won't take any new tasks
                    p.join()
            Hypocenters = np.array([updatedHypo])[0]
            # Calculate the hypocenter parameter uncertainty
    uncertnty = []
    if par.uncertainty:
        print("Uncertainty evaluation \n")
        sys.stdout.flush()
        if nThreads == 1:
            varData = [[], []]
            for ev in range(numberOfEvents):
                uncertnty.append(
                    _uncertaintyEstimat(
                         ev, evID, Hypocenters,
                         (dataP, dataS), rcv, (scP, scS),
                         (SlowP, SlowS), par, varData))
        else:
            varData = manager.list([[], []])
            with Pool(processes=nThreads) as p:
                uncertnty = p.starmap(
                     _uncertaintyEstimat,
                     [(int(ev), evID, Hypocenters,
                       (dataP, dataS), rcv,
                       (scP, scS), (SlowP, SlowS),
                       par, varData)
                      for ev in range(numberOfEvents)])
                p.close()
                p.join()
        sgmData = np.sqrt(np.sum(varData[0]) /
                          (np.sum(varData[1]) -
                           4 * numberOfEvents -
                           scP.size - scS.size))
        for ic in range(numberOfEvents):
            uncertnty[ic] = tuple([sgmData * x for x in uncertnty[ic]])
    output = OrderedDict()
    output['Hypocenters'] = Hypocenters
    output['Convergence'] = list(hypo_convergence)
    output['Uncertainties'] = uncertnty
    output['P_velocity'] = Velp
    output['S_velocity'] = Vels
    output['P_StsCorrections'] = scP
    output['S_StsCorrections'] = scS
    output['Residual_norm'] = ResidueNorm
    return output


def jntHyposlowPS_T(obsData, calibdata, Vinit, cells, nodes, rcv, Hypo0,
                    par, threads=1, vPnts=(np.array([]), np.array([])),
                    basename='Slowness'):
    """
     Joint hypocenter-velocity inversion from P- and S-wave arrival time data
     parametrized using the slowness models.

     Parameters
     ----------
     obsData : tuple of two np.ndarrays (shape(observed data number, 3))
          Observed arrival time data of P- and S-waves.
     calibdata : tuple of two np.ndarrays (shape (calibration shot number, 6))
          Calibration data of P- and S-waves.
     Vinit : tuple of np.ndarrays (shape (nnodes, 1) or (1,1))
          Initial velocity models of P- and S-waves.
     cells : np.ndarray of int, shape (cell number, 4)
          Indices of nodes forming the cells.
     nodes : np.ndarray, shape (nnodes, 3)
          Node coordinates.
     rcv : np.ndarray, shape (receiver number,3)
          Coordinates of receivers.
     Hypo0 : np.ndarray, shape(event number, 5)
          First guesses of the hypocenter coordinates (must be all diffirent).
     par : instance of the class Parameters
          The inversion parameters.
     threads : int, optional
          Thread number. The default is 1.
     vPnts : tuple of two np.ndarrays, optional
          Known velocity points of P- and S-waves.
          The default is (np.array([]),np.array([])).
     basename : string, optional
          The filename used to save the output files. The default is 'Slowness'.

     Raises
     ------
     ValueError
          If the Ss/Sp ratio is inverted instead of Vs model and some
          known velocity points are given for the S wave and not for the P wave.

     Returns
     -------
     output : python dictionary
          It contains the estimated hypocenter coordinates and their origin times,
          static correction values, velocity models of P- and S-waves,
          hypocenter convergence states, parameter uncertainty and
          residual norm in each iteration.


     """

    if par.verbose:
        print(par)
        print('inversion involves the slowness model\n')
        sys.stdout.flush()
    if par.use_sc:
        nstation = rcv.shape[0]
    else:
        nstation = 0
    scP = np.zeros([nstation, 1])
    scS = np.zeros([nstation, 1])
    nnodes = nodes.shape[0]
    # observed traveltimes
    dataP, dataS = obsData
    data = np.vstack([dataP, dataS])
    if data.size > 0:
        evID = np.unique(data[:, 0])
        tObserved = data[:, 1]
        numberOfEvents = evID.size
    else:
        tObserved = np.array([])
        numberOfEvents = 0
    rcvData_P = np.zeros([dataP.shape[0], 3])
    rcvData_S = np.zeros([dataS.shape[0], 3])
    for ev in range(numberOfEvents):
        indr = np.where(dataP[:, 0] == evID[ev])[0]
        rcvData_P[indr] = rcv[dataP[indr, 2].astype(int) - 1, :]
        indr = np.where(dataS[:, 0] == evID[ev])[0]
        rcvData_S[indr] = rcv[dataS[indr, 2].astype(int) - 1, :]
    # calibration data
    caldataP, caldataS = calibdata
    if calibdata[0].size * calibdata[1].size > 0:
        caldata = np.vstack([caldataP, caldataS])
        calID = np.unique(caldata[:, 0])
        ncal = calID.size
        nttcalp = caldataP.shape[0]
        nttcals = caldataS.shape[0]
        time_calibration = caldata[:, 1]

        TxCalibP = np.zeros((caldataP.shape[0], 5))
        TxCalibP[:, 0] = caldataP[:, 0]
        TxCalibP[:, 2:] = caldataP[:, 3:]
        TxCalibS = np.zeros((caldataS.shape[0], 5))
        TxCalibS[:, 0] = caldataS[:, 0]
        TxCalibS[:, 2:] = caldataS[:, 3:]
        rcvCalibP = np.zeros([nttcalp, 3])
        rcvCalibS = np.zeros([nttcals, 3])
        if par.use_sc:
            Mscp_cal = []
            Mscs_cal = []
        for nc in range(ncal):
            indr = np.where(caldataP[:, 0] == calID[nc])[0]
            rcvCalibP[indr] = rcv[caldataP[indr, 2].astype(int) - 1, :]
            if par.use_sc:
                Mscp_cal.append(sp.csr_matrix((np.ones([indr.size, ]),
                                               (range(indr.size), caldataP[indr, 2]
                                                - 1)), shape=(indr.size, nstation)))
            indr = np.where(caldataS[:, 0] == calID[nc])[0]
            rcvCalibS[indr] = rcv[caldataS[indr, 2].astype(int) - 1, :]
            if par.use_sc:
                Mscs_cal.append(sp.csr_matrix((np.ones([indr.size, ]),
                                               (range(indr.size), caldataS[indr, 2]
                                                - 1)), shape=(indr.size, nstation)))
    else:
        ncal = 0
        time_calibration = np.array([])
    # initial velocity models for P and S waves
    Vpint, Vsint = Vinit
    if Vpint.size == 1:
        SlowP = np.ones([nnodes, 1]) / Vpint
    elif Vpint.size == nnodes:
        SlowP = 1. / Vpint
    else:
        print("invalid P Velocity model\n")
        sys.stdout.flush()
        return 0
    if Vsint.size == 1:
        SlowS = np.ones([nnodes, 1]) / Vsint
    elif Vsint.size == nnodes:
        SlowS = 1. / Vsint
    else:
        print("invalid S Velocity model\n")
        sys.stdout.flush()
        return 0
    if par.invert_VpVs:
        SlsSlp = SlowS / SlowP
        Slowness = np.vstack((SlowP, SlsSlp))
    else:
        Slowness = np.vstack((SlowP, SlowS))
    # Hypocenter
    if numberOfEvents > 0 and Hypo0.shape[0] != numberOfEvents:
        print("invalid Hypocenters0 format\n")
        sys.stdout.flush()
        return 0
    else:
        Hypocenters = Hypo0.copy()
    # number of threads
    nThreadsSystem = cpu_count()
    nThreads = np.min((threads, nThreadsSystem))

    global Mesh3D, Dimensions
    Mesh3D = tmesh.Mesh3d(nodes, tetra=cells, method='DSPM', cell_slowness=0,
                          n_threads=nThreads, n_secondary=2, n_tertiary=1,
                          radius_factor_tertiary=2, translate_grid=1)
    Dimensions = np.empty(6)
    Dimensions[0] = min(nodes[:, 0])
    Dimensions[1] = max(nodes[:, 0])
    Dimensions[2] = min(nodes[:, 1])
    Dimensions[3] = max(nodes[:, 1])
    Dimensions[4] = min(nodes[:, 2])
    Dimensions[5] = max(nodes[:, 2])
    if par.invert_vel:
        if par.use_sc:
            U = sp.hstack((sp.csr_matrix(
                 np.vstack((np.zeros([2 * nnodes, 1]),
                            np.ones([nstation, 1]),
                            np.zeros([nstation, 1])))),
                           sp.csr_matrix(np.vstack(
                                (np.zeros([2 * nnodes + nstation, 1]),
                                 np.ones([nstation, 1]))))))
            nbre_param = 2 * (nnodes + nstation)
            if par.max_sc > 0. and par.max_sc < 1.:
                N = sp.bsr_matrix(np.hstack(
                    (np.zeros([2 * nstation, 2 * nnodes]), np.eye(2 * nstation))))
                NtN = (1. / par.max_sc**2) * N.T.dot(N)
        else:
            U = sp.csr_matrix(np.zeros([2 * nnodes, 2]))
            nbre_param = 2 * nnodes
        # calculate statistical moments of VpVs ratio
        if par.stig != 0.:
            momnts = np.zeros([4, ])
            VpVs = caldataS[:, 1] / caldataP[:, 1]
            for m in np.arange(4):
                if m == 0:
                    momnts[m] = VpVs.mean() * nnodes
                else:
                    momnts[m] = scps.moment(VpVs, m + 1) * nnodes
        # build matrix D
        vPoints_p, vPoints_s = vPnts
        if vPoints_p.size > 0 or vPoints_s.size > 0:
            if par.invert_VpVs:
                for i in np.arange(vPoints_s.shape[0]):
                    dist = np.sqrt(
                        np.sum((vPoints_p[:, 2:] - vPoints_s[i, 2:])**2, axis=1))
                    indp = np.where(dist < 1.e-5)[0]
                    if indp.size > 0:
                        vPoints_s[i, 1] /= vPoints_p[indp, 1]  # Vs/Vp
                    else:
                        raise ValueError('Missing Vp data point for Vs data at '
                                         '({0:f}, {1:f}, {2:f})'.format(
                                              vPoints_s[i, 2], vPoints_s[i, 3],
                                              vPoints_s[i, 4]))
                        sys.stdout.flush()
                vPoints = np.vstack((vPoints_p, vPoints_s))
                if par.verbose:
                    print('Building velocity data point matrix D\n')
                    sys.stdout.flush()
                Ds = Mesh3D.compute_D(vPoints_s[:, 2:])
                D = sp.block_diag((Ds, Ds)).tocsr()
                D = sp.hstack((D, sp.csr_matrix((D.shape[0], 2*nstation)))).tocsr()
                DtD = D.T @ D
                nD = spl.norm(DtD)
            else:
                vPoints = np.vstack((vPoints_p, vPoints_s))
                Dp = Mesh3D.compute_D(vPoints_p[:, 2:])
                Ds = Mesh3D.compute_D(vPoints_s[:, 2:])
                D = sp.block_diag((Dp, Ds)).tocsr()
                D = sp.hstack((D, sp.csr_matrix((D.shape[0], 2*nstation)))).tocsr()
                DtD = D.T @ D
                nD = spl.norm(DtD)
            vPoints[:, 1] = 1. / vPoints[:, 1]
        else:
            vPoints = np.array([])
        # Build regularization matrix
        if par.verbose:
            print('\n...Building regularization matrix K\n')
            sys.stdout.flush()
        kx, ky, kz = Mesh3D.compute_K(order=2, taylor_order=2,
                                      weighting=1, squared=0,
                                      s0inside=0, additional_points=3)
        kx = sp.block_diag((kx, kx))
        ky = sp.block_diag((ky, ky))
        kz = sp.block_diag((kz, kz))
        KX = sp.hstack((kx, sp.csr_matrix((2 * nnodes, 2 * nstation))))
        KX_Square = KX.transpose() @ KX
        KY = sp.hstack((ky, sp.csr_matrix((2 * nnodes, 2 * nstation))))
        KY_Square = KY.transpose() @ KY
        KZ = sp.hstack((kz, sp.csr_matrix((2 * nnodes, 2 * nstation))))
        KZ_Square = KZ.transpose() @ KZ
        KtK = KX_Square + KY_Square + par.wzK * KZ_Square
        nK = spl.norm(KtK)
    if nThreads == 1:
        hypo_convergence = list(np.zeros(numberOfEvents, dtype=bool))
    else:
        manager = Manager()
        hypo_convergence = manager.list(np.zeros(numberOfEvents, dtype=bool))
    ResidueNorm = np.zeros([par.maxit])
    for i in range(par.maxit):
        if par.verbose:
            print("Iteration N : {0:d}\n".format(i + 1))
            sys.stdout.flush()
        if par.invert_vel:
            if par.verbose:
                print(
                    'Iteration {0:d} - Updating velocity model\n'.format(i + 1))
                print("Updating penalty vector\n")
                sys.stdout.flush()
            # Build vector C
            cx = kx.dot(Slowness)
            cy = ky.dot(Slowness)
            cz = kz.dot(Slowness)
            # build matrix P and dP
            indSpmin = np.where(Slowness[:nnodes] < 1. / par.Vpmax)[0]
            indSpmax = np.where(Slowness[:nnodes] > 1. / par.Vpmin)[0]
            if par.invert_VpVs:
                indVpVsmin = np.where(Slowness[nnodes:] < par.VpVsmin)[0] + nnodes
                indVpVsmax = np.where(Slowness[nnodes:] > par.VpVsmax)[0] + nnodes

                pinality_V = np.vstack([par.PAp *
                                        (1. /
                                         par.Vpmax -
                                         Slowness[indSpmin]), par.PAp *
                                        (Slowness[indSpmax] -
                                         1. /
                                         par.Vpmin), par.Pvpvs *
                                        (par.VpVsmin -
                                            Slowness[indVpVsmin]), par.Pvpvs *
                                        (Slowness[indVpVsmax] -
                                            par.VpVsmax)])

                indPinality = np.hstack(
                    [indSpmin, indSpmax, indVpVsmin, indVpVsmax])
                dPinality_V = np.hstack([-par.PAp * np.ones(indSpmin.size), par.PAp
                                         * np.ones(indSpmax.size), - par.Pvpvs *
                                         np.ones(indVpVsmin.size), par.Pvpvs *
                                         np.ones(indVpVsmax.size)])

            else:
                indSsmin = np.where(
                    Slowness[nnodes:] < 1. / par.Vsmax)[0] + nnodes
                indSsmax = np.where(
                    Slowness[nnodes:] > 1. / par.Vsmin)[0] + nnodes

                pinality_V = np.vstack([par.PAp *
                                        (1. / par.Vpmax -
                                         Slowness[indSpmin]), par.PAp *
                                        (Slowness[indSpmax] -
                                         1. / par.Vpmin), par.PAs *
                                        (1. / par.Vsmax -
                                            Slowness[indSsmin]), par.PAs *
                                        (Slowness[indSsmax] -
                                            1. / par.Vsmin)])

                indPinality = np.hstack(
                    [indSpmin, indSpmax, indSsmin, indSsmax])
                dPinality_V = np.hstack([-par.PAp * np.ones(indSpmin.size),
                                         par.PAp * np.ones(indSpmax.size),
                                         -par.PAs * np.ones(indSsmin.size),
                                         par.PAs * np.ones(indSsmax.size)])
                if par.VpVsmin and par.VpVsmax:
                    indvpvs_min = np.where(SlowS / SlowP <= par.VpVsmin)[0]
                    indvpvs_max = np.where(SlowS / SlowP >= par.VpVsmax)[0]
                    if par.verbose and indvpvs_max.size > 0:
                        print('\n{0:d} nodes have Vp/Vs ratio higher than the upper'
                              ' VpVs limit\n'.format(indvpvs_max.size))
                        sys.stdout.flush()
                    if par.verbose and indvpvs_min.size > 0:
                        print('\n{0:d} nodes have Vp/Vs ratio lower than the lower '
                              'VpVs limit\n'.format(indvpvs_min.size))
                        sys.stdout.flush()
                    indPnltvpvs = np.hstack([indvpvs_min, indvpvs_max])
                    no = 2  # order or pinality function
                    pinlt_vpvs = np.vstack([par.Pvpvs * (par.VpVsmin -
                                                         SlowS[indvpvs_min] /
                                                         SlowP[indvpvs_min])**no,
                                            par.Pvpvs * (SlowS[indvpvs_max] /
                                                         SlowP[indvpvs_max] -
                                                         par.VpVsmax)**no])

                    PinltVpVs = sp.csr_matrix((pinlt_vpvs.reshape([-1, ]),
                                               (indPnltvpvs, np.zeros(
                                                    [indPnltvpvs.shape[0]]))),
                                              shape=(nnodes, 1))
                    dPinltVpVsind = (np.hstack([indvpvs_min, indvpvs_max,
                                                indvpvs_min, indvpvs_max]),
                                     np.hstack([indvpvs_min, indvpvs_max,
                                                indvpvs_min + nnodes,
                                                indvpvs_max + nnodes]))

                    dPinltVpVs_V = np.vstack((par.Pvpvs * SlowS[indvpvs_min] /
                                              (SlowP[indvpvs_min]**2) * no *
                                              (par.VpVsmin - SlowS[indvpvs_min] /
                                              SlowP[indvpvs_min])**(no - 1), -
                                              par.Pvpvs * SlowS[indvpvs_max] /
                                              (SlowP[indvpvs_max]**2) *
                                              no * (SlowS[indvpvs_max] /
                                              SlowP[indvpvs_max] -
                                              par.VpVsmax)**(no - 1), -
                                              par.Pvpvs / SlowP[indvpvs_min] *
                                              no * (par.VpVsmin -
                                                    SlowS[indvpvs_min] /
                                             SlowP[indvpvs_min])**(no - 1),
                                             par.Pvpvs / SlowP[indvpvs_max] *
                                             no * (SlowS[indvpvs_max] /
                                                   SlowP[indvpvs_max] -
                                                   par.VpVsmax) **
                                             (no - 1)))

                    dPinltVpVs = sp.csr_matrix((dPinltVpVs_V.reshape(
                        [-1, ]), dPinltVpVsind), shape=(nnodes, nbre_param))

            d_Pinality = sp.csr_matrix(
                (dPinality_V, (indPinality, indPinality)), shape=(
                    2 * nnodes, nbre_param))
            Pinality = sp.csr_matrix((pinality_V.reshape([-1, ]),
                                      (indPinality, np.zeros([
                                           indPinality.shape[0]]))),
                                     shape=(2 * nnodes, 1))
            if par.verbose:
                print('P wave Penalties were applied at {0:d} nodes\n'.format(
                        indSpmin.shape[0] +
                        indSpmax.shape[0]))
                sys.stdout.flush()
                if par.invert_VpVs:
                    print('Vp/Vs ratio Penalties were applied '
                          'at {0:d} nodes\n'.format(indVpVsmin.shape[0] +
                                                    indVpVsmax.shape[0]))
                    sys.stdout.flush()
                else:
                    print('S wave Penalties were applied at {0:d} nodes\n'.format(
                            indSsmin.shape[0] + indSsmax.shape[0]))
                    sys.stdout.flush()
                print('...Start Raytracing\n')
                sys.stdout.flush()
            if numberOfEvents > 0:
                sourcesp = np.empty((dataP.shape[0], 5))
                if par.use_sc:
                    scp_data = np.zeros((dataP.shape[0], 1))
                for ev in np.arange(numberOfEvents):
                    indrp = np.where(dataP[:, 0] == evID[ev])[0]
                    indh = np.where(Hypocenters[:, 0] == evID[ev])[0]
                    sourcesp[indrp, :] = Hypocenters[indh, :]
                    if par.use_sc:
                        scp_data[indrp, :] = scP[dataP[indrp, 2].astype(int) - 1]
                if par.use_sc:
                    sourcesp[:, 1] += scp_data[:, 0]
                    ttp, raysp, M0p = Mesh3D.raytrace(
                         source=sourcesp, rcv=rcvData_P,
                         slowness=SlowP, aggregate_src=False,
                         compute_L=True, return_rays=True)
                else:
                    ttp, raysp, M0p = Mesh3D.raytrace(
                         source=sourcesp, rcv=rcvData_P,
                         slowness=SlowP, aggregate_src=False,
                         compute_L=True, return_rays=True)
                if par.verbose:
                    inconverged = np.where(ttp == 0)[0]
                    for icr in inconverged:
                        print('\033[43m' +
                              '\nWarning: raypath failed to converge for even '
                              'N {0:d} :({1:5.4f},{2:5.4f},{3:5.4f}) and receiver N'
                              ' {4:d} :({5:5.4f},{6:5.4f},{7:5.4f})\n'.format(
                                   int(dataP[icr, 0]), sourcesp[icr, 2],
                                   sourcesp[icr, 3], sourcesp[icr, 4],
                                   int(dataP[icr, 2]), rcvData_P[icr, 0],
                                   rcvData_P[icr, 1], rcvData_P[icr, 2]) +
                              '\033[0m')
                        print('\033[43m' + 'ray will be temporary removed' +
                              '\033[0m')
                        sys.stdout.flush()
                slowP_0 = Mesh3D.get_s0(sourcesp)

                sourcesS = np.empty((dataS.shape[0], 5))
                if par.use_sc:
                    scs_data = np.zeros((dataS.shape[0], 1))
                for ev in np.arange(numberOfEvents):
                    indrs = np.where(dataS[:, 0] == evID[ev])[0]
                    indh = np.where(Hypocenters[:, 0] == evID[ev])[0]
                    sourcesS[indrs, :] = Hypocenters[indh, :]
                    if par.use_sc:
                        scs_data[indrs, :] = scS[dataS[indrs, 2].astype(int) - 1]
                if par.use_sc:
                    sourcesS[:, 1] += scs_data[:, 0]
                    tts, rayss, M0s = Mesh3D.raytrace(
                         source=sourcesS, rcv=rcvData_S,
                         slowness=SlowS, aggregate_src=False,
                         compute_L=True, return_rays=True)
                else:
                    tts, rayss, M0s = Mesh3D.raytrace(
                         source=sourcesS, rcv=rcvData_S,
                         slowness=SlowS, aggregate_src=False,
                         compute_L=True, return_rays=True)
                if par.verbose:
                    inconverged = np.where(tts == 0)[0]
                    for icr in inconverged:
                        print('\033[43m' +
                              '\nWarning: raypath failed to converge for even '
                              'N {0:d} :({1:5.4f},{2:5.4f},{3:5.4f}) and receiver'
                              ' {4:d} :({5:5.4f},{6:5.4f},{7:5.4f})\n'.format(
                                   int(dataS[icr, 0]), sourcesS[icr, 2],
                                   sourcesS[icr, 3], sourcesS[icr, 4],
                                   int(dataS[icr, 2]), rcvData_S[icr, 0],
                                   rcvData_S[icr, 1], rcvData_S[icr, 2]) +
                              '\033[0m')
                        print('\033[43m' + 'ray will be temporary removed' +
                              '\033[0m')
                        sys.stdout.flush()
                slowS_0 = Mesh3D.get_s0(sourcesS)
                tt = np.hstack((ttp, tts))
                slow_0 = np.hstack((slowP_0, slowS_0))
                rays = raysp + rayss
            else:
                tt = np.array([])
            if nttcalp > 0:
                if par.use_sc:
                    scp_cal = scP[caldataP[:, 2].astype(int) - 1, 0]
                    TxCalibP[:, 1] = scp_cal
                    tt_Calibp, Mcalibp = Mesh3D.raytrace(
                         source=TxCalibP, rcv=rcvCalibP,
                         slowness=SlowP, aggregate_src=False,
                         compute_L=True, return_rays=False)
                else:
                    tt_Calibp, Mcalibp = Mesh3D.raytrace(
                         source=TxCalibP, rcv=rcvCalibP,
                         slowness=SlowP, aggregate_src=False,
                         compute_L=True, return_rays=False)
                if par.verbose:
                    inconverged = np.where(tt_Calibp == 0)[0]
                    for icr in inconverged:
                        print('\033[43m' + '\nWarning: raypath failed to converge '
                              'for calibration shot N {0:d} :({1:5.4f},{2:5.4f},'
                              '{3:5.4f}) and receiver N {4:d} :({5:5.4f},{6:5.4f},'
                              '{7:5.4f})\n'.format(
                                   int(caldataP[icr, 0]), TxCalibP[icr, 2],
                                   TxCalibP[icr, 3], TxCalibP[icr, 4],
                                   int(caldataP[icr, 2]),
                                   rcvCalibP[icr, 0],
                                   rcvCalibP[icr, 1],
                                   rcvCalibP[icr, 2]) + '\033[0m')
                        print('\033[43m' + 'ray will be temporary removed' +
                              '\033[0m')
                        sys.stdout.flush()
            else:
                tt_Calibp = np.array([])
            if nttcals > 0:
                if par.use_sc:
                    scs_cal = scS[caldataS[:, 2].astype(int) - 1, 0]
                    TxCalibS[:, 1] = scs_cal
                    tt_Calibs, Mcalibs = Mesh3D.raytrace(
                         source=TxCalibS, rcv=rcvCalibS,
                         slowness=SlowS, aggregate_src=False,
                         compute_L=True, return_rays=False)
                else:
                    tt_Calibs, Mcalibs = Mesh3D.raytrace(
                         source=TxCalibS, rcv=rcvCalibS,
                         slowness=SlowS, aggregate_src=False,
                         compute_L=True, return_rays=False)
                if par.verbose:
                    inconverged = np.where(tt_Calibs == 0)[0]
                    for icr in inconverged:
                        print('\033[43m' +
                              '\nWarning: raypath failed to converge '
                              'for calibration shot N {0:d} :({1:5.4f},{2:5.4f},'
                              '{3:5.4f}) and receiver N {4:d} :({5:5.4f},{6:5.4f},'
                              '{7:5.4f})\n'.format(
                                   int(caldataS[icr, 0]), TxCalibS[icr, 2],
                                   TxCalibS[icr, 3], TxCalibS[icr, 4],
                                   int(caldataS[icr, 2]), rcvCalibS[icr, 0],
                                   rcvCalibS[icr, 1], rcvCalibS[icr, 2])
                              + '\033[0m')
                        print('\033[43m' + 'ray will be temporary removed' +
                              '\033[0m')
                        sys.stdout.flush()
            else:
                tt_Calibs = np.array([])
            tt_Calib = np.hstack((tt_Calibp, tt_Calibs))
            Resid = tObserved - tt
            convrayData = np.where(tt != 0)[0]
            convrayClib = np.where(tt_Calib != 0)[0]
            if Resid.size == 0:
                Residue = time_calibration[convrayClib] - tt_Calib[convrayClib]
            else:
                Residue = np.hstack((np.zeros([np.count_nonzero(
                    tt) - 4 * numberOfEvents]), time_calibration[convrayClib] -
                     tt_Calib[convrayClib]))
            ResidueNorm[i] = np.linalg.norm(np.hstack(
                (Resid[convrayData], time_calibration[convrayClib] -
                 tt_Calib[convrayClib])))
            if par.verbose:
                print('...Building matrix M\n')
                sys.stdout.flush()
            M = sp.csr_matrix((0, nbre_param))
            ir = 0
            for even in range(numberOfEvents):
                Mpi = M0p[even]
                nst_ev = Mpi.shape[0]
                if par.use_sc:
                    indrp = np.where(dataP[:, 0] == evID[even])[0]
                    convrays = np.where(ttp[indrp] != 0)[0]
                    Lscp = sp.csr_matrix(
                         (np.ones(nst_ev,),
                          (range(nst_ev), dataP[indrp[convrays], 2] - 1)),
                         shape=(nst_ev, nstation))
                else:
                    Lscp = sp.csr_matrix((nst_ev, 0))
                Mpi = sp.hstack((Mpi, sp.csr_matrix((nst_ev, nnodes)),
                                 Lscp, sp.csr_matrix((nst_ev, nstation)))).tocsr()

                Msi = M0s[even]
                nst_ev = Msi.shape[0]
                if par.use_sc:
                    indrs = np.where(dataS[:, 0] == evID[even])[0]
                    convrays = np.where(tts[indrs] != 0)[0]
                    Lscs = sp.csr_matrix(
                         (np.ones(nst_ev,),
                          (range(nst_ev), dataS[indrs[convrays], 2] - 1)),
                         shape=(nst_ev, nstation))
                else:
                    Lscs = sp.csr_matrix((nst_ev, 0))
                if par.invert_VpVs:
                    dTsdSp = Msi.multiply(np.tile(SlsSlp, nst_ev).T)
                    dTsdVpVs = Msi.multiply(np.tile(SlowP, nst_ev).T)
                    Msi = sp.hstack((dTsdSp, dTsdVpVs, sp.csr_matrix(
                        (nst_ev, nstation)), Lscs)).tocsr()
                else:
                    Msi = sp.hstack((sp.csr_matrix((nst_ev, nnodes)), Msi,
                                     sp.csr_matrix((nst_ev, nstation)),
                                     Lscs)).tocsr()

                Mi = sp.vstack((Mpi, Msi))

                indh = np.where(Hypocenters[:, 0] == evID[even])[0]
                indr = np.where(data[:, 0] == evID[even])[0]
                nst_ev = indr.size
                Hi = np.ones([nst_ev, 4])
                for nr in range(nst_ev):
                    rayi = rays[indr[nr]]
                    if rayi.shape[0] == 1:
                        continue
                    slw0 = slow_0[indr[nr]]
                    dx = rayi[1, 0] - Hypocenters[indh[0], 2]
                    dy = rayi[1, 1] - Hypocenters[indh[0], 3]
                    dz = rayi[1, 2] - Hypocenters[indh[0], 4]
                    ds = np.sqrt(dx * dx + dy * dy + dz * dz)
                    Hi[nr, 1] = -slw0 * dx / ds
                    Hi[nr, 2] = -slw0 * dy / ds
                    Hi[nr, 3] = -slw0 * dz / ds
                convrays = np.where(tt[indr] != 0)[0]
                if convrays.shape[0] < nst_ev:
                    Hi = Hi[convrays, :]
                    nst_ev = convrays.size
                if Hi.shape[0] < 4:
                    print('\n Warning : even contains less than 4 rays')
                    sys.stdout.flush()
                Q, _ = np.linalg.qr(Hi, mode='complete')
                Ti = sp.csr_matrix(Q[:, 4:])
                Ti = Ti.T
                Mi = sp.csr_matrix(Ti @ Mi)
                M = sp.vstack([M, Mi])
                Residue[ir:ir + (nst_ev - 4)] = Ti.dot(Resid[indr[convrays]])
                ir += nst_ev - 4
            for evCalp in range(len(Mcalibp)):
                Mpi = Mcalibp[evCalp]
                nst_evcal = Mpi.shape[0]
                if par.use_sc:
                    indrCalp = np.where(caldataP[:, 0] == calID[evCalp])[0]
                    convraysp = np.where(tt_Calibp[indrCalp] != 0)[0]
                    Mpi = sp.hstack((
                         Mpi, sp.csr_matrix((nst_evcal, nnodes)),
                         Mscp_cal[evCalp][convraysp],
                         sp.csr_matrix((nst_evcal,
                                        nstation)))).tocsr()
                else:
                    Mpi = sp.hstack(
                        (Mpi, sp.csr_matrix((nst_evcal, nnodes)))).tocsr()
                M = sp.vstack([M, Mpi])
            for evCals in range(len(Mcalibs)):
                Msi = Mcalibs[evCals]
                nst_evcal = Msi.shape[0]
                if par.invert_VpVs:
                    dTsdSp = Msi.multiply(np.tile(SlsSlp, nst_evcal).T)
                    dTsdVpVs = Msi.multiply(np.tile(SlowP, nst_evcal).T)
                    if par.use_sc:
                        indrCals = np.where(caldataS[:, 0] == calID[evCals])[0]
                        convrayss = np.where(tt_Calibs[indrCals] != 0)[0]
                        Msi = sp.hstack((dTsdSp, dTsdVpVs,
                                         sp.csr_matrix((nst_evcal, nstation)),
                                         Mscs_cal[evCals][convrayss])).tocsr()
                    else:
                        Msi = sp.hstack((dTsdSp, dTsdVpVs)).tocsr()
                else:
                    if par.use_sc:
                        indrCals = np.where(caldataS[:, 0] == calID[evCals])[0]
                        convrayss = np.where(tt_Calibs[indrCals] != 0)[0]
                        Msi = sp.hstack((sp.csr_matrix((nst_evcal, nnodes)),
                                         Msi, sp.csr_matrix((nst_evcal, nstation)),
                                         Mscs_cal[evCals][convrayss])).tocsr()
                    else:
                        Msi = sp.hstack((sp.csr_matrix((nst_evcal, nnodes)),
                                         Msi)).tocsr()
                M = sp.vstack((M, Msi))
            if par.stig != 0.:
                dPr = sp.lil_matrix((4, nbre_param))
                Pr = np.zeros([4, 1])
                for m in np.arange(4):
                    if par.invert_VpVs:
                        Gamma_mean = SlsSlp.mean()
                        dPr[m, nnodes:2 * nnodes] = ((m + 1) *
                                                     (SlsSlp -
                                                      Gamma_mean) **
                                                     m).reshape([-1, ])
                        if m == 0:
                            Pr[m, 0] = momnts[m] - np.sum(SlsSlp)
                        else:
                            Pr[m, 0] = momnts[m] - np.sum((SlsSlp -
                                                           Gamma_mean)**(m + 1))
                    else:
                        Gamma_mean = (SlowS / SlowP).mean()
                        dPr[m, :nnodes] = (-(m + 1) *
                                           (SlowS / SlowP -
                                            Gamma_mean) ** m *
                                           SlowS / (SlowP**2)).reshape([-1, ])
                        dPr[m, nnodes:2 * nnodes] = ((m + 1) *
                                                     (SlowS / SlowP - Gamma_mean)**m
                                                     * (1. / SlowP)).reshape([-1, ])
                        if m == 0:
                            Pr[m, 0] = momnts[m] - np.sum(SlowS / SlowP)

                        else:
                            Pr[m, 0] = momnts[m] - np.sum((SlowS / SlowP -
                                                           Gamma_mean)**(m + 1))

                dPr2 = dPr.T @ dPr
            if par.verbose:
                print('Assembling matrices and solving system\n')
                sys.stdout.flush()
            S = np.array([np.sum(scP), np.sum(scS)]).reshape([-1, 1])
            term1 = (M.T).dot(M)
            nM = spl.norm(term1[:2 * nnodes, :2 * nnodes])
            λ = par.λ * nM / nK
            term2 = (d_Pinality.T).dot(d_Pinality)
            nP = spl.norm(term2)
            if nP != 0:
                γ = par.γ * nM / nP
            else:
                γ = par.γ
            term3 = U.dot(U.T)
            A = term1 + λ * KtK + γ * term2 + term3
            if par.use_sc and par.max_sc > 0. and par.max_sc < 1.:
                A += NtN
            term1 = (M.T).dot(Residue.reshape([-1, 1]))
            term2 = KX.T @ cx + KY.T @ cy + par.wzK * KZ.T @ cz
            term3 = d_Pinality.T @ Pinality
            term4 = U.dot(S)
            b = term1 - λ * term2 - γ * term3 - term4
            if par.stig != 0.:
                if spl.norm(dPr2) != 0:
                    stg = par.stig * nM / spl.norm(dPr2)
                    A += stg * dPr2
                    b += stg * dPr.T @ Pr
            if vPoints.size > 0:
                α = par.α * nM / nD
                A += α * DtD
                b += α * D.T @ (vPoints[:, 1].reshape(-1, 1) -
                                D[:, :2 * nnodes] @ Slowness)
            if not par.invert_VpVs and par.VpVsmin and par.VpVsmax:
                dPinltVpVs2 = dPinltVpVs.T @ dPinltVpVs
                if spl.norm(dPinltVpVs2) != 0:
                    γvpvs = par.γ_vpvs * nM / spl.norm(dPinltVpVs2)
                    A += γvpvs * dPinltVpVs2
                    b -= γvpvs * dPinltVpVs.T @ PinltVpVs

            x = spl.minres(A, b, tol=1.e-8)
            deltam = x[0].reshape(-1, 1)
            # update velocity vector and static correction
            deltaVp_max = np.max(
                abs(1. / (SlowP + deltam[:nnodes]) - 1. / SlowP))
            if deltaVp_max > par.dVp_max:
                print('\n...Rescale P slowness\n')
                sys.stdout.flush()
                L1 = np.max(deltam[:nnodes] / (-par.dVp_max * (SlowP**2) /
                                               (1 + par.dVp_max * SlowP)))
                L2 = np.max(deltam[:nnodes] / (par.dVp_max * (SlowP**2) /
                                               (1 - par.dVp_max * SlowP)))
                deltam[:nnodes] /= np.max([L1, L2])
                print('P wave: maximum ds= {0:4.3f}, '
                      'maximum dV = {1:4.3f}\n'.format(
                           max(abs(deltam[:nnodes]))[0],
                           np.max(abs(1. / (SlowP + deltam[:nnodes]) -
                                      1. / SlowP))))
                sys.stdout.flush()
            if par.invert_VpVs:
                deltaVsVp_max = np.max(abs((1. / ((deltam[:nnodes] + SlowP) *
                                                  (Slowness[nnodes:2 * nnodes] +
                                                   deltam[nnodes:2 * nnodes])) -
                                            1. / SlowS)))
                if deltaVsVp_max > par.dVs_max:
                    print('\n...Rescale VpVs\n')
                    sys.stdout.flush()
                    L1 = np.max((
                         deltam[nnodes:2 * nnodes] /
                         (1. / ((-par.dVs_max + 1. / SlowS) *
                                (deltam[:nnodes] + SlowP)) -
                          Slowness[nnodes:2 * nnodes])))
                    L2 = np.max((deltam[nnodes:2 * nnodes] /
                                 (1. / ((par.dVs_max + 1. / SlowS) *
                                        (deltam[:nnodes] + SlowP)) -
                                  Slowness[nnodes:2 * nnodes])))
                    deltam[nnodes:2 * nnodes] /= np.max([L1, L2])
            else:
                deltaVs_max = np.max(
                    abs(1. / (SlowS + deltam[nnodes:2 * nnodes]) - 1. / SlowS))
                if deltaVs_max > par.dVs_max:
                    print('\n...Rescale S slowness\n')
                    sys.stdout.flush()
                    L1 = np.max(deltam[nnodes:2 * nnodes] /
                                (-par.dVs_max * (SlowS**2) /
                                 (1 + par.dVs_max * SlowS)))
                    L2 = np.max(deltam[nnodes:2 * nnodes] /
                                (par.dVs_max * (SlowS**2) /
                                 (1 - par.dVs_max * SlowS)))
                    deltam[nnodes:2 * nnodes] /= np.max([L1, L2])
                    print('S wave: maximum ds= {0:4.3f}, maximum '
                          'dV = {1:4.3f}\n'.format(max(abs(
                               deltam[nnodes:2 * nnodes]))[0],
                               np.max(abs(1. / (SlowS + deltam[nnodes:2 * nnodes])
                                          - 1. / SlowS))))
                    sys.stdout.flush()
            if par.use_sc and par.max_sc > 0. and par.max_sc < 1.:
                sc_mean = np.mean(abs(deltam[2 * nnodes:]))
                if sc_mean > par.max_sc * np.mean(abs(Residue)):
                    deltam[2 * nnodes:] *= par.max_sc * \
                        np.mean(abs(Residue)) / sc_mean
            Slowness += np.matrix(deltam[:2 * nnodes])
            SlowP = Slowness[:nnodes]
            if par.invert_VpVs:
                SlsSlp = Slowness[nnodes:2 * nnodes]
                SlowS = SlsSlp * SlowP
            else:
                SlowS = Slowness[nnodes:2 * nnodes]
            scP += deltam[2 * nnodes:2 * nnodes + nstation]
            scS += deltam[2 * nnodes + nstation:]
            if par.saveVel == 'all':
                if par.verbose:
                    print('...Saving Velocity model of interation '
                          'N: {0:d}\n'.format(i + 1))
                    sys.stdout.flush()
                try:
                    msh2vtk(nodes, cells, 1. / SlowP, basename +
                            '_Vp_it{0}.vtk'.format(i + 1))
                except ImportError:
                    print('cannot save P wave velocity model in format vtk\n')
                    sys.stdout.flush()
                try:
                    msh2vtk(nodes, cells, 1. / SlowS, basename +
                            '_Vs_it{0}.vtk'.format(i + 1))
                except ImportError:
                    print('cannot save S wave velocity model in format vtk\n')
                    sys.stdout.flush()
                if par.invert_VpVs:
                    try:
                        msh2vtk(nodes, cells, SlsSlp, basename +
                                '_VsVp Ratio_it{0}.vtk'.format(i + 1))
                    except ImportError:
                        print('cannot save Vs/Vp ration model in format vtk\n')
                        sys.stdout.flush()
            elif par.saveVel == 'last' and i == par.maxit - 1:
                if par.verbose:
                    print('...Saving Velocity models of the last iteration\n')
                    sys.stdout.flush()
                try:
                    msh2vtk(nodes, cells, 1. / SlowP, basename +
                            '_Vp_final.vtk')
                except ImportError:
                    print('cannot save the final P wave velocity '
                          'model in format vtk\n')
                    sys.stdout.flush()
                try:
                    msh2vtk(nodes, cells, 1. / SlowS,
                            basename + '_Vs_final.vtk')
                except ImportError:
                    print('cannot save the final S wave velocity model '
                          'in format vtk\n')
                    sys.stdout.flush()
                if par.invert_VpVs:
                    try:
                        msh2vtk(nodes, cells, SlsSlp, basename +
                                '_VsVp Ratio_final.vtk')
                    except ImportError:
                        print(
                            'cannot save the final Vs/Vp ratio model in '
                            'format vtk\n')
                        sys.stdout.flush()
                #######################################
                        # relocate Hypocenters
                #######################################

        if numberOfEvents > 0:
            print("\nIteration N {0:d} : Relocation of events".format(
                    i + 1) + '\n')
            sys.stdout.flush()
            if nThreads == 1:
                for ev in range(numberOfEvents):
                    Hypocenters[ev, :] = _hypo_relocationPS(
                         ev, evID, Hypocenters, (dataP, dataS), rcv, (scP, scS),
                         hypo_convergence, (SlowP, SlowS), par)
            else:
                with Pool(processes=nThreads) as p:
                    updatedHypo = p.starmap(
                         _hypo_relocationPS, [(int(ev),
                                               evID, Hypocenters,
                                               (dataP, dataS), rcv,
                                               (scP, scS), hypo_convergence,
                                               (SlowP, SlowS), par) for ev
                                              in range(numberOfEvents)])
                    p.close()  # pool won't take any new tasks
                    p.join()
                Hypocenters = np.array([updatedHypo])[0]

    #  Calculate the hypocenter parameter uncertainty
    uncertnty = []
    if par.uncertainty:
        print("Uncertainty evaluation" + '\n')
        sys.stdout.flush()
        if nThreads == 1:
            varData = [[], []]
            for ev in range(numberOfEvents):
                uncertnty.append(
                     _uncertaintyEstimat(ev, evID, Hypocenters,
                                         (dataP, dataS), rcv, (scP, scS),
                                         (SlowP, SlowS), par, varData))
        else:
            varData = manager.list([[], []])
            with Pool(processes=nThreads) as p:
                uncertnty = p.starmap(
                     _uncertaintyEstimat, [(int(ev), evID,
                                            Hypocenters, (dataP, dataS), rcv,
                                            (scP, scS), (SlowP, SlowS), par,
                                            varData)
                                           for ev in range(numberOfEvents)])
                p.close()
                p.join()
        sgmData = np.sqrt(np.sum(varData[0]) /
                          (np.sum(varData[1]) - 4 *
                           numberOfEvents - scP.size - scS.size))
        for ic in range(numberOfEvents):
            uncertnty[ic] = tuple([sgmData * x for x in uncertnty[ic]])
    output = OrderedDict()
    output['Hypocenters'] = Hypocenters
    output['Convergence'] = list(hypo_convergence)
    output['Uncertainties'] = uncertnty
    output['P_velocity'] = 1. / SlowP
    output['S_velocity'] = 1. / SlowS
    output['P_StsCorrections'] = scP
    output['S_StsCorrections'] = scS
    output['Residual_norm'] = ResidueNorm

    return output


def jointHypoVel_T(inputFileParam, model='slow'):
    """
     Joint hypocenter-velocity inversion using P wave data.

     Parameters
     ----------
     inputFileParam : string
          Text file containing inversion parameters and data filenames.
     model : string
          Sought model : 'vel' for an inversion problem parametrized using
          the velocity model,'slow' for an inversion problem parametrized using
          the slowness model. The default is 'slow'.

     Returns
     -------
     python dictionary
          It contains the estimated hypocenter coordinates and their origin times,
          static correction values, velocity model, convergence states,
          parameter uncertainty and residual norm in each iteration.

     """
    ParametersFile = fileReader(inputFileParam)
    par = ParametersFile.getIversionParam()
    basename = ParametersFile.readParameter('base name')
    # read mesh parameters
    MESH = MSHReader(ParametersFile.readParameter('mesh file'))
    nodes = MESH.readNodes()
    cells = MESH.readTetraherdonElements()
    #  red rcv coordinates
    stations = RCVReader(ParametersFile.readParameter('rcvfile	'))
    rcv = stations.getStation()
    # read observed traveltimes
    if ParametersFile.readParameter('arrival times'):
        data = readEventsFiles(ParametersFile.readParameter('arrival times'))
    else:
        data = np.array([])
    # get calibration data
    if ParametersFile.readParameter('Time calibration'):
        caldata = readEventsFiles(
            ParametersFile.readParameter('Time calibration'))
    else:
        caldata = np.array([])
    # get initial velocity model
    Vint = np.loadtxt(ParametersFile.readParameter('Velocity P waves'), ndmin=2)
    # get initial parameters Hyocenters0 and origin times
    Hypo0 = readEventsFiles(ParametersFile.readParameter('Hypo0'))
    # get and set number of threads
    NThreadsUser = ParametersFile.readParameter('number of threads', int)
    # known v points
    vptsfile = ParametersFile.readParameter('known velocity points')
    if vptsfile:
        vPoints = readVelpoints(vptsfile)

    else:
        vPoints = np.array([])
    if model == 'slow':
        return jntHyposlow_T(data, caldata, Vint, cells, nodes, rcv, Hypo0,
                             par, NThreadsUser, vPoints, basename)
    elif model == 'vel':
        return jntHypoVel_T(data, caldata, Vint, cells, nodes, rcv, Hypo0,
                            par, NThreadsUser, vPoints, basename)
    else:
        print('invalide variable model\n')
        sys.stdout.flush()
        return 0.


def jointHypoVelPS_T(inputFileParam, model='slow'):
    """
     Joint hypocenter-velocity inversion using P- and S-wave arrival time data.

     Parameters
     ----------
     inputFileParam : string
          Text file containing inversion parameters and data filenames.
     model : string
          Sought models: 'vel' for an inversion problem parametrized using
          the velocity model, 'slow' for an inversion problem parametrized
          using the slowness model. The default is 'slow'.

     Returns
     -------
     python dictionary
          It contains the estimated hypocenter coordinates and their origin times,
          static correction values, velocity models of P and S waves, hypocenter
          convergence states, parameter uncertainty and residual norm in each
          iteration.

     """
    ParametersFile = fileReader(inputFileParam)
    par = ParametersFile.getIversionParam()
    basename = ParametersFile.readParameter('base name')
    #  read mesh parameters
    MESH = MSHReader(ParametersFile.readParameter('mesh file'))
    nodes = MESH.readNodes()
    cells = MESH.readTetraherdonElements()
    # red rcv coordinates
    stations = RCVReader(ParametersFile.readParameter('rcvfile'))
    rcv = stations.getStation()
    # observed traveltimes
    if ParametersFile.readParameter('arrival times'):
        data = readEventsFiles(
            ParametersFile.readParameter('arrival times'), True)
    else:
        data = (np.array([]), np.array([]))
    # get calibration data
    if ParametersFile.readParameter('Time calibration'):
        caldata = readEventsFiles(
            ParametersFile.readParameter('Time calibration'), True)
    else:
        caldata = (np.array([]), np.array([]))
    # get initial velocity models for P and S waves
    Vpint = np.loadtxt(ParametersFile.readParameter('Velocity P waves'), ndmin=2)
    Vsint = np.loadtxt(ParametersFile.readParameter('Velocity S waves'), ndmin=2)
    Vinit = (Vpint, Vsint)
    # get initial parameters of Hyocenters0 and origin times
    Hypo0 = readEventsFiles(ParametersFile.readParameter('Hypo0'))
    # get and set number of threads
    NThreadsUser = ParametersFile.readParameter('number of threads', int)
    # known v points
    vptsfile = ParametersFile.readParameter('known velocity points')
    if vptsfile:
        vPoints_p, vPoints_s = readEventsFiles(vptsfile, True)
        vPoints = (vPoints_p, vPoints_s)
    else:
        vPoints = (np.array([]), np.array([]))
    if model == 'slow':
        return jntHyposlowPS_T(data, caldata, Vinit, cells, nodes, rcv, Hypo0,
                               par, NThreadsUser, vPoints, basename)
    elif model == 'vel':
        return jntHypoVelPS_T(data, caldata, Vinit, cells, nodes,
                              rcv, Hypo0, par, NThreadsUser, vPoints, basename)
    else:
        print('invalide variable model\n')
        sys.stdout.flush()
        return 0.


if __name__ == '__main__':
    results = jointHypoVel_T('localisation_P.par', 'slow')

from __future__ import division

import numpy
import globl
from kalman import *


def func_stage3(filter_range, logrgdp, exanterr, pceinflation, piimpgap, pioilgap, pi3, pi5, a1, a2, a3, b1, b2, b3, b4, b5, c, sig1, sig2, sig4, lamg, lamz, x0):
    # Set Kalman filter coefficients
    Cf = numpy.array([[1,0], [-a1,-b3], [-a2,0], [-c*a3/2,0], [-c*a3/2,0], [-a3/2,0], [-a3/2,0]])   # Measurement matrix

    swf = numpy.diag(numpy.array([sig4**2, 0, 0, (lamg*sig4)**2, 0, 2*(lamz*sig1/a3)*2, 0]))   # Process noise covariance
    swv = numpy.diag(numpy.array([sig1**2, sig2**2]))   # Measurement noise covariance

    # Compute mu_f - measurement shift
    regf = numpy.array([[a1, a2, a3/2, a3/2, 0, 0, 0, 0, 0], [b3, 0, 0, 0, b1, b2, 1-b1-b2, b4, b5]])

    eqnxvector_st2 = numpy.array(zip(logrgdp[filter_range-1], logrgdp[filter_range-2], exanterr[filter_range-1], exanterr[filter_range-2], pceinflation[filter_range-1], pi3[filter_range-2], pi5[filter_range-5],     pioilgap[filter_range-1], piimpgap[filter_range]))

    globl.muf = numpy.dot(eqnxvector_st2, numpy.transpose(regf))   # Measurement shift

    filter_measurement = numpy.array(zip(logrgdp[filter_range], pceinflation[filter_range])) - globl.muf

    ## Test
    #numpy.savetxt("3_filter_measurement_1.dat", filter_measurement, fmt='%16.8E', delimiter='', newline='\n', header='', footer='', comments='# ')
    ##tmp = raw_input("Press ENTER to continue...")

    # Initialize learnt coefficients
    Ql = numpy.copy(swf)
    Rl = numpy.copy(swv)
    Cl = numpy.transpose(Cf)
    initx = numpy.copy(x0)
    initV = numpy.copy(swf)

    # State Transition matrix
    # Now the state contains y* (with 2 lags), g and z (with 1 lag each)
    A = numpy.array([[1,0,0,1,0,0,0], [1,0,0,0,0,0,0], [0,1,0,0,0,0,0], [0,0,0,1,0,0,0], [0,0,0,1,0,0,0], [0,0,0,0,0,1,0], [0,0,0,0,0,1,0]])

    # Learning
    # First learning phase is with logrgdp only
    [A, Cl, Ql, Rl, initx, initV, LL] = learn_kalman(filter_measurement[:,0:1].T, A, Cf[:,0:1].T, swf, swv[0,0], np.matrix(x0),initV,\
    5000,1,1,0,kalmanLearningConstraint,A,Cf[:,0:1].T,3);

    Cl = numpy.vstack((Cl, numpy.transpose(Cf[:,1])))
    Rl = numpy.array([[Rl,0], [0,swv[1,1]]])

    # Second stage with complete data
    [A, Cl, Ql, Rl, initx, initV, LL] = learn_kalman(filter_measurement.T, A, Cl, Ql, Rl, initx,initV,\
    5000,1,1,0,kalmanLearningConstraint,A,Cf.T,3);

    # Apply the Filter with learnt coefficients
    (xstates, _, _, _) = kalman_smoother(filter_measurement.T, A, Cl, Ql, Rl,initx,initV);

    ## Test
    #numpy.savetxt("3_xstates_save.dat", xstates, fmt='%25.16E', delimiter='', newline='\n', header='', footer='', comments='# ')
    ##tmp = raw_input("Press ENTER to continue...")

    #xstates = numpy.loadtxt("MATLAB/3_xstates.dat", usecols=(0,1,2,3,4,5,6), unpack=False)
    ## Test
    #numpy.savetxt("3_xstates_load.dat", xstates, fmt='%25.16E', delimiter='', newline='\n', header='', footer='', comments='# ')

    xstates = numpy.transpose(xstates)

    SIG1 = numpy.sqrt(Rl[0,0])
    SIG2 = numpy.sqrt(Rl[1,1])
    SIG3 = numpy.sqrt(Ql[5,5])
    SIG4 = numpy.sqrt(Ql[0,0])
    SIG5 = numpy.sqrt(Ql[3,3])

    LAMG = SIG5 / SIG4
    LAMZ = (SIG3 / SIG1) * numpy.abs(a3/numpy.sqrt(2))

    C = Cl[0,3] / Cl[0,5]

    xmeasure = numpy.dot(Cl, xstates.T)

    return (C, SIG1, SIG2, SIG3, SIG4, SIG5, LAMG, LAMZ, filter_measurement, xstates, xmeasure)



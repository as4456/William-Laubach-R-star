from __future__ import division

from scipy.signal import lfilter
from scipy import linalg
from kalman import *
import matplotlib.pyplot as pyplot

#from import_data_file import *
from import_data_csv import*
from functions import *


class stage1_class:
    # Compute the 3 and 5 period moving averages of inflation
    def run(self):
        #import_data()
        import_data_csv()
        globl.n_data = len(globl.logrgdp)
        print "globl.logrgdp", globl.logrgdp
        print "globl.n_data", globl.n_data

        # time vector
        t = numpy.arange(1, globl.n_data+1)

        f3 = numpy.ones(3) / 3
        f5 = numpy.ones(5) / 5

        # Compute running averages of pce-inflation
        globl.pi3 = lfilter(f3, 1, globl.pceinflation)
        globl.pi5 = lfilter(f5, 1, globl.pceinflation)

        ## Test
        #numpy.savetxt("1_pi3_pi5.dat", numpy.array(zip(globl.pceinflation, globl.pi3, globl.pi5)), fmt='%10.4f', delimiter='', newline='\n', header='', footer='', comments='# ')

        # Scale up GDP
        globl.logrgdp = 100.0 * globl.logrgdp

        # Compute auxiliary variables
        globl.pioilgap = globl.oilinflation - globl.pceinflation
        globl.piimpgap = globl.impinflation - globl.pceinflation
        globl.exanterr = globl.shortrate - globl.expinflation

        # Allocate variables
        globl.gap    = numpy.zeros(globl.n_data)
        globl.ystar  = numpy.zeros(globl.n_data)
        globl.grates = numpy.zeros(globl.n_data)
        globl.nrr    = numpy.zeros(globl.n_data)   # natural rate of interest

        # Calculate indices corresponding to different dates
        indx_1960_1 = yearQuarter2Index(1960, 1)#start quarter
        #indx_start
        #print "indx_1960_1", indx_1960_1
        globl.indx_1961_1 = yearQuarter2Index(1961, 1)
        #print "globl.indx_1961_1", globl.indx_1961_1
        indx_2002_2 = yearQuarter2Index(2002, 2) 
        #print "indx_2002_2", indx_2002_2
        # indx_2002_2 = yearQuarter2Index(2013, 4)
        

        

        # Stage 1
        #==================================================================================================================
        print "STAGE 1: GENERATION OF FIRST ESTIMATES\n\n"

        trend = numpy.copy(t)
        
        btrend1 = t - yearQuarter2Index(1973, 4)
        #print yearQuarter2Index(1973, 4)
        #print "btrend1", btrend1
        btrend2 = t - yearQuarter2Index(1995, 2)
        #print yearQuarter2Index(1995, 2)
        #print "btrend2", btrend2
        btrend1 = numpy.where(btrend1>0, btrend1, 0)
        #print "btrend1 again", btrend1
        btrend2 = numpy.where(btrend2>0, btrend2, 0)
        #print btrend2

        ## Test
        #numpy.savetxt("1_trend_btrend1.dat", numpy.array(zip(trend,btrend1,btrend2)), fmt='%6i', delimiter='', newline='\n', header='', footer='', comments='# ')

        n_rows = indx_2002_2 - indx_1960_1 + 1
        print "n_rows", n_rows
        # Design matrix is the left handside matrix in linear least-squares
        print "\nCreating an estimate for y* and output gap\n"
        #print "trend[indx_1960_1-1:indx_2002_2]", trend[indx_1960_1-1:indx_2002_2]
        #print "btrend1[indx_1960_1-1:indx_2002_2]", btrend1[indx_1960_1-1:indx_2002_2]
        #print "btrend2[indx_1960_1-1:indx_2002_2]", btrend2[indx_1960_1-1:indx_2002_2]
        design_matrix = numpy.array(zip(numpy.ones(n_rows), trend[indx_1960_1-1:indx_2002_2], btrend1[indx_1960_1-1:indx_2002_2], btrend2[indx_1960_1-1:indx_2002_2]))
        #print "design_matrix", design_matrix
        #print "shape", numpy.shape(design_matrix)
        #print "shape gdp", numpy.shape(globl.logrgdp[indx_1960_1-1:indx_2002_2])
        lin_reg_output = linalg.lstsq(design_matrix, globl.logrgdp[indx_1960_1-1:indx_2002_2])[0]

        ## Test
        #numpy.savetxt("1_design_matrix.dat", design_matrix, fmt='%6i', delimiter='', newline='\n', header='', footer='', comments='# ')
        #numpy.savetxt("1_lin_reg_output.dat", lin_reg_output, fmt='%12.6f', delimiter='', newline='\n', header='', footer='', comments='# ')


        globl.ystar[indx_1960_1-1:indx_2002_2] = numpy.dot(design_matrix, lin_reg_output)
        globl.gap = globl.logrgdp - globl.ystar
        print "Done.\n"

        # IS equation
        print "\nFitting IS-equation\n"
        rang = numpy.arange(globl.indx_1961_1, globl.n_data+1)
        design_matrix = numpy.array(zip(globl.gap[rang-2], globl.gap[rang-3]))
        lin_reg_output = linalg.lstsq(design_matrix, globl.gap[rang-1])[0]

        ## Test
        #numpy.savetxt("1_design_matrix_2.dat", design_matrix, fmt='%12.6f', delimiter='', newline='\n', header='', footer='', comments='# ')
        #numpy.savetxt("1_lin_reg_output_2.dat", lin_reg_output, fmt='%12.6f', delimiter='', newline='\n', header='', footer='', comments='# ')

        a1_st1 = lin_reg_output[0]
        a2_st1 = lin_reg_output[1]
        sig1_st1 = numpy.std(numpy.dot(design_matrix,lin_reg_output)-globl.gap[rang-1], ddof=1)
        print "Done.\n"


        # Phillips curve
        print "\nFitting Phillips curve\n"
        design_matrix = numpy.array(zip(globl.gap[rang-2], globl.pceinflation[rang-2], globl.pi3[rang-3], globl.pi5[rang-6], globl.pioilgap[rang-2], globl.piimpgap[rang-1]))
        lin_reg_output = linalg.lstsq(design_matrix, globl.pceinflation[rang-1])[0]

        ## Test
        #numpy.savetxt("1_design_matrix_3.dat", design_matrix, fmt='%12.6f', delimiter='', newline='\n', header='', footer='', comments='# ')
        #numpy.savetxt("1_lin_reg_output_3.dat", lin_reg_output, fmt='%12.6f', delimiter='', newline='\n', header='', footer='', comments='# ')

        b3_st1 = lin_reg_output[0];
        b1_st1 = lin_reg_output[1];
        b2_st1 = lin_reg_output[2];
        b4_st1 = lin_reg_output[4];
        b5_st1 = lin_reg_output[5];
        sig2_st1 = numpy.std(numpy.dot(design_matrix,lin_reg_output)-globl.pceinflation[rang-1], ddof=1)
        print "Done.\n"

        # Trend
        dystar = numpy.diff(globl.ystar)
        g = numpy.mean(dystar[indx_1960_1-1:])
        globl.sig4_st1 = numpy.std(dystar[indx_1960_1-1:], ddof=1)

        #KALMAN FILTER
        print "\nApplying Kalman Filter\n"

        # A state vector consists of ystar and its 2 lags, and g (which is modelled as a constant)

        A = numpy.array([[1,0,0,1], [1,0,0,0], [0,1,0,0], [0,0,0,1]])      # State Transition matrix
        f = numpy.array([1, 0, 0, 0])
        #zf = [1;0;0];                                                     # Process shift (control term) - removed
        Cf = numpy.array([[1,0], [-a1_st1,-b3_st1], [-a2_st1,0], [0,0]])   # Measurement matrix
        swf = numpy.diag(globl.sig4_st1**2*f)                                    # Process noise covariance
        swv = numpy.diag(numpy.array([sig1_st1**2, sig2_st1**2]))       # Measurement noise covariance
        regf = numpy.array([[a1_st1,a2_st1,0,0,0,0,0], [b3_st1,0,b1_st1,b2_st1,1-b1_st1-b2_st1,b4_st1,b5_st1]])
        filter_range = numpy.arange(globl.indx_1961_1-1, globl.n_data)

        # Compute mu_f - measurement shift
        eqnxvector_st1 = numpy.array(zip(globl.logrgdp[filter_range-1], globl.logrgdp[filter_range-2], globl.pceinflation[filter_range-1], globl.pi3[filter_range-2], globl.pi5[filter_range-5], globl.pioilgap[filter_range-1], globl.piimpgap[filter_range]))

        globl.muf = numpy.dot(eqnxvector_st1, numpy.transpose(regf))   # Measurement shift

        filter_measurement =  numpy.array(zip(globl.logrgdp[filter_range], globl.pceinflation[filter_range])) - globl.muf

        # Allocate filter output
        xmeasure = numpy.zeros((2, len(filter_range)))
        xstates  = numpy.zeros((3, len(filter_range)))

        x0 = numpy.hstack((globl.ystar[filter_range[0]-2:filter_range[0]+1], g))   # initial estimate of the state vector
        xcorr0 = swf

        # Learn filter coefficients (x0 and covariances)
        Cl = numpy.transpose(Cf)
        globl.Ql = numpy.copy(swf)
        globl.Rl = numpy.copy(swv)

        ## Test
        #numpy.savetxt("1_filter_measurement.dat", filter_measurement, fmt='%12.6f', delimiter='', newline='\n', header='', footer='', comments='# ')

        # Learn filter coefficients (x0 and covariances)
        #kf = KalmanFilter(transition_matrices=A, observation_matrices=Cl, transition_covariance=globl.Ql, observation_covariance=globl.Rl, initial_state_mean=x0, initial_state_covariance=xcorr0)
        #kf.em(filter_measurement, n_iter=50)
        #globl.Ql = numpy.copy(kf.transition_covariance)
        #globl.Rl = numpy.copy(kf.observation_covariance)
        #(xstates, xvariance) = kf.smooth(filter_measurement)
        (A, Cl, Ql, Rl, x0, xcorr0, LL) = learn_kalman(filter_measurement.T, A, Cl, swf, swv, x0,xcorr0,\
        5000,1,1,0,kalmanLearningConstraint,A,Cl,1);

        (xstates, _, _, _) = kalman_smoother(filter_measurement.T, A, Cl, Ql, Rl, x0,xcorr0);

        ## Test
        #numpy.savetxt("1_xstates_save.dat", xstates, fmt='%25.16E', delimiter='', newline='\n', header='', footer='', comments='# ')

        #xstates = numpy.loadtxt("MATLAB/1_xstates.dat", usecols=(0,1,2,3), unpack=False)
        ## Test
        #numpy.savetxt("1_xstates_load.dat", xstates, fmt='%25.16E', delimiter='', newline='\n', header='', footer='', comments='# ')

        xmeasure = numpy.dot(numpy.transpose(Cf), xstates)

        sig1_st1 = numpy.sqrt(globl.Rl[0,0])
        sig2_st1 = numpy.sqrt(globl.Rl[1,1])
        globl.sig4_st1 = numpy.sqrt(globl.Ql[0,0])

        globl.ystar[filter_range] = xstates[0,:]

        globl.gap[filter_range] = globl.logrgdp[filter_range] - globl.ystar[filter_range]


        #return
        # Create figure
        figure = pyplot.figure(num=None, figsize=(8, 6), dpi=100, facecolor='w')
        subplot = figure.add_subplot(1,1,1, position=[0.15, 0.10, 0.75, 0.80], frame_on=True, zorder=0)

        subplot.set_title("KALMAN FILTER - STAGE 1")
        subplot.xaxis.set_label_text("Quarter number", fontsize=15, verticalalignment='top')
        subplot.yaxis.set_label_text("Measurement", fontsize=15, verticalalignment='center')
        subplot.yaxis.labelpad = 25

        subplot.plot(filter_range, filter_measurement[:,0]+globl.muf[:,0],        linewidth=2, marker='', markersize=3, zorder=1, label="log(rGDP) - measured")
        subplot.plot(filter_range, filter_measurement[:,1]+globl.muf[:,1],        linewidth=2, marker='', markersize=3, zorder=2, label="PCE-inflation - measured")
        subplot.plot(filter_range, numpy.transpose(xmeasure)[:,0]+globl.muf[:,0], linewidth=2, marker='', markersize=3, zorder=3, label="log(rGDP) - filtered")
        subplot.plot(filter_range, numpy.transpose(xmeasure)[:,1]+globl.muf[:,1], linewidth=2, marker='', markersize=3, zorder=4, label="PCE-inflation - filtered")

        leg = subplot.legend(loc='upper right', bbox_to_anchor=(0.95,0.70), ncol=1, borderaxespad=0.0, borderpad=0.6, numpoints=1, handlelength=1.3, fancybox=True, shadow=True)
        leg_frame = leg.get_frame()
        leg_frame.set_linewidth(2)
        leg_frame.set_facecolor('0.90')
        for t in leg.get_texts():
            t.set_fontsize(10)    # the legend text fontsize

        pyplot.savefig("stage1.pdf")
        pyplot.show()

#stage1 = stage1_class()
#stage1.run()
    #





from __future__ import division

from kalman import *
from stage1_file import *


class stage2_class:
    def run(self):
        # Execute stage 1
        stage1 = stage1_class()
        stage1.run()

        # Stage 2
        #==============================================================================================================
        print "\n\nSTAGE 2: Ex-ante real rates are added to the regressor list\n\n"

        # IS equation
        print "\nFitting IS-equation\n"
        rang = numpy.arange(globl.indx_1961_1-1, globl.n_data)
        n_row = len(rang)

        # Design matrix is the left handside matrix in linear least-squares
        design_matrix = numpy.array(zip(globl.gap[rang-1], globl.gap[rang-2], globl.shortrate[rang-1], globl.shortrate[rang-2], numpy.ones(n_row)))
        lin_reg_output = linalg.lstsq(design_matrix, globl.gap[rang])[0]

        ## Test
        #numpy.savetxt("2_design_matrix_1.dat", design_matrix, fmt='%12.6f', delimiter='', newline='\n', header='', footer='', comments='# ')
        #numpy.savetxt("2_lin_reg_output_1.dat", lin_reg_output, fmt='%12.6f', delimiter='', newline='\n', header='', footer='', comments='# ')

        print "\nDone.\n"

        a1 = lin_reg_output[0]
        a2 = lin_reg_output[1]
        a3 = (lin_reg_output[2] + lin_reg_output[3]) / 2
        a4 = lin_reg_output[4]
        sig1 = numpy.std(numpy.dot(design_matrix,lin_reg_output)-globl.gap[rang], ddof=1)
        a5 = -a3

        # Phillips curve
        print "\nFitting Phillips curve\n"
        design_matrix = numpy.array(zip(globl.gap[rang-1], globl.pceinflation[rang-1], globl.pi3[rang-2], globl.pi5[rang-5], globl.pioilgap[rang-1], globl.piimpgap[rang]))
        lin_reg_output = linalg.lstsq(design_matrix, globl.pceinflation[rang])[0]

        ## Test
        #numpy.savetxt("2_design_matrix_2.dat", design_matrix, fmt='%12.6f', delimiter='', newline='\n', header='', footer='', comments='# ')
        #numpy.savetxt("2_lin_reg_output_2.dat", lin_reg_output, fmt='%12.6f', delimiter='', newline='\n', header='', footer='', comments='# ')

        b3 = lin_reg_output[0];
        b1 = lin_reg_output[1];
        b2 = lin_reg_output[2];
        b4 = lin_reg_output[4];
        b5 = lin_reg_output[5];
        sig2 = numpy.std(numpy.dot(design_matrix,lin_reg_output)-globl.pceinflation[rang], ddof=1)

        print "\nDone.\n"

        globl.sig4 = globl.sig4_st1

        #KALMAN FILTER
        print "\nApplying Kalman Filter\n"

        filter_range = numpy.arange(globl.indx_1961_1-1, globl.n_data)
        # The state vector of the process conists of ystar and its 2 lags; grate
        # without lags and a constant term. The constant term models the constant
        # in RATS

        A = numpy.array([[1,0,0,1,0], [1,0,0,0,0], [0,1,0,0,0], [0,0,0,1,0], [0,0,0,0,1]])   # State Transition matrix

        Cf = numpy.array([[1,0], [-a1,-b3], [-a2,0], [a5,0], [1,0]])   # Measurement matrix

        globl.lamg = 0.11
        swf = numpy.diag(numpy.array([globl.sig4**2, 0, 0, (globl.lamg*globl.sig4)**2, 0]))   # Process noise covariance
        swv = numpy.diag(numpy.array([sig1**2, sig2**2]))   # Measurement noise covariance

        # Compute mu_f - measurement shift
        regf = numpy.array([[a1,a2,a3/2,a3/2,0,0,0,0,0], [b3,0,0,0,b1,b2,1-b1-b2,b4,b5]])
        eqnxvector_st2 = numpy.array(zip(globl.logrgdp[filter_range-1], globl.logrgdp[filter_range-2], globl.exanterr[filter_range-1], globl.exanterr[filter_range-2], globl.pceinflation[filter_range-1], globl.pi3[filter_range-2], globl.pi5[filter_range-5], globl.pioilgap[filter_range-1], globl.piimpgap[filter_range]))

        globl.muf = numpy.dot(eqnxvector_st2, numpy.transpose(regf))   # Measurement shift (intercept)

        filter_measurement = numpy.array(zip(globl.logrgdp[filter_range], globl.pceinflation[filter_range])) - globl.muf
        ise = numpy.hstack((globl.ystar[filter_range[0]-2:filter_range[0]+1], globl.ystar[filter_range[1]]-globl.ystar[filter_range[0]], -a4))   # initial state estimate

        ## Test
        #numpy.savetxt("2_filter_measurement.dat", filter_measurement, fmt='%12.6f', delimiter='', newline='\n', header='', footer='', comments='# ')
        #numpy.savetxt("2_ise.dat", ise, fmt='%12.6f', delimiter='', newline='\n', header='', footer='', comments='# ')

        x0 = numpy.copy(ise)
        xcorr0 = numpy.copy(swf)

        # Learn coefficients when possible
        Cl = numpy.transpose(Cf)
        [A, Cl, Ql, Rl, x0, xcorr0, LL ] = learn_kalman(filter_measurement.T, A, Cl, swf, swv, x0,xcorr0,\
          5000,1,1,0,kalmanLearningConstraint,A,Cl,2);

        # Apply the filter
        (xstates, _, _, _) = kalman_smoother(filter_measurement.T, A, Cl, Ql, Rl, x0,xcorr0);

        ## Test
        #numpy.savetxt("2_xstates_save.dat", xstates, fmt='%25.16E', delimiter='', newline='\n', header='', footer='', comments='# ')
        #fout = open("2_xvariance_save.dat", 'w')
        #(ii, jj, kk) = numpy.shape(xvariance)
        #for i in range(0, ii):
            #for j in range(0, jj):
                #for k in range(0, kk):
                    #print >> fout, "%4i%4i%4i   %25.16E" % (i+1, j+1, k+1, xvariance[i,j,k])
        ##
        #fout.close()
        ##numpy.savetxt("2_xvariance_save.dat", numpy.reshape(xvariance, (xvariance.size,)), fmt='%25.16E', delimiter='', newline='\n', header='', footer='', comments='# ')

        #xstates = numpy.loadtxt("MATLAB/2_xstates.dat", usecols=(0,1,2,3,4), unpack=False)
        ## Test
        #numpy.savetxt("2_xstates_load.dat", xstates, fmt='%25.16E', delimiter='', newline='\n', header='', footer='', comments='# ')

        #xvariance = numpy.transpose(xvariance)

        xmeasure = numpy.dot(numpy.transpose(Cf), xstates)

        # Save learnt coefficients
        sig1 = numpy.sqrt(Rl[0,0])
        sig2 = numpy.sqrt(Rl[1,1])
        globl.sig4 = numpy.sqrt(Ql[0,0])
        sig5 = numpy.sqrt(Ql[3,3])
        a5 = Cl[0,3]
        a4 = -numpy.mean(xstates[4,:])
        globl.lamg = sig5 / globl.sig4

        globl.ystar[filter_range] = xstates[0,:]
        globl.grates[filter_range] = xstates[3,:]
        globl.gap[filter_range] = globl.logrgdp[filter_range] - globl.ystar[filter_range]

        # Save parameters
        a1_st2 = a1
        a2_st2 = a2
        a3_st2 = a3
        a4_st2 = a4
        a5_st2 = a5

        b1_st2 = b1
        b2_st2 = b2
        b3_st2 = b3
        b4_st2 = b4
        b5_st2 = b5

        sig1_st2 = sig1
        sig2_st2 = sig2
        sig4_st2 = globl.sig4


        #return
        # Create figure
        figure = pyplot.figure(num=None, figsize=(8, 6), dpi=100, facecolor='w')
        subplot = figure.add_subplot(1,1,1, position=[0.15, 0.10, 0.75, 0.80], frame_on=True, zorder=0)

        subplot.set_title("KALMAN FILTER - STAGE 2")
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

        pyplot.savefig("stage2.pdf")
        pyplot.show()



#!/usr/bin/env python
# The above line is needed when the code is executed in Linux.
from __future__ import division
import numpy
from stage2_file import *
from linear_estimates_file import *
from func_stage3_file import *

stage2 = stage2_class()
stage2.run()


# Stage 3
#======================================================================================================================

print "\n\nSTAGE 3:FINAL STAGE\n\n"

rang = numpy.arange(globl.indx_1961_1-1, globl.n_data)
ncycles_st3_1 = 1

filter_range = numpy.arange(globl.indx_1961_1-1, globl.n_data)

# Initial guesses for some unknown parameters
lamz = numpy.sqrt(2) * .058
c = 1.068
# Initial estimation of the first state
ise = numpy.array(numpy.hstack((globl.ystar[filter_range[0]-2:filter_range[0]+1], globl.grates[filter_range[1]], globl.grates[filter_range[0]], [0], [0])))   # initial state estimate
x0 = numpy.copy(ise)

# Get all the linear estimates of a's, b's and associated sigma's
(a1, a2, a3, a4, a5, b1, b2, b3, b4, b5, sig1, sig2) = linear_estimates_st3(rang, globl.gap, globl.shortrate, globl.pceinflation, globl.pi3, globl.pi5, globl.pioilgap, globl.piimpgap)

## Test
#fileID = open("3_a_b.dat", "w");
#print >> fileID, "%12.6f" % a1
#print >> fileID, "%12.6f" % a2
#print >> fileID, "%12.6f" % a3
#print >> fileID, "%12.6f" % a4
#print >> fileID, "%12.6f" % a5
#print >> fileID, "%12.6f" % b1
#print >> fileID, "%12.6f" % b2
#print >> fileID, "%12.6f" % b3
#print >> fileID, "%12.6f" % b4
#print >> fileID, "%12.6f" % b5
#print >> fileID, "%12.6f" % sig1
#print >> fileID, "%12.6f" % sig2
#fileID.close()

for i in range(1, ncycles_st3_1+1):
    if (i > 1):
        x0 = xstates[:,0]
    #

    #KALMAN FILTER
    print "\nApplying Kalman Filter\n"

    # All the Kalman filtering happens now inside this function
    (c, sig1, sig2, sig3, globl.sig4, sig5, globl.lamg, lamz, filter_measurement, xstates, xmeasure) = func_stage3(filter_range, globl.logrgdp, globl.exanterr, globl.pceinflation, globl.piimpgap, globl.pioilgap, globl.pi3, globl.pi5, a1, a2, a3, b1, b2, b3, b4, b5, c, sig1, sig2, globl.sig4, globl.lamg, lamz, x0)

    xstates = xstates.T

    globl.ystar[filter_range] = xstates[0,:]
    globl.grates[filter_range] = xstates[3,:]
    globl.gap[filter_range] = globl.logrgdp[filter_range] - globl.ystar[filter_range]

    globl.nrr[filter_range] = numpy.dot(c, globl.grates[filter_range]) + numpy.transpose(xstates[5,:])

    # Get all the linear estimates of a's, b's. sigma's from Kalman filter are more accurate
    (a1, a2, a3, a4, a5, b1, b2, b3, b4, b5, dummy1, dummy2) = linear_estimates_st3(rang, globl.gap, globl.shortrate, globl.pceinflation, globl.pi3, globl.pi5, globl.pioilgap, globl.piimpgap)
#

# Save parameters
a1_st3_1 = a1
a2_st3_1 = a2
a3_st3_1 = a3
a4_st3_1 = a4
a5_st3_1 = a5

b1_st3_1 = b1
b2_st3_1 = b2
b3_st3_1 = b3
b4_st3_1 = b4
b5_st3_1 = b5

sig1_st3_1 = sig1
sig2_st3_1 = sig2
sig3_st3_1 = sig3
sig4_st3_1 = globl.sig4
sig5_st3_1 = sig5

c_st3_1 = c
lamg_st3_1 = globl.lamg
lamz_st3_1 = lamz

## Test
#fileID = open("3_a_b_2.dat", "w");
#print >> fileID, "%12.6f" % a1_st3_1
#print >> fileID, "%12.6f" % a2_st3_1
#print >> fileID, "%12.6f" % a3_st3_1
#print >> fileID, "%12.6f" % a4_st3_1
#print >> fileID, "%12.6f" % a5_st3_1
#print >> fileID
#print >> fileID, "%12.6f" % b1_st3_1
#print >> fileID, "%12.6f" % b2_st3_1
#print >> fileID, "%12.6f" % b3_st3_1
#print >> fileID, "%12.6f" % b4_st3_1
#print >> fileID, "%12.6f" % b5_st3_1
#print >> fileID
#print >> fileID, "%12.6f" % sig1
#print >> fileID, "%12.6f" % sig2
#print >> fileID, "%12.6f" % sig3
#print >> fileID, "%12.6f" % globl.sig4
#print >> fileID, "%12.6f" % sig5
#print >> fileID
#print >> fileID, "%12.6f" % c_st3_1
#print >> fileID, "%12.6f" % globl.lamg
#print >> fileID, "%12.6f" % lamz
#fileID.close()


# Create figure
figure = pyplot.figure(num=None, figsize=(8, 6), dpi=100, facecolor='w')
subplot = figure.add_subplot(1,1,1, position=[0.15, 0.10, 0.75, 0.80], frame_on=True, zorder=0)

subplot.set_title("KALMAN FILTER - STAGE 3(1)")
subplot.xaxis.set_label_text("Quarter number", fontsize=15, verticalalignment='top')
subplot.yaxis.set_label_text("Measurement", fontsize=15, verticalalignment='center')
subplot.yaxis.labelpad = 25

subplot.plot(filter_range, filter_measurement[:,0]+globl.muf[:,0],        linewidth=2, marker='', markersize=3, zorder=1, label="log(rGDP) - measured")
subplot.plot(filter_range, filter_measurement[:,1]+globl.muf[:,1],        linewidth=2, marker='', markersize=3, zorder=2, label="PCE-inflation - measured")
subplot.plot(filter_range, np.asarray(xmeasure)[0,:]+globl.muf[:,0], linewidth=2, marker='', markersize=3, zorder=3, label="log(rGDP) - filtered")
subplot.plot(filter_range, np.asarray(xmeasure)[1,:]+globl.muf[:,1], linewidth=2, marker='', markersize=3, zorder=4, label="PCE-inflation - filtered")

leg = subplot.legend(loc='upper right', bbox_to_anchor=(0.95,0.70), ncol=1, borderaxespad=0.0, borderpad=0.6, numpoints=1, handlelength=1.3, fancybox=True, shadow=True)
leg_frame = leg.get_frame()
leg_frame.set_linewidth(2)
leg_frame.set_facecolor('0.90')
for t in leg.get_texts():
    t.set_fontsize(10)    # the legend text fontsize

pyplot.savefig("stage3_1.pdf")
pyplot.show()



# Plot the unobserved variables
figure2 = pyplot.figure(num=None, figsize=(8, 6), dpi=100, facecolor='w')

subplot1 = figure2.add_subplot(3,1,1, position=[0.15, 0.70, 0.75, 0.20], frame_on=True, zorder=0)
subplot1.set_title("Natural rate of interest")
subplot1.plot(filter_range, globl.nrr[filter_range], linewidth=2, marker='', markersize=3, zorder=1, label="")
#subplot1.set_ylim([-5, 10])

subplot1 = figure2.add_subplot(3,1,2, position=[0.15, 0.40, 0.75, 0.20], frame_on=True, zorder=0)
subplot1.set_title("Trend growth rate")
subplot1.plot(filter_range, 4*globl.grates[filter_range], linewidth=2, marker='', markersize=3, zorder=1, label="")
#subplot1.set_ylim([2, 5])

subplot1 = figure2.add_subplot(3,1,3, position=[0.15, 0.10, 0.75, 0.20], frame_on=True, zorder=0)
subplot1.set_title("Output gap")
subplot1.plot(filter_range, globl.gap[filter_range], linewidth=2, marker='', markersize=3, zorder=1, label="")
#subplot1.set_ylim([-10, 5])

pyplot.savefig("stage3_2.pdf")
pyplot.show()

# Saving data also to csv format
out = np.vstack((filter_range,
                 globl.nrr[filter_range],
                 4*globl.grates[filter_range],
                 globl.gap[filter_range])).T
numpy.savetxt("output.csv", out, delimiter=",", fmt="%10.5f")


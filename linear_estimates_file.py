from scipy import linalg
import numpy


def linear_estimates_st3(rang, gap, shortrate, pceinflation, pi3, pi5, pioilgap, piimpgap):
    # IS equation

    print "\nFitting IS-equation\n"
    n_rows = len(rang)
    # Design matrix is the left handside matrix in linear least-squares
    design_matrix = numpy.array(zip(gap[rang-1], gap[rang-2], shortrate[rang-1], shortrate[rang-2], numpy.ones(n_rows)))
    lin_reg_output = linalg.lstsq(design_matrix, gap[rang])[0]

    print "\nDone.\n"

    ## Test
    #numpy.savetxt("3_design_matrix_1.dat", design_matrix, fmt='%16.8E', delimiter='', newline='\n', header='', footer='', comments='# ')
    #numpy.savetxt("3_gap_1.dat", gap, fmt='%16.8E', delimiter='', newline='\n', header='', footer='', comments='# ')
    #numpy.savetxt("3_lin_reg_output_1.dat", lin_reg_output, fmt='%16.8E', delimiter='', newline='\n', header='', footer='', comments='# ')

    a1 = lin_reg_output[0]
    a2 = lin_reg_output[1]
    a3 = (lin_reg_output[2] + lin_reg_output[3]) / 2
    a4 = lin_reg_output[4]
    a5 = 0
    sig1 = numpy.std(numpy.dot(design_matrix,lin_reg_output)-gap[rang], ddof=1)

    # Phillips curve
    print "\nFitting Phillips curve\n"
    design_matrix = numpy.array(zip(gap[rang-1], pceinflation[rang-1], pi3[rang-2], pi5[rang-5], pioilgap[rang-1], piimpgap[rang]))
    lin_reg_output = linalg.lstsq(design_matrix, pceinflation[rang])[0]

    ## Test
    #numpy.savetxt("3_design_matrix_2.dat", design_matrix, fmt='%16.8E', delimiter='', newline='\n', header='', footer='', comments='# ')
    #numpy.savetxt("3_pceinflation_2.dat", pceinflation, fmt='%16.8E', delimiter='', newline='\n', header='', footer='', comments='# ')
    #numpy.savetxt("3_lin_reg_output_2.dat", lin_reg_output, fmt='%16.8E', delimiter='', newline='\n', header='', footer='', comments='# ')

    b3 = lin_reg_output[0]
    b1 = lin_reg_output[1]
    b2 = lin_reg_output[2]
    b4 = lin_reg_output[4]
    b5 = lin_reg_output[5]
    sig2 = numpy.std(numpy.dot(design_matrix,lin_reg_output)-pceinflation[rang], ddof=1)

    print "\nDone.\n"

    return (a1, a2, a3, a4, a5, b1, b2, b3, b4, b5, sig1, sig2)




import numpy
import globl
import pandas as pd

import numpy as np



def read_data_csv(filename='nr_1214.csv'):

    a=pd.read_csv(filename)
    new=pd.DataFrame(a,index=a.index[1:])
    new["Quarter No."]=a["Quarter No."][1:]
    new["XGDP"]=np.log(new["XGDP"])
    s=[str(i)[-1] for i in new["Quarter No."][0:14]]
    
    if '5' in s:
        new["PCXFE"]=400*np.log(np.divide(new["PCXFE"],a["PCXFE"][0:len(new)]))
        new["IMPPETROL"]=400*np.log(np.divide(new["IMPPETROL"],a["IMPPETROL"][0:len(new)]))
        new["IMPXCOMP"]=400*np.log(np.divide(new["IMPXCOMP"],a["IMPXCOMP"][0:len(new)]))
    else:
        new["PCXFE"]=1200*np.log(np.divide(new["PCXFE"],a["PCXFE"][0:len(new)]))
        new["IMPPETROL"]=1200*np.log(np.divide(new["IMPPETROL"],a["IMPPETROL"][0:len(new)]))
        new["IMPXCOMP"]=1200*np.log(np.divide(new["IMPXCOMP"],a["IMPXCOMP"][0:len(new)]))
    new.to_csv("1.csv")
    #print(len(new))
    
read_data_csv()

# Import data from text file.
def import_data_csv(filename="1.csv"):
    a=pd.read_csv(filename)
    # Initialize arrays
    
    globl.logrgdp      = numpy.array(a["XGDP"])
    #globl.loghoursix   = numpy.zeros(n_line)
    globl.pceinflation = numpy.array(a["PCXFE"])
    globl.expinflation = numpy.array(a["InflExpn"])
    globl.oilinflation = numpy.array(a["IMPPETROL"])
    globl.impinflation = numpy.array(a["IMPXCOMP"])
    globl.shortrate    = numpy.array(a["FFR"])


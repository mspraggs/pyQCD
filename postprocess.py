from os import listdir,system
from os.path import isfile, join
import sys
import IPython
import pylab as pl
from numpy import load
from scipy import optimize

def bin(Ws,binsize=1):
    """Split Ws into bins and return the average of each bin"""
    if binsize == 1:
        return Ws;
    else:
        extra = 0 if pl.size(Ws,axis=0) % binsize == 0 else 1
        dims = [i for i in pl.shape(Ws)]
        dims[0] = dims[0] / binsize + extra
        dims = tuple(dims)
        W_binned = pl.zeros(dims)

        for i in xrange(pl.size(W_binned,axis=0)):
            W_binned[i] = pl.mean(Ws[i*binsize:(i+1)*binsize],axis=0)

        return W_binned          

def V(b,r):
    """Calculates the potential as b[0]*r - b[1]/r + b[2]"""
    return b[0]*r - b[1]/r + b[2]

def efunc(b,r,y):
    """Calculates the error between data and curve"""
    return y - V(b,r)

def potfit(data):
    """Extracts the three paramaeters for the potential
    function from data"""

    r = pl.arange(1,len(data)+1)
    b0 = [1.,1.,1.]
    b,result = optimize.leastsq(efunc,b0,args=(r,data))

    if result != 1:
        print("Warning! Fit failed.")

    return b

def calcaV(W):
    """Calculate aV"""
    return pl.log(pl.absolute(W/pl.roll(W,-1,axis=1)))

def bootstrap(Ws):
    """Bootstraps Ws N times and returns the average"""
    Ws_bstrp = Ws[pl.randint(0,pl.size(Ws,axis=0),pl.size(Ws,axis=0))]
    return Ws_bstrp

def Vplot(Ws,N_bstrp):
    """Calculat the potential function and plot it"""
        
    for i in xrange(N_bstrp):
        W = pl.mean(bootstrap(Ws),axis=0)
        aVs[i] = calcaV(W)

    r = pl.arange(1,7)
    aV = pl.mean(aVs,axis=0)
    e = pl.std(aVs,axis=0)

    b = potfit(aV[:,0])

    print("Fit parameters:")
    print("sigma = %f" % b[0])
    print("b = %f" % b[1])
    print("c = %f" % b[2])

    r_fit = pl.arange(0.25,r[-1]+1,0.1)
    aV_fit = V(b,r_fit)
       
    pl.errorbar(r,aV[:,0],yerr=e[:,0],fmt='ok')
    pl.plot(r_fit,aV_fit,'r--')
    pl.ylim([0,pl.nanmax(aV)+0.25])
    pl.xlim([0,pl.nanmax(r_fit)+0.25])
    pl.xlabel("$r / a$")
    pl.ylabel("$aV(r)$")    
    pl.show()

if __name__ == "__main__":

    files = [f for f in listdir("results") if isfile(join("results",f)) and f[-4:] == ".npy"]

    files.sort()
    
    if len(files) == 0:
        print("No data available.")
        sys.exit()
    
    print("Available data:")
    for i in xrange(len(files)):
        print("(%d) %s" % (i,files[i]))
        
    file_num = input("File: ")
    filename = "results/%s" % files[file_num]
    Ws = load(filename)

    N_bstrp = 100
    binsize = 1
    aVs = pl.zeros((N_bstrp,) + pl.shape(Ws)[1:])
    Ws = bin(Ws)

    Vplot(Ws,N_bstrp=N_bstrp)

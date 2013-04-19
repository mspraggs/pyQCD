from os import listdir,system
from os.path import isfile, join
import sys
import IPython
import pylab as pl
from numpy import load

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

def calcaV(W):
    """Calculate aV"""
    return pl.log(pl.absolute(W/pl.roll(W,-1,axis=1)))

def bootstrap(Ws):
    """Bootstraps Ws N times and returns the average"""
    Ws_bstrp = Ws[pl.randint(0,pl.size(Ws,axis=0),pl.size(Ws,axis=0))]
    return Ws_bstrp

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
        
    for i in xrange(N_bstrp):
        W = pl.mean(bootstrap(Ws),axis=0)
        aVs[i] = calcaV(W)

    t = pl.arange(1,7)
    aV = pl.mean(aVs,axis=0)
    e = pl.std(aVs,axis=0)

    for i in xrange(len(t)):
        pl.errorbar(t,aV[i],yerr=e[i],fmt='o',label=str(i))

    IPython.embed()

    pl.legend()
    pl.show()

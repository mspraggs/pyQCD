from os import listdir,system
from os.path import isdir, join
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
    return b[0]*r + b[1]/r + b[2]

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

def calcaV(W,method = "ratio"):
    """Calculate aV"""
    if method == "ratio":
        return pl.log(pl.absolute(W/pl.roll(W,-1,axis=1)))

    else:
        aVs = pl.zeros(pl.shape(W))
        n = pl.arange(1,pl.size(W,axis=1)+1)
        f = lambda b,t,W: W - b[0] * pl.exp(-b[1] * t)
        
        for i in xrange(pl.size(W,axis=0)):
            params,result = optimize.leastsq(f,[1.,1.],args=(n,W[i]))
            aVs[i] = params[1] * pl.ones(pl.shape(W[i]))
            
        return aVs

def bootstrap(Ws):
    """Bootstraps Ws N times and returns the average"""
    Ws_bstrp = Ws[pl.randint(0,pl.size(Ws,axis=0),pl.size(Ws,axis=0))]
    return Ws_bstrp

def autoCor(Ps,t):
    """Calculates autocorrelation function"""
    meanP = pl.mean(Ps)
    return pl.mean((Ps - meanP) * (pl.roll(Ps,-t) - meanP))

def plotAutoCor(Ps):
    """Calculates and plots the autocorrelation function"""
    
    Cs = pl.zeros(pl.size(Ps)/2)
    t = pl.arange(pl.size(Ps)/2)

    style = raw_input("Please enter a line style: ")

    for i in xrange(pl.size(Ps)/2):
        Cs[i] = autoCor(Ps,t[i])

    pl.plot(t,Cs,style)
    pl.xlabel("$t$")
    pl.ylabel("$C(t)$")

    return Cs

def plotPs(Ps):
	"""Plots Ps values"""
	col = raw_input("Please enter a linestyle: ")
	pl.plot(pl.arange(pl.size(Ps)),Ps,col)

def Vplot(Ws):
    """Calculate the potential function and plot it"""

    N_bstrp = input("Please enter the number of bootstraps: ")
    N_bin = input("Please enter the bin size: ")
    style = raw_input("Please enter a linestyle: ")

    Ws = bin(Ws,N_bin)
    aVs = pl.zeros((N_bstrp,) + pl.shape(Ws)[1:])
        
    for i in xrange(N_bstrp):
        W = pl.mean(bootstrap(Ws),axis=0)
        aVs[i] = calcaV(W,method="fit")

    r = pl.arange(1,7)
    aV = pl.mean(aVs,axis=0)
    e = pl.std(aVs,axis=0)

    b = potfit(aV[:,0])

    a = 0.5 / pl.sqrt((1.65 + b[1]) / b[0])
    sigma = b[0] / a**2
    B = b[1]
    A = b[2] / a

    print("Fit parameters:")
    print("sigma = %f" % sigma)
    print("B = %f" % B)
    print("A = %f" % A)
    print("Lattice spacing, a = %f fm" % a)

    r_fit = pl.arange(0.25,r[-1]+1,0.1)
    aV_fit = V(b,r_fit)

    handles = []
    handles.append(pl.errorbar(r,aV[:,0],yerr=e[:,0],fmt='o'+style[0]))
    handles.append(pl.plot(r_fit,aV_fit,style))
    pl.ylim([0,pl.nanmax(aV)+0.25])
    pl.xlim([0,pl.nanmax(r_fit)+0.25])
    pl.xlabel("$r / a$")
    pl.ylabel("$aV(r)$")

    return aV,handles

if __name__ == "__main__":

    pl.ion()

    folders = [f for f in listdir("results") if isdir(join("results",f))]

    folders.sort()
    
    if len(folders) == 0:
        print("No data available.")
        sys.exit()

    handles = []

    while True:
    
        print("Available data:")
        for i in xrange(len(folders)):
                print("(%d) %s" % (i,folders[i]))
        
        folder_num = input("Folder: ")
        folder = "results/%s" % folders[folder_num]
        try:
                Ws = load(join(folder,"Ws.npy"))
        except IOError:
                print("Warning! File Ws.npy does not exist.")
        try:
                Ps = load(join(folder,"Ps.npy"))
        except IOError:
                print("Warning! File Ps.npy does not exist.")
        
        print("Data loaded!")
        selection = 1
        while selection > 0:
				print("Please select an option:")
				print("(1) Plot the quark pair potential as a function of separation")
				print("(2) Plot the autocorrelation function")
				print("(3) Plot the evolution of the mean plaquette value")
				print("(4) Enter an IPython prompt")
				print("(5) Select different data")
				print("(0) Exit")
				selection = input("Option: ")
        
				if selection == 1:
					aV,hs = Vplot(Ws)
					handles += hs
				elif selection == 2:
					Cs = plotAutoCor(Ps)
				elif selection == 3:
					plotPs(Ps)
				elif selection == 4:
					IPython.embed()
				elif selection == 5:
					selection = -1
				else:
					sys.exit()
            

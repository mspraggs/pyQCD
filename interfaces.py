import numpy as np
import lib.pyQCD as pyQCD
try:
    from multiprocessing import Process, Array
except ImportError:
    import sys
    sys.path.append("/home/ms10g12/lib64/python")
    from multiprocessing import Process, Array

def getLinks(lattice):
    """Extracts links from lattice as a compound list of numpy arrays"""
    out = []
    for i in xrange(lattice.n):
        ilist = []
        for j in xrange(lattice.n):
            jlist = []
            for k in xrange(lattice.n):
                klist = []
                for l in xrange(lattice.n):
                    llist = []
                    for m in xrange(4):
                        llist.append(np.matrix(lattice.getLink(i,j,k,l,m)))
                    klist.append(llist)
                jlist.append(klist)
            ilist.append(jlist)
        out.append(ilist)

    return out

def calcW(lattice,out,r,t,n_smears,rmax):
    """The worker function"""
    out[(r-1)*(rmax-1)+t-1] = lattice.Wav(r,t,n_smears)


def calcWs(lattice,rmax,tmax,n_smears=0):
    """Calculates a series of Wilson loops up to the maximum r and t values"""
    out = [0.] * (tmax - 1) * (rmax - 1)
    outshare = Array('d',out)

    rts = [(r,t) for r in range(1,rmax) for t in range(1,tmax)]
    ps = []

    for r,t in rts:
        p = Process(target=calcW,args=(lattice,outshare,r,t,n_smears,rmax))
        p.start()
        ps.append(p)

    for p in ps:
        p.join()
    
    result = np.reshape(np.array(outshare[:]),(rmax-1,tmax-1))
    return result

if __name__ == "__main__":
	L = pyQCD.Lattice()
	Ws = calcWs(L,7,7,1)

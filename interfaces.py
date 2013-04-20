import numpy as np

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

def calcWs(lattice,rmax,tmax,n_smears=0):
    """Calculates a series of Wilson loops up to the maximum r and t values"""
    out = np.zeros((rmax-1,tmax-1))

    indices = [(i,j,k,l,m) for i in range(lattice.n) for j in range(lattice.n) for k in range(lattice.n) for l in range(lattice.n) for m in range(4)]

    for r in xrange(1,rmax):
        for t in xrange(1,tmax):
            for i,j,k,l,m in indices:
                out[r-1,t-1] += lattice.W([i,j,k,l],r,t,m,n_smears)
            
    return out / len(indices)

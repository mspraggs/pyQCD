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
                    for m in xrange(lattice.n):
                        llist.append(np.matrix(lattice.getLink(i,j,k,l,m)))
                    klist.append(llist)
                jlist.append(klist)
            ilist.append(jlist)
        out.append(ilist)

    return out

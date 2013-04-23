import lattice
import time

actions = range(3)
betas = [5.5,1.719,1.719]
u0s = [1,0.797,0.797]

for i in xrange(3):
    print("")    
    print("For action = %d, beta = %f and u0 = %f" % (actions[i],betas[i],u0s[i]))    
    L = lattice.Lattice(8,betas[i],50,1000,0.24,0.25,1./12,u0s[i],actions[i])
    print("Timing update function")
    t1 = time.time()
    L.update()
    t2 = time.time()
    print("Run time = %f" % (t2-t1))
    
    print("Timing 1000 5x5 Wilson loop functions")
    t1 = time.time()
    for i in xrange(1000): L.W([0,0,0,0],5,5,3)
    t2 = time.time()
    print("Run time = %f" % (t2-t1))
    
    print("Timing 1000 1x1 Wilson loop functions")
    t1 = time.time()
    for i in xrange(1000): L.W([0,0,0,0],1,1,3)
    t2 = time.time()
    print("Run time = %f" % (t2-t1))
    
    print("Timing 1000 plaquette functions")
    t1 = time.time()
    for i in xrange(1000): L.P([0,0,0,0],0,3)
    t2 = time.time()
    print("Run time = %f" % (t2-t1))

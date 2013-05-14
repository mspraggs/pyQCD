import pyQCD
import time

actions = range(3)
betas = [5.5,1.719,1.719]
u0s = [1,0.797,0.797]

for i in xrange(3):
    print("")    
    print("For action = %d, beta = %f and u0 = %f" % (actions[i],betas[i],u0s[i]))    
    L = pyQCD.Lattice(8,betas[i],u0s[i],actions[i],50,0.25,0.24)
    print("Timing update function")
    t1 = time.time()
    L.update()
    t2 = time.time()
    print("Run time = %f" % (t2-t1))
    
    print("Timing Schwarz update function")
    t1 = time.time()
    L.schwarz_update(4,1)
    t2 = time.time()
    print("Run time = %f" % (t2-t1))
    
    print("Timing 1000 5x5 Wilson loop functions")
    t1 = time.time()
    for i in xrange(1000): L.wilson_loop([0,0,0,0],5,5,3)
    t2 = time.time()
    print("Run time = %f" % (t2-t1))
    
    print("Timing 1000 1x1 Wilson loop functions")
    t1 = time.time()
    for i in xrange(1000): L.wilson_loop([0,0,0,0],1,1,3)
    t2 = time.time()
    print("Run time = %f" % (t2-t1))
    
    print("Timing 1000 plaquette functions")
    t1 = time.time()
    for i in xrange(1000): L.plaquette([0,0,0,0],0,3)
    t2 = time.time()
    print("Run time = %f" % (t2-t1))

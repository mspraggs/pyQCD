import lattice
import time

L = lattice.Lattice()
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

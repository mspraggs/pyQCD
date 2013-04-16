import lattice
import time

L = lattice.Lattice()
print("Timing update function")
t1 = time.time()
L.update()
t2 = time.time()
print("Run time = %f" % (t2-t1))
print("Timing 1000 Wilson loop functions")
t1 = time.time()
for i in xrange(1000): L.W([0,0,0,0],[5,0,0,5])
t2 = time.time()
print("Run time = %f" % (t2-t1))


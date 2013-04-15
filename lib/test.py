import lattice
import time

L = lattice.Lattice()
t1 = time.time()
L.update()
t2 = time.time()

print("Run time = %f" % (t2-t1))

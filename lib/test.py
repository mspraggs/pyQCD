import lattice
import time

L = lattice.Lattice()
print(L.Pav())
t1 = time.time()
L.update()
t2 = time.time()
print(L.Pav())

print("Run time = %f" % (t2-t1))

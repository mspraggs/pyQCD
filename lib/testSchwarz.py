import pyQCD
import time

L = pyQCD.Lattice()
t1 = time.time()
i1 = 0

while L.av_plaquette() > 0.5:
	L.schwarz_update(4,1)
	print(L.av_plaquette())
	i1 += 1

t2 = time.time()
L = pyQCD.Lattice()
t3 = time.time()
i2 = 0

while L.av_plaquette() > 0.5:
	L.update()
	print(L.av_plaquette())
	i2 += 1

t4 = time.time()

print("Schwarz convergence = %f seconds, %d iterations" % ((t2-t1), i1))
print("Standard convergence = %f seconds, %d iterations" % ((t4-t3), i2))

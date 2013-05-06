import pyQCD
import time

L = pyQCD.Lattice()
t1 = time.time()

while L.av_plaquette() > 0.5:
	L.schwarz_update(4,10)
	print(L.av_plaquette())

t2 = time.time()
L = pyQCD.Lattice()
t3 = time.time()

while L.av_plaquette() > 0.5:
	L.update()
	print(L.av_plaquette())

t4 = time.time()

print("Schwarz convergence = %f seconds" % (t2-t1))
print("Standard convergence = %f seconds" % (t4-t3))

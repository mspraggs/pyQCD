import pyQCD
import time

L = pyQCD.Lattice()
t1 = time.time()

while L.Pav() > 0.5:
	L.updateSchwarz(4,10)
	print(L.Pav())

t2 = time.time()
L = pyQCD.Lattice()
t3 = time.time()

while L.Pav() > 0.5:
	L.update()
	print(L.Pav())

t4 = time.time()

print("Schwarz convergence = %f seconds" % (t2-t1))
print("Standard convergence = %f seconds" % (t4-t3))

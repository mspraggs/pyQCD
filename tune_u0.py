
import lib.pyQCD as pyQCD
import sys
import pylab as pl
from optparse import OptionParser

def printConfig(options):
    """Outputs the simulation configuration to the screen"""
    print("Configuration:")
    print("n = %d" % options.n)
    print("beta = %f" % options.beta)
    print("Ncor = %d" % options.Ncor)
    print("Ncf = %d" % options.Ncf)
    print("eps = %f" % options.eps)
    print("a = %f" % options.a)
    print("rho = %f" % options.rho)
    print("n_smears = %d" % options.n_smears)
    print("u0 = %f" % options.u0)
    print("action = %d" % options.action)

parser = OptionParser()
parser.add_option("-b", "--beta", action = "store", type = "float",
				  dest = "beta", default = 5.5)
parser.add_option("-u", "--u0", action = "store", type = "float",
				  dest = "u0", default = 1)
parser.add_option("-a", "--action", action = "store", type = "int",
				  dest = "action", default = 0)
parser.add_option("--nsmears", action = "store", type = "int",
				  dest = "n_smears", default = 0)
parser.add_option("-n", "--n", action = "store", type = "int", dest = "n",
				  default = 4)
parser.add_option("--Ncor", action = "store", type = "int", dest = "Ncor",
				  default = 50)
parser.add_option("--Ncf", action = "store", type = "int", dest = "Ncf",
				  default = 1000)
parser.add_option("-e", "--eps", action = "store", type = "float",
				  dest = "eps", default = 0.24)
parser.add_option("-s", "--spacing", action = "store", type = "float",
				  dest = "a", default = 0.25)
parser.add_option("--rho", action = "store", type = "float",
				  dest = "rho", default = 0.3)

(options,args) = parser.parse_args()

printConfig(options)

u0 = options.u0
u0s = []

#Thermalize the lattice
for i in xrange(10):
	L = pyQCD.Lattice(options.n, #n
		options.beta, #beta
		options.u0, #u0
		options.action, #action			
		options.Ncor, #Ncor
		options.rho, #rho
		options.eps) #epsilon
		
	print("Iteration: %d" % i)
	print("Thermalizing...")
	sys.stdout.flush()
	L.thermalize()
	print("Done!")
	u0 = (L.av_plaquette()) ** 0.25
	if i > 1:
		u0s.append(u0)
	print("New u0 = %f" % u0)
        sys.stdout.flush()

print("Average new u0 (for iterations > 1) = %f" % (sum(u0s)/len(u0s)))
print("Standard deviation in new u0 (for iterations > 1) = %f" % pl.std(u0s))

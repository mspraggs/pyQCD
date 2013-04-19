BOOST = -I/usr/include/python2.7/
EIGEN = -I/usr/include/eigen3/

LBOOST = -lboost_python
LPYTHON = -lpython2.7

lattice.so : lattice.o
	g++ -g -shared -Wall -o lattice.so lattice.o $(LPYTHON) $(LBOOST)

lattice.o : lattice.cpp
	g++ -O3 -c -fPIC -o lattice.o lattice.cpp $(BOOST) $(EIGEN)

clean:
	rm lattice.o lattice.so	

BOOST_INCLUDE = -I/usr/include/python2.7/
EIGEN_INCLUDE = -I/usr/include/eigen3/

lattice.so : lattice.o
	g++ -g -shared -Wall -o lattice.so lattice.o -lpython2.7 -lboost_python

lattice.o : lattice.cpp
	g++ -O3 -c -fPIC -o lattice.o lattice.cpp $(BOOST_INCLUDE) $(EIGEN_INCLUDE)

clean:
	rm lattice.o lattice.so	

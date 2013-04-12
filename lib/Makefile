lattice.so : lattice.o
	g++ -g -shared -Wl -o lattice.so lattice.o -lpython2.7 -lboost_python

lattice.o : lattice.cpp
	g++ -O3 -c -g -fPIC -o lattice.o lattice.cpp -I/usr/include/python2.7/

clean:
	rm lattice.o lattice.so	

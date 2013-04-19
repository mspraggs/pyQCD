
ifeq "$(IRIDIS)" "true"
	PYTHON = -I/local/software/rh53/python/2.6.5/gcc/include/python2.6/
	BOOST = -I${BOOST_DIR}/include/
	EIGEN = -I/home/ms10g12/libs/Eigen/
	INC = $(EIGEN) $(PYTHON) $(BOOST)

	LBOOST = -dynamic-linker $(BOOST_DIR)/lib/libboost_python.so
	LPYTHON = -L/local/software/rh53/python/2.6.5/gcc/lib/
	
	LIBS = $(LBOOST) $(LPYTHON)
else
	EIGEN = -I/usr/include/eigen3/
	PYTHON = -I/usr/include/python2.7/
	INC = $(EIGEN) $(PYTHON)
	
	LBOOST = -lboost_python
	LPYTHON = -lpython2.7
	LIBS = $(LBOOST) $(LPYTHON)
endif

lattice.so : lattice.o
	g++ -g -shared -Wall -o lattice.so lattice.o $(LIBS)

lattice.o : lattice.cpp
	g++ -O3 -c -fPIC -o lattice.o lattice.cpp $(INC)

clean:
	rm lattice.o lattice.so	

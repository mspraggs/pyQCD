
ifeq "$(IRIDIS)" "true"
	PYTHON = -I/local/software/rh53/python/2.6.5/gcc/include/python2.6/
	BOOST = -I${BOOST_DIR}/include/
	EIGEN = -I/home/ms10g12/libs/Eigen/
	INC = $(EIGEN) $(PYTHON) $(BOOST)

	LBOOST = -dynamic-linker $(BOOST_DIR)/lib/libboost_python.so $(BOOST_DIR)/lib/libboost_thread.so
	LPYTHON = -L/local/software/rh53/python/2.6.5/gcc/lib/

	LIBS = $(LBOOST) $(LPYTHON)
else
	EIGEN = -I/usr/include/eigen3/
	PYTHON = -I/usr/include/python2.7/
	INC = $(EIGEN) $(PYTHON)

	LBOOST = -lboost_python -lboost_thread
	LPYTHON = -lpython2.7
	LIBS = $(LBOOST) $(LPYTHON)
endif

pyQCD.so : pyQCD.o
	g++ -g -fopenmp -shared -Wall -o pyQCD.so pyQCD.o $(LIBS)

pyQCD.o : pyQCD.cpp lattice.cpp
	g++ -O3 -fopenmp -c -fPIC -o pyQCD.o pyQCD.cpp $(INC)

clean:
	rm pyQCD.o pyQCD.so	

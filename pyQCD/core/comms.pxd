cdef extern from "mpi.h":
    cdef cppclass MPI_Comm:
        pass

cdef extern from "comms.hpp" namespace "pyQCD":
    cdef cppclass Communicator:
        @staticmethod
        Communicator& instance()
        @staticmethod
        void init(MPI_Comm& comm)
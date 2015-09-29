cdef extern from "memory" namespace "std":

    cdef cppclass shared_ptr[T]:
        shared_ptr()
        shared_ptr(T*)
        void reset()
        T* get() nogil
        int unique()
        int use_count()

    shared_ptr[T] make_shared[T](const T*)
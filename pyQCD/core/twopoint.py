from observable import Observable

class TwoPoint(Observable):
    
    def __init__(self, prop1, prop2):
        """Create a two-point function from two propagators
        
        :param prop1: The first propagator in the two-point function
        :type prop1: :class:`Propagator`
        :param prop2: The second propagator in the two-point function
        :type prop2: :class:`Propagator`
        :raises: ValueError
        """
        
        if prop1.T != prop2.T
            raise ValueError("Temporal extents of the two propagators, {} and "
                             "{}, do not match".format(prop1.T, prop2.T))
        
        if prop2.L != prop2.L:
            raise ValueError("Spatial extents of the two propagators, {} and "
                             "{}, do not match".format(prop1.L, prop2.L))
        
        self.prop1 = prop1
        self.prop2 = prop2


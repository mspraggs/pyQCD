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
        
        for member in TwoPoint.common_members:
            prop1_member = getattr(prop1, member)
            prop2_member = getattr(prop2, member)
            
            if prop1_member != prop2_member:
                raise ValueError("{} members in propagators 1 ({}) and "
                                 "2 ({}) do not match."
                                 .format(member, prop1_member, prop2_member))
        
        self.prop1 = prop1
        self.prop2 = prop2
        self.computed_correlators = []


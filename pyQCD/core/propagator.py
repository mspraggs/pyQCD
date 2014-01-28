import numpy as np
from observable import Observable

class Propagator(Observable):
    
    members = ['L', 'T', 'beta', 'u0', 'action', 'mass',
               'source_site', 'num_field_smears',
               'field_smearing_param', 'num_source_smears',
               'source_smearing_param', 'num_sink_smears',
               'sink_smearing_param']
        
    def __init__(self, propagator, L, T, beta, u0, action, mass, source_site,
                 num_field_smears, field_smearing_param, num_source_smears,
                 source_smearing_param, num_sink_smears, sink_smearing_param):
        """Create a propagator container.
                 
        Args:
            propagator (numpy.ndarray): The propagator data, with shape
              (T, L, L, L, 4, 4, 3, 3), the last four indices in the shape
              corresponding to spin and colour
            L (int): The spatial extent of the corresponding lattice
            T (int): The temporal extent of the corresponding lattice
            beta (float): The inverse coupling
            u0 (float): The mean link/tadpole coefficient
            action (str): The gauge action
            mass (float): The bare quark mass
            source_site (list): The source site to use when constructing the
              source for the inversion, of the form [t, x, y, z]
            num_field_smears (int): The number of stout field smears applied
              before doing the inversion
            field_smearing_param (float): The stout field smearing parameter to
              use before doing the inversion
            num_source_smears (int): The number of Jacobi smears to apply
              to the source before inverting.
            source_smearing_param (float): The Jacobi field smearing parameter to
              use before doing the inversion
            num_sink_smears (int): The number of Jacobi smears to apply
              to the sink before inverting.
            sink_smearing_param (float): The Jacobi field smearing parameter to
              use before doing the inversion
              
        Returns:
            Propagator: The propagator object
        
        Raises:
            ValueError: Shape of specified propagator array does not match the
              specified lattice extents.
              
        Examples:
            Create a dummy propagator array and use it to generate a Propagator
            object.
            
            >>> import pyQCD
            >>> import numpy
            >>> prop_data = numpy.zeros((8, 4, 4, 4, 4, 4, 3, 3))
            >>> prop = pyQCD.Propagator(prop_data, 4, 8, 5.5, 1.0, "wilson", 0.4,
            ...                         [0, 0, 0, 0], 0, 1.0, 0, 1.0, 0, 1.0)
            
            Ordinarily, one would generate a Propagator object from a given
            lattice, using the get_propagator member function.
        """
        
        expected_shape = (T, L, L, L, 4, 4, 3, 3)
        
        if propagator.shape != expected_shape:
            raise ValueError("Shape of specified propagator array, {}, does not "
                             "match the specified lattice extents, {}"
                             .format(propagator.shape, expected_shape))
        
        self.L = L
        self.T = T
        self.beta = beta
        self.u0 = u0
        self.action = action
        self.mass = mass
        self.source_site = source_site
        self.num_field_smears = num_field_smears
        self.field_smearing_param = field_smearing_param
        self.num_source_smears = num_source_smears
        self.source_smearing_param = source_smearing_param
        self.num_sink_smears = num_sink_smears
        self.sink_smearing_param = sink_smearing_param
        
        self.data = propagator
    
    def save(self, filename):
        """Saves the propagator to a numpy zip archive
        
        Args:
            filename (str): The name of the file in which to save the propagator
        
        Examples:
            Generate a propagator object from a lattice object and save it to
            disk.
            
            >>> import pyQCD
            >>> lattice = pyQCD.Lattice()
            >>> lattice.thermalize(100)
            >>> prop = lattice.get_propagator(0.4)
            >>> prop.save("myprop.npz")
        """
        Observable.save(self, filename)
        
    @classmethod
    def load(cls, filename):
        """Loads and returns a propagator object from a numpy zip
        archive
        
        Args:
            filename (str): The filename from which to load the observable
            
        Returns:
            Propagator: The loaded propagator object.
            
        Examples:
            Load a propagator object from disk
            
            >>> import pyQCD
            >>> prop = pyQCD.Propagator.load("myprop.npz")
        """
        return super(Propagator, cls).load(filename)
    
    def conjugate(self):
        """Returns the complex-conjugated propagator
        
        Returns:
            Propagator: The complex-conjugated propagator object
            
        Examples:
            Load a propagator from disk and compute the complex-conjugated
            propagator, before saving back to disk.
            
            >>> import pyQCD
            >>> prop = pyQCD.Propagator.load("myprop.npz")
            >>> prop_conjugated = prop.conjugate()
            >>> prop_conjugated.save("myprop_conjugated.npz")
        """
        
        new_data = np.conj(self.data)
        
        return Propagator(new_data, **self.header())
    
    def transpose_spin(self):
        """Returns the propagator with spin indices transposed
        
        Returns:
            Propagator: The spin-transposed propagator object
            
        Examples:
            Load a propagator from disk and use it to compute the correlation
            function at every site on the lattice. Here we use transpose spin,
            transpose colour and conjugation, along with gamma 5 hermiticity to
            compute the hermitian conjugate of the propagator
            
            >>> import numpy as np
            >>> import pyQCD
            >>> g5 = np.matrix(pyQCD.constants.gamma5)
            >>> g0 = np.matrix(pyQCD.constants.gamma0)
            >>> interpolator = g0 * g5
            >>> prop = pyQCD.Propagator.load("myprop.npz")
            >>> prop_herm = prop.transpose_spin().transpose_colour().conjugate()
            >>> prop_herm = g5 * prop_herm * g5
            >>> first_product = interpolator * prop_herm
            >>> second_product = interpolator * prop
            >>> correlator = np.einsum('txyzijab, txyzjiab->txyz',
            ...                        first_product.data, second_product.data)
        """
        
        new_data = np.swapaxes(self.data, 4, 5)
        
        return Propagator(new_data, **self.header())
    
    def transpose_colour(self):
        """Returns the propagator with colour indices transposed
        
        Returns:
            Propagator: The colour-transposed propagator object
            
        Examples:
            Load a propagator from disk and use it to compute the correlation
            function at every site on the lattice. Here we use transpose colour,
            transpose colour and conjugation, along with gamma 5 hermiticity to
            compute the hermitian conjugate of the propagator
            
            >>> import numpy as np
            >>> import pyQCD
            >>> g5 = np.matrix(pyQCD.constants.gamma5)
            >>> g0 = np.matrix(pyQCD.constants.gamma0)
            >>> interpolator = g0 * g5
            >>> prop = pyQCD.Propagator.load("myprop.npz")
            >>> prop_herm = \
            ...     prop.transpose_colour().transpose_colour().conjugate()
            >>> prop_herm = g5 * prop_herm * g5
            >>> first_product = interpolator * prop_herm
            >>> second_product = interpolator * prop
            >>> correlator = np.einsum('txyzijab, txyzjiab->txyz',
            ...                        first_product.data, second_product.data)
        """
        
        new_data = np.swapaxes(self.data, 6, 7)
        
        return Propagator(new_data, **self.header())
        
    def __mul__(self, matrix):
        
        if type(matrix) != np.matrixlib.defmatrix.matrix \
          and type(matrix) != float and type(matrix) != int:
            raise ValueError("Error: Propagator cannot be multiplied by type "
                             "{}".format(type(matrix)))
        
        properties = dict(zip(Propagator.members,
                              [getattr(self, m) for m in Propagator.members]))
        
        if type(matrix) == np.matrixlib.defmatrix.matrix:
            if matrix.shape == (4, 4): # Multiply by a spin matrix
                out = np.tensordot(self.data, matrix, (5, 0))
                out = np.transpose(out, (0, 1, 2, 3, 4, 7, 5, 6))
            
                return Propagator(out, **properties)
        
            elif matrix.shape == (3, 3): # Multiply by a colour matrix
                out = np.tensordot(self.data, matrix, (7, 0))
            
                return Propagator(out, **properties)
        
        elif type(matrix) == float or type(matrix) == int:
            out = self.data * matrix
            return Propagator(out, **properties)
            
        
    def __rmul__(self, matrix):
        
        if type(matrix) != np.matrixlib.defmatrix.matrix \
          and type(matrix) != float and type(matrix) != int:
            raise ValueError("Error: Propagator cannot multiply by type "
                             "{}".format(type(matrix)))
        
        properties = dict(zip(Propagator.members,
                              [getattr(self, m) for m in Propagator.members]))
        
        if type(matrix) == np.matrixlib.defmatrix.matrix:
            if matrix.shape == (4, 4):
                out = np.tensordot(matrix, self.data, (1, 4))
                out = np.transpose(out, (1, 2, 3, 4, 0, 5, 6, 7))
                
                return Propagator(out, **properties)
        
            elif matrix.shape == (3, 3):
                out = np.tensordot(matrix, self.data, (1, 6))
                out = np.transpose(out, (1, 2, 3, 4, 5, 6, 0, 7))
            
                return Propagator(out, **properties)
        
        elif type(matrix) == float or type(matrix) == int:
            out = self.data * matrix
            return Propagator(out, **properties)
        
    def __str__(self):
        
        out = \
          "Propagator Object\n" \
        "-----------------\n" \
        "Spatial extent: {}\n" \
        "Temportal extent: {}\n" \
        "Gauge action: {}\n" \
        "Inverse coupling (beta): {}\n" \
        "Mean link (u0): {}\n" \
        "Bare quark mass (m): {}\n" \
        "Inversion source site: {}\n" \
        "Number of stout field smears: {}\n" \
        "Stout smearing parameter: {}\n" \
        "Number of source Jacobi smears: {}\n" \
        "Source Jacobi smearing parameter: {}\n" \
        "Number of sink Jacobi smears: {}\n" \
        "Sink Jacobi smearing parameter: {}\n" \
        .format(self.L, self.T, self.action, self.beta,
                self.u0, self.mass, self.source_site,
                self.num_field_smears, self.field_smearing_param,
                self.num_source_smears, self.source_smearing_param,
                self.num_sink_smears, self.sink_smearing_param)
        
        return out

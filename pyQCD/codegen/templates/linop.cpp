#include <linear_operators/{{ include_name }}.hpp>

{{ class_name }}::{{ class_name }}({% for arg in ctor_args %}
  {{ arg }}, {% endfor %}
  const vector<complex<double> >& boundaryConditions,
  const Lattice* lattice) : LinearOperator::LinearOperator()
{
  // Class constructor - we set the fermion mass, create a pointer to the 
  // lattice and compute the frequently used spin structures used within the
  // Dirac operator.
  this->operatorSize_ 
    = 12 * int(pow(lattice->spatialExtent, 3)) * lattice->temporalExtent;
  this->lattice_ = lattice;

{{ ctor_body }}

  // These should be generated depending on whether there's a hopping matrix
  // available
{{ even_odd_handling }}
}



{{ class_name }}::~{{ class_name }}()
{
  // Need to determine which member_functions
  {% for member in destructibles %}
  delete {{ member }};
  {% endfor %}
}



VectorXcd {{ class_name }}::apply(const VectorXcd& {{ apply_arg }})
{
{{ apply_body }}
}



VectorXcd {{ class_name }}::applyHermitian(const VectorXcd& {{ apply_herm_arg }})
{
{{ apply_herm_body }}
}



VectorXcd {{ class_name }}::makeHermitian(const VectorXcd& {{ make_herm_arg }})
{
{{ make_herm_body }}
}



VectorXcd {{ class_name }}::applyEvenEvenInv(const VectorXcd& {{ apply_even_even_inv_arg }})
{
  // Invert the even diagonal piece
{{ apply_even_even_inv_body }}
}



VectorXcd {{ class_name }}::applyOddOdd(const VectorXcd& {{ apply_odd_odd_arg }})
{
  // Invert the even diagonal piece
{{ apply_odd_odd_body }}
}



VectorXcd {{ class_name }}::applyEvenOdd(const VectorXcd& {{ apply_even_odd_arg }})
{
{{ apply_even_odd_body }}
}



VectorXcd {{ class_name }}::applyOddEven(const VectorXcd& {{ apply_odd_even_arg }})
{
{{ apply_odd_even_body }}
}

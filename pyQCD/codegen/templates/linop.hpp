#ifndef {{ include_guard }}_HPP
#define {{ include_guard }}_HPP

#include <Eigen/Dense>

#include <complex>

#include <omp.h>

#include <lattice.hpp>
#include <utils.hpp>
#include <linear_operators/linear_operator.hpp>
#include <linear_operators/hopping_term.hpp>

using namespace Eigen;
using namespace std;

class {{ class_name }} : public LinearOperator
{

public:
  {{ class_name }}({% for arg in ctor_args %}{{ arg }},
    {% endfor %}
	 const vector<complex<double> >& boundaryConditions,
	 const Lattice* lattice);
  ~{{ class_name }}();

  VectorXcd apply(const VectorXcd& psi);
  VectorXcd applyHermitian(const VectorXcd& psi);
  VectorXcd makeHermitian(const VectorXcd& psi);

  VectorXcd applyEvenEvenInv(const VectorXcd& psi);
  VectorXcd applyOddOdd(const VectorXcd& psi);
  VectorXcd applyEvenOdd(const VectorXcd& psi);
  VectorXcd applyOddEven(const VectorXcd& psi);

private:
  // Member variables
  {% for var in member_vars %}
  {{ var }};
  {% endfor %}  
  const Lattice* lattice_;
  vector<vector<complex<double> > > boundaryConditions_;
};

#endif

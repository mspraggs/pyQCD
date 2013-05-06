#ifndef SCOPEDGILRELEASE_HPP
#define SCOPEDGILRELEASE_HPP
#include <boost/python.hpp>

using namespace boost::python;

class ScopedGILRelease
{

public:
  inline ScopedGILRelease()
  {
    m_thread_state = PyEval_SaveThread();
  }
  
  inline ~ScopedGILRelease()
  {
    PyEval_RestoreThread(m_thread_state);
    m_thread_state = NULL;
  }
  
private:
  PyThreadState * m_thread_state;
  
  ScopedGILRelease(const ScopedGILRelease& scoped);
  ScopedGILRelease& operator=(const ScopedGILRelease& scoped);
};

#endif

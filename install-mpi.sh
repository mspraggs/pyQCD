#!/bin/sh

MPI_IMPL="$1"

case "$MPI_IMPL" in
  openmpi16)
    if [ ! -d "$HOME/.local/$MPI_IMPL/bin" ]
    then
      echo "Installing OpenMPI 1.6.5"
      wget --no-check-certificate https://www.open-mpi.org/software/ompi/v1.6/downloads/openmpi-1.6.5.tar.gz
      tar xvf openmpi-1.6.5.tar.gz
      mkdir -p openmpi-1.6.5/build
      cd openmpi-1.6.5/build
      ../configure --prefix="$HOME/.local/$MPI_IMPL" && make && make install
    else
      echo "Using cached OpenMPI 1.6.5"
    fi
    ;;
  *)
    echo "Unknown MPI implementation"
esac

export PATH=$PATH:$HOME/.local/$MPI_IMPL/bin
export LIBRARY_PATH=$LIBRARY_PATH:$HOME/.local/$MPI_IMPL/lib
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$HOME/.local/$MPI_IMPL/lib

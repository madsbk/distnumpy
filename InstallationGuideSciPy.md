## Requirements ##
DistNumPy works together with SciPy version 0.7.1.

## Generic Installation ##
To install SciPy together with MPICH2 we need to tell the mpicc compiler not to use the C++ header:
```
CXX=mpic++ CC="mpicc -DMPICH_SKIP_MPICXX" python setup.py install --prefix /your/dir/
```
Using OpenMPI the flag is called `OMPI_SKIP_MPICXX`:
```
CXX=mpic++ CC="mpicc -DOMPI_SKIP_MPICXX" python setup.py install --prefix /your/dir/
```

## Cray XE6 (hopper.nersc.gov) ##
On Hopper the frontend node is not allowed to call the MPI API. Therefore, you have to use a clean NumPy installation (v 1.3) when compiling SciPy.
```
CXX=CC CC="cc -DMPICH_SKIP_MPICXX" python setup.py install --prefix /your/dir/
```
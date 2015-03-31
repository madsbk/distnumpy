## Download ##
svn checkout http://distnumpy.googlecode.com/svn/trunk/ distnumpy

## Requirements ##
Beside the requirements described in `INSTALL.txt`, you need a working MPI-2 installation and the MPI-2 installation must expose `mpicc` in `$PATH`.

All MPI-2 implementations should work. However, at the moment we recommend MPICH2, which is available [her](http://www.mcs.anl.gov/research/projects/mpich2/).
Note if using MPICH2 it should be compiled with "-fPIC".

It should also be possible to use Open MPI, in which case it should be configured and compiled with the "--disable-dlopen" flag:
```
./configure --disable-dlopen && make all install
```

## Generic Installation ##
To install DistNumPy run the following command:
```
CC=mpicc python setup.py install --prefix /your/dir/
```

And then make sure that `/your/dir/` is in your `$PYTHONPATH`.

### OpenMP ###
DistNumPy can make use of [OpenMP](http://openmp.org) to utilize shared memory parallelism. To enable you compiler must support OpenMP - using GCC it looks like this:
```
CFLAGS=-fopenmp LDFLAGS=-lgomp CC=mpicc python setup.py install --prefix /your/dir/
```
NB: the macro `_OPENMP` must be defined, which must compiler automatically do when compiling with the OpenMP flag.

## Blue Gene/P (surveyor.alcf.anl.gov) ##
To install DistNumPy run the following command:
```
CC="/bgsys/drivers/ppcfloor/comm/bin/mpicc -fno-strict-aliasing " /bgsys/drivers/ppcfloor/gnu-linux/bin/python setup.py install --prefix /your/dir/
```
And then submit like:
```
qsub -n 2 -t 30 --mode smp --env PYTHONPATH=/your/dir/ /bgsys/drivers/ppcfloor/gnu-linux/bin/python your_program.py
```


## Cray XE6 (hopper.nersc.gov) ##
Compile Python:
```
module unload xt-shmem
export XTPE_LINK_TYPE=dynamic
./configure CC=cc CXX=cc --prefix=/your/dir/ LINKFORSHARED=-Wl,--export-dynamic
make all install
```

Compile DistNumPy using the compiled Python:
```
module unload xt-shmem
export XTPE_LINK_TYPE=dynamic
CFLAGS=-D_OPENMP CC=cc ~/Python2.6/bin/python dnumpy/setup.py install --prefix=/your/dir/
```

A job file could look like this:
```
#!/bin/bash -l
#PBS -q debug
#PBS -l mppwidth=24
#PBS -l walltime=00:10:00
#PBS -N test
#PBS -j oe

cd $PBS_O_HOME
module unload xt-shmem
export CRAY_ROOTFS=DSL

aprun -n 2 Python-2.6/install/bin/python distnumpy/test/dnumpytest.py -e test_IO.py
```

## Optimized BLAS ##
In order to use a vendor optimized BLAS library, such as MKL, ACL and SCI, you have to compile CBLAS, which is are C wrapper for BLAS.

First, download the CBLAS source from netlib:
```
wget http://www.netlib.org/blas/blast-forum/cblas.tgz
tar -xzf cblas.tgz
```
Change to the CBLAS directory and copy Makefile.LINUX to Makefile.in. Add correct compiler commands and paths to Makefile.in:
```
...
#-----------------------------------------------------------------------------
# Libraries and includs
#-----------------------------------------------------------------------------

BLLIB = libblas.a
CBLIB = ../lib/libblas.a

#-----------------------------------------------------------------------------
# Compilers
#-----------------------------------------------------------------------------

CC = cc
FC = ftn
LOADER = $(FC)

#-----------------------------------------------------------------------------
# Flags for Compilers
#-----------------------------------------------------------------------------

CFLAGS = -O3 -DADD_ -fPIC
FFLAGS = -O3 -fPIC
...
```
Finally, build CBLAS:
```
make
```
You are now ready to build DistNumPy with the newly created CBLAS wrapper. The standard Numpy tries to use only the ATLAS BLAS, and in order to use different BLAS one has to manually edit the file `numpy/core/setup.py`. Comment out an if statement as follows:
```
def get_dotblas_sources(ext, build_dir):
  if blas_info:
      # if ('NO_ATLAS_INFO',1) in blas_info.get('define_macros',[]):
      #     return None # dotblas needs ATLAS, Fortran compiled blas will not be sufficient.
      return ext.depends[:1]
```
Then, add the correct libraries and paths to the file `site.cfg`:
```
[blas]
blas_libs = blas
library_dirs = /global/homes/m/madsbk/CBLAS/lib

[lapack]
lapack_libs = sci
library_dirs = /opt/xt-libsci/10.5.01/pgi/lib
```

Now, one should be able to build DistNumPy as usual.
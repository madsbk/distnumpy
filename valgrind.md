[Valgrind](http://valgrind.org/) is a programming tool for memory debugging, memory leak detection, and profiling.

Before committing to trunk make sure that valgrind does not return any memory errors. However, both Python and NumPy returns some memory errors - it is therefore necessary to use a debug and valgrind friendly version of Python and DistNumPy.

1)Compile a debug and valgrind friendly version of Python.
```
./configure --with-pydebug --without-pymalloc --prefix /opt/debugpython/
make OPT=-g
make install
```

2)Compile a debug and valgrind friendly version of DistNumPy.
```
CC=mpicc /opt/debugpython/bin/python setup.py build
```

3)Run DistNumPy with valgrind and use the suppression file in the misc dir.
```
valgrind --suppressions=misc/valgrind.supp /opt/debugpython/bin/python test.py
```
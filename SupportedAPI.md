# Fully supported Distributed Arrays API in NumPy #

## C API ##
|PyUFunc\_GenericFunction|
|:-----------------------|
|PyUFunc\_Reduce|
|PyArray\_Flatten (returns a non-distributed array)|
|PyArray\_Ravel (returns a non-distributed array)|
|PyArray\_NewCopy|
|PyArray\_Empty|
|PyArray\_Zeros|
|PyArray\_Diagonal (Only complete views and axises must be default)|


## Python API ##
|array|
|:----|
|empty|
|zeros|
|[ufunc](http://docs.scipy.org/doc/numpy/reference/ufuncs.html)|
|[ufunc.reduce](http://docs.scipy.org/doc/numpy/reference/generated/numpy.ufunc.reduce.html) (Only complete views)|
|print|
|ravel (returns a non-distributed array)|
|flatten (returns a non-distributed array)|
|diagonal (Only complete views and axises must be default)|
|save|
|load|

# NumPy API not supported #

## C API ##
|PyArray\_Newshape|
|:----------------|


## Python API ##
|any()|
|:----|
|all()|
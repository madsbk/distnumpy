#include "Python.h"
#include "numpy/noprefix.h"
#include "structmember.h"
#include "numpy/arrayobject.h"
#include "hashdescr.c"

static PyObject *ErrorObject;
#define Py_Try(BOOLEAN) {if (!(BOOLEAN)) goto fail;}
#define Py_Assert(BOOLEAN,MESS) {if (!(BOOLEAN)) {      \
            PyErr_SetString(ErrorObject, (MESS));       \
            goto fail;}                                 \
    }

#define PYSETERROR(message) \
{ PyErr_SetString(ErrorObject, message); goto fail; }

static PyObject *
distio_load(PyObject *NPY_UNUSED(ignored), PyObject *args, PyObject *kwds)
{
    /* DISTNUMPY */
    static char *kwlist[] = {"filename","datapos","shape","order","dtype",NULL}; /* XXX ? */
    char *filename;
    long datapos;
    PyArray_Descr *typecode = NULL;
    PyArray_Dims shape = {NULL, 0};
    NPY_ORDER order = PyArray_CORDER;
    PyObject *ret = NULL;
    int flags = DNPY_DISTRIBUTED;

    if (!PyArg_ParseTupleAndKeywords(args, kwds, "slO&O&O&",
                                     kwlist, &filename, &datapos, 
                                     PyArray_IntpConverter,
                                     &shape,
                                     PyArray_OrderConverter,
                                     &order,
                                     PyArray_DescrConverter,
                                     &typecode)) {
        printf("Failed!\n");
        goto fail;
    }

    if (order == PyArray_FORTRANORDER) {
        flags |= NPY_FORTRAN;
    }

    if (!typecode) {
        typecode = PyArray_DescrFromType(PyArray_DEFAULT);
    }

    ret = (PyArrayObject *)PyArray_NewFromDescr(&PyArray_Type,
                                                typecode,
                                                shape.len, shape.ptr,
                                                NULL, NULL,
                                                flags, NULL);

    if (ret == NULL) {
        return NULL;
    }

    dnumpy_datafill(PyArray_DNDUID(ret), filename, datapos);

    PyDimMem_FREE(shape.ptr);
    return ret;

 fail:
    Py_XDECREF(typecode);
    PyDimMem_FREE(shape.ptr);
    return ret;
}


static struct PyMethodDef methods[] = {
    {"dist_load",  (PyCFunction)distio_load, METH_VARARGS | METH_KEYWORDS, 
     "Load binary file into distributed array."},
    {NULL, NULL}    /* sentinel */
};


/* Initialization function for the module (*must* be called init<name>) */

PyMODINIT_FUNC init_distio(void) {
    PyObject *m;

    /* Create the module and add the functions */
    m = Py_InitModule("_distio", methods);

    import_array();

    return;
}

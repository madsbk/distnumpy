#ifndef DISTNUMPY_H
#define DISTNUMPY_H

//#define DISTNUMPY_DEBUG

//Easy distribute check
#define CHECK_DISTRIBUTED(x) (((PyArrayObject *)x)->flags \
                                & NPY_DISTRIBUTED)

//Easy retrieval of dnduid
#define PyArray_DNDUID(obj) (((PyArrayObject *)(obj))->dnduid)                                

//Maximum message size (in bytes)
#define DNPY_MAX_MSG_SIZE 1024

//Maximum number of allocated arrays
#define DNPY_MAX_NARRAYS 1024

//Default blocksize
#define DNPY_BLOCKSIZE 1

//Operation types
enum opt {DNPY_MSG_END, DNPY_CREATE_ARRAY, DNPY_DESTROY_ARRAY,
          DNPY_CREATE_VIEW, DNPY_SHUTDOWN, DNPY_SET_ITEM, DNPY_GET_ITEM,
          DNPY_UFUNC, DNPY_UFUNC_REDUCE, DNPY_ZEROFILL};

//dndslice constants.
#define PseudoIndex -1//Adds a extra 1-dim - 'A[1,newaxis]'
#define RubberIndex -2//A[1,2,...]
#define SingleIndex -3//Dim not visible - 'A[1]'

//Type describing a distributed array
typedef struct
{
    //Reference count
    int refcount;
    //Number of dimensions
    int ndims;
    //Size of dimensions
    int dims[NPY_MAXDIMS];
    //Blocksize in every dimension
    int blocksize[NPY_MAXDIMS];    
    //Data type of elements in array
    int dtype;
    //Size of an element in bytes
    int elsize;
    //Pointer to local data
    char *data;
    //Number of local elements
    int nelements;
    //Size of local dimensions
    int localdims[NPY_MAXDIMS];
} dndarray;

//Type describing a slice of a dimension.
typedef struct
{
    //Start index
    int start;
    //Elements between index
    int step;
    //Number of steps (Length of the dimension)
    int nsteps; 
} dndslice;

//View-alteration flags
#define DNPY_NDIMS    0x001
#define DNPY_STEP     0x002
#define DNPY_NSTEPS   0x004

//Type describing a view of a distributed array
typedef struct
{
    //Unique identification.
    int uid;
    //The array this view is a view of.
    dndarray *base;
    //Number of sliceses. NB: nslice >= base->ndims.
    int nslice;
    //Sliceses.
    dndslice slice[NPY_MAXDIMS];
    //Number of viewable dimensions.
    int ndims;
    //A binary mask specifying which alterations this view represents.
    //Possible flags:
    //Zeros       - no alterations.
    //DNPY_NDIMS  - number of dimensions altered.
    //DNPY_STEP   - 'step' altered.
    //DNPY_NSTEPS - 'nsteps' altered.
    int alterations;
} dndview;

//Type describing a slice of a dimension.
typedef struct
{
    //Start index
    int start;
    //End index
    int end;
    //Elements between index
    int step;
} dslice;

#endif

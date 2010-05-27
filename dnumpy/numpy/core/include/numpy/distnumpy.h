#ifndef DISTNUMPY_H
#define DISTNUMPY_H
#include "mpi.h"

//#define DISTNUMPY_DEBUG

//Easy retrieval of dnduid
#define PyArray_DNDUID(obj) (((PyArrayObject *)(obj))->dnduid)

//Maximum message size (in bytes)
#define DNPY_MAX_MSG_SIZE 1024*10

//Maximum number of allocated arrays
#define DNPY_MAX_NARRAYS 1024

//Default blocksize
#define DNPY_BLOCKSIZE 2

#ifdef HAS_PROFILING
#define DNPY_PROFILING 0
#endif

//Operation types
enum opt {DNPY_MSG_END, DNPY_CREATE_ARRAY, DNPY_DESTROY_ARRAY,
          DNPY_CREATE_VIEW, DNPY_SHUTDOWN, DNPY_PUT_ITEM, DNPY_GET_ITEM,
          DNPY_UFUNC, DNPY_UFUNC_REDUCE, DNPY_ZEROFILL, DNPY_DATAFILL,
          DNPY_DATADUMP, DNPY_INIT_BLOCKSIZE, DNPY_INIT_PROFILING,
          DNPY_DIAGONAL, DNPY_MATMUL};

//Type describing a distributed array.
typedef struct
{
    //Reference count.
    int refcount;
    //Number of dimensions.
    int ndims;
    //Size of dimensions.
    npy_intp dims[NPY_MAXDIMS];
    //Data type of elements in array.
    int dtype;
    //Size of an element in bytes.
    int elsize;
    //Pointer to local data.
    char *data;
    //Number of local elements (local to the MPI-process).
    npy_intp nelements;
    //Size of local dimensions (local to the MPI-process).
    npy_intp localdims[NPY_MAXDIMS];
    //Size of local block-dimensions (local to the MPI-process).
    npy_intp localblockdims[NPY_MAXDIMS];
    //One-sided communication window (used by MPI_Get and MPI_Put).
    MPI_Win comm_win;
    //MPI-datatype that correspond to an array element.
    MPI_Datatype mpi_dtype;
} dndarray;

//dndslice constants.
#define PseudoIndex -1//Adds a extra 1-dim - 'A[1,newaxis]'
#define RubberIndex -2//A[1,2,...] (Not used in distnumpy.inc)
#define SingleIndex -3//Dim not visible - 'A[1]'

//Type describing a slice of a dimension.
typedef struct
{
    //Start index.
    npy_intp start;
    //Elements between index.
    npy_intp step;
    //Number of steps (Length of the dimension).
    npy_intp nsteps;
} dndslice;

//Type describing a sub-section of a view block.
typedef struct
{
    //The rank of the MPI-process that owns this sub-block.
    int rank;
    //Start index (one per base-dimension).
    npy_intp start[NPY_MAXDIMS];
    //Number of elements (one per base-dimension).
    npy_intp nsteps[NPY_MAXDIMS];
    //Number of elements to next dimension (one per base-dimension).
    npy_intp stride[NPY_MAXDIMS];

    //The MPI datatype for the view.
    MPI_Datatype comm_dtype_view;
    npy_intp offset_view;
    //The MPI datatype and offset for the buffer (in bytes).
    MPI_Datatype comm_dtype_buf;
    npy_intp offset_buf;

    //Number of elements in this sub-block.
    npy_intp nelem;
    //Pointer to data. NULL if data needs to be fetched. 
    char *data;
} dndsvb;

//Type describing a view block.
typedef struct
{
    //The id of the view block.
    npy_intp id;
    //All sub-view-blocks in this view block (Row-major).
    dndsvb *sub;
    //Number of sub-view-blocks.
    npy_intp nsub;
    //Number of sub-view-blocks in each dimension.
    npy_intp svbdims[NPY_MAXDIMS];
} dndvb;

//View-alteration flags.
#define DNPY_NDIMS    0x001
#define DNPY_STEP     0x002
#define DNPY_NSTEPS   0x004

//Type describing a view of a distributed array.
typedef struct
{
    //Unique identification.
    npy_intp uid;
    //The array this view is a view of.
    dndarray *base;
    //Number of viewable dimensions.
    int ndims;
    //Number of sliceses. NB: nslice >= base->ndims.
    int nslice;
    //Sliceses - the global view of the base-array.
    dndslice slice[NPY_MAXDIMS];
    //A bit mask specifying which alterations this view represents.
    //Possible flags:
    //Zero        - no alterations.
    //DNPY_NDIMS  - number of dimensions altered.
    //DNPY_STEP   - 'step' altered.
    //DNPY_NSTEPS - 'nsteps' altered.
    int alterations;
    //Number of view-blocks.
    npy_intp nblocks;
    //Number of view-blocks in each viewable dimension.
    npy_intp blockdims[NPY_MAXDIMS];
} dndview;

#endif

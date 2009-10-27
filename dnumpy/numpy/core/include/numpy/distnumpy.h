#ifndef DISTNUMPY_H
#define DISTNUMPY_H

//#define DISTNUMPY_DEBUG

//Easy retrieval of dnduid
#define PyArray_DNDUID(obj) (((PyArrayObject *)(obj))->dnduid)                                

//Maximum message size (in bytes)
#define DNPY_MAX_MSG_SIZE 1024*10

//Maximum number of allocated arrays
#define DNPY_MAX_NARRAYS 1024

//Default blocksize
#define DNPY_BLOCKSIZE 2

//Operation types
enum opt {DNPY_MSG_END, DNPY_CREATE_ARRAY, DNPY_DESTROY_ARRAY,
          DNPY_CREATE_VIEW, DNPY_SHUTDOWN, DNPY_SET_ITEM, DNPY_GET_ITEM,
          DNPY_UFUNC, DNPY_UFUNC_REDUCE, DNPY_ZEROFILL,
          DNPY_INIT_BLOCKSIZE, DNPY_DIAGONAL};

//dndslice constants.
#define PseudoIndex -1//Adds a extra 1-dim - 'A[1,newaxis]'
#define RubberIndex -2//A[1,2,...]
#define SingleIndex -3//Dim not visible - 'A[1]'

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
    //Number of local blocks (local to the MPI-process).
    //npy_intp nblocks;
    //Size of local block-dimensions (local to the MPI-process).
    npy_intp localblockdims[NPY_MAXDIMS];
    
} dndarray;

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
    //Number of sliceses. NB: nslice >= base->ndims.
    int nslice;
    //Sliceses - the global view of the base-array.
    dndslice slice[NPY_MAXDIMS];
    //Block slice - sliceses that indicate the viewable blocks in this
    //view (local to the MPI-process).
    //NB: number of bsliceses is always base->ndims.
//    dndslice bslice[NPY_MAXDIMS];
    //Number of viewable dimensions.
    int ndims;
    //A binary mask specifying which alterations this view represents.
    //Possible flags:
    //Zero        - no alterations.
    //DNPY_NDIMS  - number of dimensions altered.
    //DNPY_STEP   - 'step' altered.
    //DNPY_NSTEPS - 'nsteps' altered.
    int alterations;
} dndview;

#endif

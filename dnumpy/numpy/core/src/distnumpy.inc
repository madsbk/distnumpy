#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <math.h>

#ifndef CBLAS_HEADER
#define CBLAS_HEADER "../blasdot/cblas.h"
#endif
#include CBLAS_HEADER

#ifdef DNPY_TIME
#include <sys/time.h>
struct timeval tv;
struct timezone tz;
unsigned long long total_rget_time=0, tmp_time, total_time_tmp, total_time, total_apply_time;
#endif

/*===================================================================
 *
 * ufunc definitions from numpy/ufuncobject.h.
 * They are include here to avoid compiler warnings.
*/
typedef void (*PyUFuncGenericFunction)
             (char **, npy_intp *, npy_intp *, void *);
typedef struct {
    PyObject_HEAD
    int nin, nout, nargs;
    int identity;
    PyUFuncGenericFunction *functions;
    void **data;
    int ntypes;
    int check_return;
    char *name, *types;
    char *doc;
    void *ptr;
    PyObject *obj;
    PyObject *userloops;
    int core_enabled;
    int core_num_dim_ix;
    int *core_num_dims;
    int *core_dim_ixs;
    int *core_offsets;
    char *core_signature;
} PyUFuncObject;
/* end of ufunc definitions */

/*===================================================================
 *
 * MPI process variables.
*/
static int myrank, worldsize;
static npy_intp blocksize;
static npy_intp msg[DNPY_MAX_MSG_SIZE];
//Unique identification counter
static npy_intp uid_count;
//Array-views belonging to local MPI process
static dndview dndviews[DNPY_MAX_NARRAYS];
static npy_intp dndviews_uid[DNPY_MAX_NARRAYS];
//Cartesian dimension information - one for every dimension-order.
static int *cart_dim_strides[NPY_MAXDIMS];
static int *cart_dim_sizes[NPY_MAXDIMS];
static int cart_used_ndims[NPY_MAXDIMS];
//Pointer to the python module who has the ufunc operators.
static PyObject *ufunc_module;

//Pointers to the cblas functions.
typedef void (cblas_func_type)(const enum CBLAS_ORDER Order,...);
static cblas_func_type *cblas_dgemm_func = NULL;
static cblas_func_type *cblas_sgemm_func = NULL;
static cblas_func_type *cblas_zgemm_func = NULL;
static cblas_func_type *cblas_cgemm_func = NULL;


/*===================================================================
 *
 * Returns a string describing dtype.
 * Private
 */
#ifdef DISTNUMPY_DEBUG
static char *types_numpy2str(int dtype)
{
    switch(dtype)
    {
        case NPY_BOOL:
            return "NPY_BOOL";
        case NPY_BYTE:
            return "NPY_BYTE";
        case NPY_UBYTE:
            return "NPY_UBYTE";
        case NPY_SHORT:
            return "NPY_SHORT";
        case NPY_USHORT:
            return "NPY_USHORT";
        case NPY_INT:
            return "NPY_INT";
        case NPY_UINT:
            return "NPY_UINT";
        case NPY_LONG:
            return "NPY_LONG";
        case NPY_ULONG:
            return "NPY_ULONG";
        case NPY_LONGLONG:
            return "NPY_LONGLONG";
        case NPY_ULONGLONG:
            return "NPY_ULONGLONG";
        case NPY_FLOAT:
            return "NPY_FLOAT";
        case NPY_DOUBLE:
            return "NPY_DOUBLE";
        case NPY_LONGDOUBLE:
            return "NPY_LONGDOUBLE";
        case NPY_CFLOAT:
            return "NPY_CFLOAT";
        case NPY_CDOUBLE:
            return "NPY_CDOUBLE";
        case NPY_CLONGDOUBLE:
            return "NPY_CLONGDOUBLE";
        case NPY_OBJECT:
            return "NPY_OBJECT";
        case NPY_STRING:
            return "NPY_STRING";
        case NPY_UNICODE:
            return "NPY_UNICODE";
        case NPY_VOID:
            return "NPY_VOID";
        case NPY_NTYPES:
            return "NPY_NTYPES";
        case NPY_CHAR:
            return "NPY_CHAR";
        case NPY_USERDEF:
            return "NPY_USERDEF";
        default:
            return "\"Unknown data type\"";
    }
} /* types_numpy2str */
#endif

/*===================================================================
 *
 * Computes the number of elements in a dimension of a distributed
 * array owned by the MPI-process indicated by proc_dim_rank.
 * From Fortran source: http://www.cs.umu.se/~dacke/ngssc/numroc.f
 * Private
*/
static npy_intp dnumroc(npy_intp nelem_in_dim, npy_intp block_size,
                        int proc_dim_rank, int nproc_in_dim)
{
    int first_process = 0; //Assume array begins at process 0.

    //Figure process's distance from source process.
    int mydist = (nproc_in_dim + proc_dim_rank - first_process) %
                  nproc_in_dim;

    //Figure the total number of whole NB blocks N is split up into.
    npy_intp nblocks = nelem_in_dim / block_size;

    //Figure the minimum number of elements a process can have.
    npy_intp numroc = nblocks / nproc_in_dim * block_size;

    //See if there are any extra blocks
    npy_intp extrablocks = nblocks % nproc_in_dim;

    //If I have an extra block.
    if(mydist < extrablocks)
        numroc += block_size;

    //If I have last block, it may be a partial block.
    else if(mydist == extrablocks)
        numroc += nelem_in_dim % block_size;

    return numroc;
} /* dnumroc */

/*===================================================================
 *
 * Process cartesian coords <-> MPI rank.
 * Private.
 */
static int cart2rank(int ndims, const int coords[NPY_MAXDIMS])
{
    int *strides = cart_dim_strides[ndims-1];
    int rank = 0;
    int i;
    for(i=0; i<cart_used_ndims[ndims-1]; i++)
        rank += coords[i] * strides[i];
    return rank;
}
static void rank2cart(int ndims, int rank, int coords[NPY_MAXDIMS])
{
    int i;
    int *strides = cart_dim_strides[ndims-1];
    memset(coords, 0, ndims*sizeof(int));
    for(i=0; i<cart_used_ndims[ndims-1]; i++)
    {
        coords[i] = rank / strides[i];
        rank = rank % strides[i];
    }
} /* cart2rank & rank2cart */

/*===================================================================
 *
 * Put, get & remove views from local MPI process.
 * Private.
 */
static dndview *get_dndview(npy_intp uid)
{
    npy_intp i;
    if(uid > 0)
        for(i=0; i < DNPY_MAX_NARRAYS; i++)
            if(dndviews_uid[i] == uid)
                return &dndviews[i];
    fprintf(stderr, "get_dndview, uid %ld does not exist\n", (long) uid);
    MPI_Abort(MPI_COMM_WORLD, -1);
    return NULL;
}
static void put_dndview(dndview *view)
{
    npy_intp i;
    for(i=0; i < DNPY_MAX_NARRAYS; i++)
        if(dndviews_uid[i] == 0)
        {
            memcpy(&dndviews[i], view, sizeof(dndview));
            dndviews_uid[i] = view->uid;
            return;
        }
    fprintf(stderr, "put_dndarray, MAX_NARRAYS is exceeded\n");
    MPI_Abort(MPI_COMM_WORLD, -1);
}

//NB: rm_dndview will also free memory allocted for the dndarray
//if it is the last reference to the dndarray.
static void rm_dndview(npy_intp uid)
{
    npy_intp i;
    if(uid)
        for(i=0; i < DNPY_MAX_NARRAYS; i++)
            if(dndviews_uid[i] == uid)
            {
                dndviews_uid[i] = 0;
                if(--dndviews[i].base->refcount == 0)
                {
                    MPI_Win_free(&dndviews[i].base->comm_win);
                    MPI_Type_free(&dndviews[i].base->mpi_dtype);
                    MPI_Free_mem(dndviews[i].base->data);
                    free(dndviews[i].base);
                }
                return;
            }
    fprintf(stderr, "rm_dndarray, uid %ld does not exist\n", (long)uid);
    MPI_Abort(MPI_COMM_WORLD, -1);
    return;
}/* Put, get & rm dndview */


/*===================================================================
 *
 * Sends a message to all slaves.
 * msgsize is in bytes.
 * Private.
 */
static void msg2slaves(npy_intp *msg, int msgsize)
{
    if(msgsize > DNPY_MAX_MSG_SIZE)
    {
        fprintf(stderr, "msg2slaves, the messages is greater "
                        "than DNPY_MAX_MSG_SIZE\n");
        MPI_Abort(MPI_COMM_WORLD, -1);
    }

    MPI_Barrier(MPI_COMM_WORLD);
    MPI_Bcast(msg, DNPY_MAX_MSG_SIZE, MPI_BYTE, 0, MPI_COMM_WORLD);

} /* msg2slaves */




/*===================================================================
 *
 * Converts a global coordinate based on a view to a global coordinate
 * based on the view's base.
 * 'coord_based_on_view' and 'coord_based_on_base' is allowed to be
 * the same adresse, but note that there size may differ.
 * 'coord_based_on_base' size must be the same as view->base->ndims and
 * 'coord_based_on_view' size must be the same as view->ndims.
 * The coordinates are in Row-major.
 * Private.
 */
static void
view2base_globalcoord(npy_intp coord_based_on_view[NPY_MAXDIMS],
                      dndview* view,
                      npy_intp coord_based_on_base[NPY_MAXDIMS])
{
    int viewi = 0;
    int basei = 0;
    int i;
    for(i=0; i<view->nslice; i++)
    {
        if(view->slice[i].nsteps == SingleIndex)
        {
            coord_based_on_base[basei++] = view->slice[i].start;
        }
        else if(view->slice[i].nsteps == PseudoIndex)
        {
            if(coord_based_on_view[viewi] != 0)
            {
                fprintf(stderr, "view2base_globalcoord - coord_based_on"
                                "_view[%d] should be zero (was %d) sinc"
                                "e that dimension is a PseudoIndex\n",
                                viewi, (int)coord_based_on_view[viewi]);
                MPI_Abort(MPI_COMM_WORLD, -1);
            }
            viewi++;
        }
        else
        {
            coord_based_on_base[basei++] = view->slice[i].start +
                                           coord_based_on_view[viewi++]
                                           * view->slice[i].step;
        }
    }
}/* view2base_globalcoord */


/*===================================================================
 *
 * Find the next coordinate thats requires are remote-get.
 * Private.
 */
static void
find_next_get_coords(npy_intp coords[NPY_MAXDIMS],
                     const dndview* currentary,
                     const dndview* lastary,
                     const int dim_iter_order[NPY_MAXDIMS])
{
    //Iterate until we find a coord which needs a remote-get.
    int use_prev_block = 1;
    int i;
    while(use_prev_block)
    {
        //Iterate coords one block.
        for(i=0; i < lastary->base->ndims; i++)
        {
            int o = dim_iter_order[i];

            if(++coords[o] >= lastary->base->localblockdims[o])
            {
                //We a finished, if wrapping around.
                if(i == lastary->base->ndims-1)
                    return;
                coords[o] = 0;
            }
            else
                break;
        }

        use_prev_block = 0;
        for(i=lastary->ndims-1; 0<=i; i--)
        {
            if(coords[i] > 0 && lastary->base->dims[i] != 1)
            {
                int lastcount = i - (lastary->ndims - currentary->ndims);
                int s, si=currentary->ndims-1, gotdim=0;
                if(lastcount >= 0)
                {
                    for(s=currentary->nslice-1; s>=0; s--)
                    {
                        if(currentary->slice[s].nsteps != SingleIndex)
                            if(si-- == lastcount)
                            {
                                gotdim = 1;
                                break;
                            }
                    }
                }
                if(gotdim == 0 || //no more dims or
                                  //dim size is 1.
                   currentary->slice[s].nsteps == 1)
                {
                    use_prev_block = 1;
                    break;
                }
            }
        }
    }
} /* find_next_get_coords */

/*===================================================================
 *
 * Apply the ufunc 'function' on 'arys' and lastary.
 * 'narys' are both in and output arrays.
 * Private.
 */
static void
apply_ufunc(int narys, PyObject *arys[NPY_MAXARGS],
            PyUFuncGenericFunction function,
            void *funcdata,
            dndview *lastary,
            const npy_intp coords[NPY_MAXDIMS])
{
    //Compute a block's dims for the dest array.
    npy_intp bdims[NPY_MAXDIMS];
    npy_intp strides[NPY_MAXDIMS];
    npy_intp offset = 0;
    int j;

    for(j=0; j < lastary->base->ndims; j++)
    {
        npy_intp stride = lastary->base->elsize;
        int s;
        for(s=j+1; s<lastary->base->ndims; s++)
            stride *= lastary->base->localdims[s];
        strides[j] = stride;

        bdims[j] = lastary->base->localdims[j] - coords[j] * blocksize;
        if(bdims[j] > blocksize)
            bdims[j] = blocksize;
        else if(bdims[j] <= 0)
            return;

        offset += stride * (coords[j] * blocksize);
    }

    if(arys[narys-1] == NULL)
    {
        arys[narys-1] = PyArray_New(&PyArray_Type,
                                    lastary->base->ndims,
                                    bdims,
                                    lastary->base->dtype,
                                    strides,
                                    lastary->base->data + offset,
                                    lastary->base->elsize,
                                    NPY_WRITEABLE | NPY_CARRAY, NULL);
    }
    else
    {
        int size = lastary->base->ndims * sizeof(npy_intp);
        memcpy(PyArray_DIMS(arys[narys-1]), bdims, size);
        memcpy(PyArray_STRIDES(arys[narys-1]), strides, size);
        PyArray_BYTES(arys[narys-1]) = lastary->base->data + offset;
    }

    npy_intp *lastdims = bdims;
    npy_intp lastndim = lastary->base->ndims;

    //If just one array is NULL we are finished.
    for(j=0; j<narys; j++)
        if(arys[j] == NULL)
            return;

    //Create broadcasted iterators.
    PyObject *iters[NPY_MAXARGS];
    for(j=0; j<narys; j++)
        iters[j] = PyArray_BroadcastToShape(arys[j], lastdims, lastndim);

    //The iterators should not iterate over the axis-dimension. That
    //will be done inside the function call.
    int axis = lastndim-1;
    for(j=0; j<narys; j++)
    {
        PyArrayIterObject *it = (PyArrayIterObject *) iters[j];
        it->contiguous = 0;
        it->dims_m1[axis] = 0;
        it->backstrides[axis] = 0;
    }

    //niters is the total array length without the axis-
    //dimension.
    npy_intp niters = 1;
    for(j=0; j<lastndim; j++)
        niters *= (((PyArrayIterObject*)iters[narys-1])->dims_m1[j] + 1);

    //'steps' is the axis-dimension's stride.
    npy_intp steps[NPY_MAXARGS];
    for(j=0; j<narys; j++)
        steps[j] = ((PyArrayIterObject *) iters[j])->strides[axis];

    char *tdata[NPY_MAXARGS];
    while(niters-- > 0)
    {
        //Need data pointers in one char array.
        for(j=0; j<narys; j++)
            tdata[j] = PyArray_ITER_DATA(iters[j]);

#ifdef DNPY_TIME
    gettimeofday(&tv, &tz);
    tmp_time = (long long)tv.tv_usec + ((long long)tv.tv_sec)*1000000;
#endif
        function(tdata, &lastdims[axis], steps, funcdata);
#ifdef DNPY_TIME
    gettimeofday(&tv, &tz);
    total_apply_time += (long long)tv.tv_usec + ((long long)tv.tv_sec)*1000000 - tmp_time;
#endif
        for(j=0; j<narys; j++)
            PyArray_ITER_NEXT(iters[j]);
    }

    //Cleanup
    for(j=0; j<narys; j++)
        Py_DECREF(iters[j]);
}/* apply_ufunc */


/*===================================================================
 *
 * Putting the element 'src' into the distributed array 'dest_ary' at
 * coordinates 'lcoords' (local to the 'src_ary' array).
 * No broadcasting, but if 'reduce_axis' >= 0, the 'reduce_axis'
 * dimension will be removed.
 * The coordinates are in Row-major.
 * Private.
 *//*
static int remote_put(char *src, dndarray *src_ary, dndarray *dest_ary,
                      int reduce_axis, MPI_Win *win,
                      npy_intp lcoords[NPY_MAXDIMS])
{
    //Convert local coordinates to global coordinates.
    npy_intp gcoords[NPY_MAXDIMS];
    int count = 0;
    int i;
    int *cdimsizes = cart_dim_sizes[src_ary->ndims-1];
    for(i=0; i<src_ary->ndims; i++)
    {
        int tcoords[NPY_MAXDIMS];
        rank2cart(src_ary->ndims, myrank, tcoords);
        if(reduce_axis != i)
            gcoords[count++] = tcoords[i] * blocksize + //process offset
                               (lcoords[i] / blocksize) * //block index
                               (cdimsizes[i] * blocksize) + //jump per block
                               lcoords[i] % blocksize; //index inside block
    }
    if(count != dest_ary->ndims)
    {
        fprintf(stderr, "remote_put, the src_ary->ndims should be the "
                        "same as dest_ary->ndims or one higher if "
                        "reduce_axis >= 0\n");
        MPI_Abort(MPI_COMM_WORLD, -1);
    }

    //Find process cartesian coords and local offset.
    int pcoords[NPY_MAXDIMS];
    npy_intp offset = 0;
    for(i=dest_ary->ndims-1; i>=0; i--)
    {
        npy_intp B = gcoords[i] / blocksize;
        int P = cart_dim_sizes[dest_ary->ndims-1][i];
        pcoords[i] = B % P;
        npy_intp block_index = B / P;
        npy_intp index = gcoords[i] % blocksize;

        npy_intp stride = 1;
        int s;
        for(s=i+1; s<dest_ary->ndims; s++)
            stride *= dnumroc(dest_ary->dims[s], blocksize,
                              pcoords[s],
                              cart_dim_sizes[dest_ary->ndims-1][s]);

        offset += stride * (block_index * blocksize +
                            index);
    }

    //Convert to a MPI rank number.
    int rank = cart2rank(dest_ary->ndims, pcoords);

    //Transfer element.
    int itemsize = dest_ary->elsize;
    MPI_Win_lock(MPI_LOCK_EXCLUSIVE, rank, 0, *win);
    MPI_Put(src, itemsize, MPI_BYTE, rank, offset*dest_ary->elsize,
            dest_ary->elsize, MPI_BYTE, *win);
    MPI_Win_unlock(rank, *win);

    return 0;
}*//* remote_put */

/*===================================================================
 *
 * Putting the block 'src' into the distributed array 'dest_ary' at
 * block-coordinate 'lcoords' (local to the 'src_ary' array).
 * No broadcasting, but if 'reduce_axis' >= 0, the 'reduce_axis'
 * dimension will be removed.
 * The coordinates are in Row-major.
 * Private.
 */
static int remote_blockput(PyObject *src, dndarray *src_ary,
                           dndarray *dest_ary,
                           int reduce_axis,
                           npy_intp lcoords[NPY_MAXDIMS])
{
    int i, j, src_ndims = PyArray_NDIM(src);

    //Find local block in src.
    int bdims[NPY_MAXDIMS];
    int ldims[NPY_MAXDIMS];
    int subary_coords[NPY_MAXDIMS];
    j=0;
    for(i=0; j<src_ndims; i++)
        if(reduce_axis != i)
            subary_coords[j++] = lcoords[i] * blocksize;

    //Convert npy_intp to int.
    for(i=0; i<src_ndims; i++)
    {
        ldims[i] = PyArray_DIM(src,i);
        bdims[i] = ldims[i] - subary_coords[i];
        if(bdims[i] > blocksize)
            bdims[i] = blocksize;
    }

    //Setup a matching MPI datatype for a block.
    MPI_Datatype src_ary_type, elem_type, dest_ary_type;
    MPI_Type_contiguous(PyArray_ITEMSIZE(src), MPI_BYTE, &elem_type);
    MPI_Type_create_subarray(src_ndims,
                             ldims,
                             bdims,
                             subary_coords,
                             MPI_ORDER_C, elem_type, &src_ary_type);
    MPI_Type_commit(&src_ary_type);

    //Convert local coordinates to global coordinates.
    npy_intp gcoords[NPY_MAXDIMS];
    int *cdimsizes = cart_dim_sizes[src_ary->ndims-1];
    j=0;
    for(i=0; i<src_ary->ndims; i++)
    {
        int tcoords[NPY_MAXDIMS];
        rank2cart(src_ary->ndims, myrank, tcoords);
        if(reduce_axis != i)
            gcoords[j++] = tcoords[i] +   //process offset
                           lcoords[i] *   //block index
                           cdimsizes[i];  //jump per block
    }
    if(j != dest_ary->ndims)
    {
        fprintf(stderr, "remote_put, the src_ary->ndims should be the "
                        "same as dest_ary->ndims or one higher if "
                        "reduce_axis >= 0\n");
        MPI_Abort(MPI_COMM_WORLD, -1);
    }

    //Find process cartesian coords and dest block.
    int pcoords[NPY_MAXDIMS];
    cdimsizes = cart_dim_sizes[dest_ary->ndims-1];
    for(i=0; i<dest_ary->ndims; i++)
    {
        subary_coords[i] = (gcoords[i] / cdimsizes[i]) * blocksize;
        pcoords[i] = gcoords[i] % cdimsizes[i];
        ldims[i] = dnumroc(dest_ary->dims[i], blocksize,
                           pcoords[i], cdimsizes[i]);
        bdims[i] = ldims[i] - subary_coords[i];
        if(bdims[i] > blocksize)
            bdims[i] = blocksize;
    }
    //Convert to a MPI rank number.
    int rank = cart2rank(dest_ary->ndims, pcoords);

    //Setup a matching MPI datatype for a block.
    MPI_Type_create_subarray(dest_ary->ndims,
                             ldims,
                             bdims,
                             subary_coords,
                             MPI_ORDER_C, elem_type, &dest_ary_type);
    MPI_Type_commit(&dest_ary_type);

    //Transfer element.
    //A exclusive lock is required since two put updates must not
    //concurrently access nonoverlapping locations in a window
    //(MPI-2.1 s. 351).
    MPI_Win_lock(MPI_LOCK_EXCLUSIVE, rank, 0, dest_ary->comm_win);
    MPI_Put(PyArray_BYTES(src), 1, src_ary_type, rank, 0, 1,
            dest_ary_type, dest_ary->comm_win);
    MPI_Win_unlock(rank, dest_ary->comm_win);

    //Clean up
    MPI_Type_free(&src_ary_type);
    MPI_Type_free(&dest_ary_type);
    MPI_Type_free(&elem_type);

    return 0;
} /* remote_blockput */


/*===================================================================
 *
 * Access a distributed array element located at 'src_ary' using local
 * coordinates (local to the 'dest_ary' array and NOT the view).
 * Broadcasting if required.
 * The coordinates are in Row-major.
 * Private.
 */
static int remote_get(char *dest, dndview *src_ary, dndview *dest_ary,
                      const npy_intp lcoords[NPY_MAXDIMS])
{
    int i;
    int ai = 0;
    int bi = 0;

    //A scalar dummy can be returned directly.
    if(src_ary->base->ndims == 0)
    {
        memcpy(dest, src_ary->base->data, src_ary->base->elsize);
        return 0;
    }
/*
    printf("src_ary-view id: %d\n",src_ary->uid);
    printf("1 - lcoords:");
    for(i=0; i<dest_ary->base->ndims; i++)
        printf(" %d", lcoords[i]);
    printf("\n");
*/

    //Convert local coordinates to global coordinates based on dest_ary.
    npy_intp coordsA[NPY_MAXDIMS];
    npy_intp coordsB[NPY_MAXDIMS];
    int *cdimsizes = cart_dim_sizes[dest_ary->base->ndims-1];
    for(i=0; i<dest_ary->base->ndims; i++)
    {
        int tcoords[NPY_MAXDIMS];
        rank2cart(dest_ary->base->ndims, myrank, tcoords);
        coordsA[i] = tcoords[i] * blocksize + //process offset
                     (lcoords[i] / blocksize) * //block index
                     (cdimsizes[i] * blocksize) + //jump per block
                     lcoords[i] % blocksize; //index inside block
    }
/*
    printf("2 - coordsA:");
    for(i=0; i<dest_ary->base->ndims; i++)
        printf(" %d", coordsA[i]);
    printf("\n");
*/

    //Convert global coordinates to inter-view coordinates.
    //Return -1 if the global coordinate it not part of the view.
    //NB: the number of coordinates in 'gcoords' may be reduced.
    for(i=0; i<dest_ary->nslice; i++)
    {
        if(dest_ary->slice[i].nsteps == SingleIndex)
        {
            if(coordsA[ai++] != dest_ary->slice[i].start)
                return -1;
            else
                continue;
        }
        if(dest_ary->slice[i].nsteps == PseudoIndex)
            continue;

        //Check that the coordinate is in the right "step".
        if(coordsA[ai] % dest_ary->slice[i].step !=
           dest_ary->slice[i].start % dest_ary->slice[i].step)
        {
            return -1;
        }

        //Check that the coordinate is not greater than the view length.
        if(coordsA[ai] / dest_ary->slice[i].step >=
           dest_ary->slice[i].nsteps + dest_ary->slice[i].start)
        {
            return -1;
        }

        //Compute inter-view coordinate and check that it not starts
        //before the view.
        coordsB[bi] = coordsA[ai] - dest_ary->slice[i].start;
        if(coordsB[bi] < 0)
        {
            return -1;
        }
        coordsB[bi] /= dest_ary->slice[i].step;

        ai++; bi++;
    }
/*
    printf("3 - coordsB:");
    for(i=0; i<bi; i++)
        printf(" %d", coordsB[i]);
    printf("\n");
*/

    //Convert inter-view coordinates to global coordinates based on
    //src_ary. Including broadcasting.
    //NB: the coordinates in gcoords is in the reverse order after
    //this routine.
    ai = 0;
    bi--;
    for(i=src_ary->nslice-1; i>=0; i--)
    {
        if(src_ary->slice[i].nsteps == SingleIndex)
        {
            coordsA[ai++] = src_ary->slice[i].start;
            continue;
        }

        if(src_ary->slice[i].nsteps == PseudoIndex)
            continue;

        if(src_ary->slice[i].nsteps == 1)
            coordsA[ai] = src_ary->slice[i].start;//broadcasting 1-dims.
        else
            coordsA[ai] = src_ary->slice[i].start +
                          coordsB[bi] * src_ary->slice[i].step;
        ai++; bi--;
    }

    //Reverse again to get the correct order of the coordinates.
    for(i=0; i<src_ary->base->ndims; i++)
        coordsB[i] = coordsA[--ai];

/*
    printf("4 - coordsB:");
    for(i=0; i<src_ary->base->ndims; i++)
        printf(" %d", coordsB[i]);
    printf("\n");
*/


    //Find process cartesian coords and local offset.
    int pcoords[NPY_MAXDIMS];
    npy_intp offset = 0;
    for(i=src_ary->base->ndims-1; i>=0; i--)
    {
        npy_intp B = coordsB[i] / blocksize;
        int P = cart_dim_sizes[src_ary->base->ndims-1][i];
        pcoords[i] = B % P;
        npy_intp block_index = B / P;
        npy_intp index = coordsB[i] % blocksize;

        npy_intp stride = 1;
        int s;
        for(s=i+1; s<src_ary->base->ndims; s++)
            stride *= dnumroc(src_ary->base->dims[s],
                              blocksize,
                              pcoords[s],
                              cart_dim_sizes[src_ary->base->ndims-1][s]);

        offset += stride * (block_index *
                            blocksize + index);
    }

    //Convert to a MPI rank number.
    int rank = cart2rank(src_ary->base->ndims, pcoords);

    //Transfer element.
    MPI_Win_lock(MPI_LOCK_SHARED, rank, MPI_MODE_NOCHECK, src_ary->base->comm_win);
    MPI_Get(dest, src_ary->base->elsize, MPI_BYTE, rank,
            offset*src_ary->base->elsize,
            src_ary->base->elsize, MPI_BYTE, src_ary->base->comm_win);
    MPI_Win_unlock(rank, src_ary->base->comm_win);

//    printf("5 - reading data at rank %d, value: %lf\n", rank, *((double*)dest));

    return 0;
}/* remote_get */


/*===================================================================
 *
 * Access a distributed array block located at 'src_ary' using local
 * block coordinates (local to the 'dest_ary' array and NOT the view).
 * 'locked_rank' is the rank which should be unlocked before the
 * transfer is complete.
 * Broadcasting if required.
 * The coordinates are in Row-major.
 * Private.
 */
static int remote_blockget(char *dest, dndview *src_ary,
                           dndview *dest_ary,
                           const npy_intp lcoords[NPY_MAXDIMS],
                           int *locked_rank,
                           PyObject **ret_ary)
{
    int i, ai, bi, vi, rbi;
#ifdef DNPY_TIME
    gettimeofday(&tv, &tz);
    tmp_time = (long long)tv.tv_usec + ((long long)tv.tv_sec)*1000000;
#endif
    //Tmp coords
    npy_intp coordsA[NPY_MAXDIMS];
    npy_intp coordsB[NPY_MAXDIMS];
    int pcoordsR[NPY_MAXDIMS];
    int pcoords[NPY_MAXDIMS];

    //-1 means that no rank where lucked.
    *locked_rank = -1;

    //A scalar dummy can be returned directly.
    if(src_ary->base->ndims == 0)
    {
        memcpy(dest, src_ary->base->data, src_ary->base->elsize);
        return -1;
    }

/*
    printf("src_ary-view id: %ld\n",src_ary->uid);
    printf("1 - lcoords:");
    for(i=0; i<dest_ary->base->ndims; i++)
        printf(" %ld", lcoords[i]);
    printf("\n");
*/

    //Convert local coordinates to global coordinates.
    for(i=0; i<dest_ary->base->ndims; i++)
    {
        rank2cart(dest_ary->base->ndims, myrank, pcoords);
        coordsA[i] = pcoords[i] + lcoords[i] *
                     cart_dim_sizes[dest_ary->base->ndims-1][i];
    }
/*
    printf("2 - coordsA (global coords  based on dest_ary):");
    for(i=0; i<dest_ary->base->ndims; i++)
        printf(" %ld", coordsA[i]);
    printf("\n");
*/

    //Convert coords to be based on 'src_ary' instead of 'dest_ary'.
    //Including broadcasting.
    //NB: the coordinates in coordsB is in reverse order after this
    //routine.
    bi=0;
    ai=dest_ary->base->ndims-1;
    rbi=src_ary->base->ndims-1;
    int *cart_dim_size = cart_dim_sizes[src_ary->base->ndims-1];
    for(i=src_ary->nslice-1; i >= 0; i--)
    {
        if(src_ary->slice[i].nsteps == PseudoIndex)
        {
            continue;
        }
        if(src_ary->slice[i].nsteps == SingleIndex)
        {
            //Note src_ary->slice[i].start is NOT a block index.
            npy_intp B = src_ary->slice[i].start / blocksize;
            int P = cart_dim_size[rbi];

            //Find process cartesian coords.
            pcoordsR[bi] = B % P;

            //Convert global coordinates to local coordinates.
            coordsB[bi] = (B / P) * blocksize +
                          src_ary->slice[i].start % blocksize;
        }
        else
        {
            if(src_ary->slice[i].nsteps == 1)
                coordsB[bi] = 0;//broadcasting 1-dims.
            else
                coordsB[bi] = coordsA[ai];

            //Find process cartesian coords.
            pcoordsR[bi] = coordsB[bi] % cart_dim_size[rbi];

            //Convert global block coordinates to local (non-block)
            //coordinates.
            coordsB[bi] = (coordsB[bi] / cart_dim_size[rbi]) * blocksize;
            ai--;
        }
        rbi--;
        bi++;
    }

    //Reverse again to get the correct order of the MPI-coordinates.
    int pcoords_size = bi;
    for(i=0; i<src_ary->base->ndims; i++)
        pcoords[i] = pcoordsR[--bi];

/*
    printf("3 - pcoords:");
    for(i=0; i<src_ary->base->ndims; i++)
        printf(" %d", pcoords[i]);
    printf("\n");
*/

    //Convert to a MPI rank number.
    int rank = cart2rank(src_ary->base->ndims, pcoords);

    //If the block is located locally, we use a methode which dont do
    //any memory copying.
    if(rank != myrank || src_ary->alterations != 0)
    {
        //Arrays for MPI_Type_create_subarray.
        int bdims[NPY_MAXDIMS];        //Using int instead of npy_intp
        int ldims[NPY_MAXDIMS];        //here because MPI_Type_create_
        int subary_coords[NPY_MAXDIMS];//subarray only support int.

        //Reverse again to get the correct order of the coordinates and
        //convert from npy_intp to int.
        for(i=0; i<src_ary->base->ndims; i++)
            subary_coords[i] = coordsB[--pcoords_size];

        //Compute block-/local-/visible-dims.
        npy_intp msgsize = src_ary->base->elsize;
        npy_intp vdims[NPY_MAXDIMS];
        bi=0, vi=0;
        for(i=0; i < src_ary->nslice; i++)
        {
            if(src_ary->slice[i].nsteps == PseudoIndex)
            {
                vdims[vi++] = 1;
                continue;
            }
            if(src_ary->slice[i].nsteps == SingleIndex)
            {
                //Block dim is just one.
                bdims[bi] = 1;

                //Compute local dims (local to the target MPI-process).
                ldims[bi] = dnumroc(src_ary->base->dims[bi],
                                    blocksize,
                                    pcoords[bi],
                                    cart_dim_size[bi]);
            }
            else
            {
                //Compute local dims (local to the target MPI-process).
                ldims[bi] = dnumroc(src_ary->base->dims[bi],
                                    blocksize,
                                    pcoords[bi],
                                    cart_dim_size[bi]);

                //Compute a block's dims.
                bdims[bi] = ldims[bi] - subary_coords[bi];
                if(bdims[bi] > blocksize)
                    bdims[bi] = blocksize;
                else if(bdims[bi] <= 0)
                    return -1;

                vdims[vi++] = bdims[bi];
            }
            //Computes total message size.
            msgsize *= bdims[bi];
            bi++;
        }

//        printf("MPI_Type_create_subarray - dist_ndims: %d -> src_ndims: %d, ldims: [%d, %d],  bdims: [%d, %d], subary_coords: [%d, %d], msgsize: %ld\n", dest_ary->base->ndims, src_ary->base->ndims, ldims[0], ldims[1], bdims[0], bdims[1], subary_coords[0], subary_coords[1], msgsize);

        //Setup a matching MPI datatype.
        MPI_Datatype arraytype;
        MPI_Type_create_subarray(src_ary->base->ndims,
                                 ldims,
                                 bdims,
                                 subary_coords,
                                 MPI_ORDER_C, src_ary->base->mpi_dtype, 
                                 &arraytype);
        MPI_Type_commit(&arraytype);

        //Transfer element.
        *locked_rank = rank;
        MPI_Win_lock(MPI_LOCK_SHARED, rank, MPI_MODE_NOCHECK, src_ary->base->comm_win);
        MPI_Get(dest, msgsize, MPI_BYTE, rank, 0, 1, arraytype, src_ary->base->comm_win);

        MPI_Type_free(&arraytype);
    /*
        printf("4 - reading data at rank %d, value: %lf, %lf, %lf, %lf\n", rank, ((double*)dest)[0],((double*)dest)[1],((double*)dest)[2],((double*)dest)[3]);
        printf("5 - Create array - ndims: %d, vdims: [%ld, %ld]\n", src_ary->ndims, vdims[0], vdims[1]);
    */

        if(*ret_ary == NULL)
        {
            *ret_ary = PyArray_New(&PyArray_Type,
                       src_ary->ndims,
                       vdims,
                       src_ary->base->dtype,
                       NULL,
                       dest,
                       src_ary->base->elsize,
                       NPY_WRITEABLE | NPY_CARRAY, NULL);
        }
        else
        {
            //Compute stride.
            npy_intp stride = src_ary->base->elsize;
            npy_intp strides[NPY_MAXDIMS];
            for(i = src_ary->ndims-1; i >= 0; i--)
            {
                strides[i] = stride;
                stride *= vdims[i];
            }
            int size = src_ary->ndims * sizeof(npy_intp);
            memcpy(PyArray_DIMS(*ret_ary), vdims, size);
            memcpy(PyArray_STRIDES(*ret_ary), strides, size);
            PyArray_BYTES(*ret_ary) = dest;
        }
    }
    else
    {
        npy_intp subary_coords[NPY_MAXDIMS];

        //Reverse again to get the correct order of the coordinates.
        for(i=0; i<src_ary->base->ndims; i++)
            subary_coords[i] = coordsB[--pcoords_size];
/*
        printf("3 - subary_coords (local non-block coords based on src_ary):");
        for(i=0; i<src_ary->base->ndims; i++)
            printf(" %ld", subary_coords[i]);
        printf("\n");
*/
        //Compute a block's dims for the dest array.
        npy_intp bdims[NPY_MAXDIMS];
        npy_intp strides[NPY_MAXDIMS];
        npy_intp offset = 0;
        int j;
        for(j=0; j < src_ary->base->ndims; j++)
        {
            npy_intp stride = src_ary->base->elsize;
            int s;
            for(s=j+1; s<src_ary->base->ndims; s++)
                stride *= src_ary->base->localdims[s];
            strides[j] = stride;

            //Check if this MPI-process has a part of the arrays.
            if(stride < 1)
                return -1;

            bdims[j] = src_ary->base->localdims[j] - subary_coords[j];
            if(bdims[j] <= 0)
                return -1;
            else if(bdims[j] > blocksize)
                bdims[j] = blocksize;

            offset += stride * subary_coords[j];
        }

        if(*ret_ary == NULL)
        {
            *ret_ary = PyArray_New(&PyArray_Type,
                                   src_ary->ndims,
                                   bdims,
                                   src_ary->base->dtype,
                                   strides,
                                   src_ary->base->data + offset,
                                   src_ary->base->elsize,
                                   NPY_WRITEABLE | NPY_CARRAY, NULL);
        }
        else
        {
            int size = src_ary->ndims * sizeof(npy_intp);
            memcpy(PyArray_DIMS(*ret_ary), bdims, size);
            memcpy(PyArray_STRIDES(*ret_ary), strides, size);
            PyArray_BYTES(*ret_ary) = src_ary->base->data + offset;
        }

        //Force some progress in the MPI-layer.
        int f=0;
        MPI_Iprobe(MPI_ANY_SOURCE, MPI_ANY_TAG,
                   MPI_COMM_WORLD, &f, MPI_STATUS_IGNORE);
    }
/*
    printf("GET:\n");
    PyObject_Print(*ret_ary,stdout,0);
    printf("\n");
*/
#ifdef DNPY_TIME
    gettimeofday(&tv, &tz);
    total_rget_time += ((long long)tv.tv_usec + ((long long)tv.tv_sec)*1000000) - tmp_time;
#endif
    return 0;
}/* remote_blockget */

/*===================================================================
 *
 * Message handler for INIT_BLOCKSIZE.
 * Private.
 */
static void do_INIT_BLOCKSIZE(npy_intp new_blocksize)
{
    //Print debug info.
    #ifdef DISTNUMPY_DEBUG
        printf("INIT_BLOCKSIZE - block size: %ld\n",(long)new_blocksize);
    #endif

    blocksize = new_blocksize;


} /* do_INIT_BLOCKSIZE */

/*===================================================================
 *
 * Message handler for CREATE_ARRAY.
 * Private.
 */
static void do_CREATE_ARRAY(dndarray *ary, dndview *view)
{
    int ndims = ary->ndims;
    int *dims = cart_dim_sizes[ndims-1];
    int i;

    //Get cartesian coords.
    int coords[NPY_MAXDIMS];
    rank2cart(ndims, myrank, coords);

    //Acummulate total number of local sizes and save it.
    npy_intp nelements = 1;
    for(i=0; i < ary->ndims; i++)
    {
        ary->localdims[i] = dnumroc(ary->dims[i],
                            blocksize, coords[i], dims[i]);
        nelements *= ary->localdims[i];
        ary->localblockdims[i] = ceil(ary->localdims[i] /
                                      (double) blocksize);
    }
    ary->nelements = nelements;

    //Allocate local data.
    if(MPI_Alloc_mem(nelements * ary->elsize, MPI_INFO_NULL,
                     &ary->data) != MPI_SUCCESS)
    {
        fprintf(stderr, "Out of memory!\n");
        MPI_Abort(MPI_COMM_WORLD, -1);
    }

    //Create the window for one-sided communication.
    MPI_Win_create(ary->data, nelements * ary->elsize, 1, MPI_INFO_NULL,
                   MPI_COMM_WORLD, &ary->comm_win);

    //Create a MPI-datatype that correspond to an array element.
    MPI_Type_contiguous(ary->elsize, MPI_BYTE, &ary->mpi_dtype);
    MPI_Type_commit(&ary->mpi_dtype);

    //Allocate and copy the array to the views's base.
    view->base = malloc(sizeof(dndarray));
    memcpy(view->base, ary, sizeof(dndarray));

    //Save the new view.
    put_dndview(view);

    //Print debug info.
    #ifdef DISTNUMPY_DEBUG
        printf("CREATE_ARRAY - uid: %ld, dtype: %s(%d),"
               " nelem: %ld, ndims: %d, dims: {",
               (long) view->uid,
               types_numpy2str(ary->dtype), ary->elsize,
               (long)nelements, ndims);
        for(i=0; i<ndims-1; i++)
            printf("%ld ", (long)ary->dims[i]);
        printf("%ld}, localdims: {", (long)ary->dims[ndims-1]);
        for(i=0; i<ndims-1; i++)
            printf("%ld ", (long)ary->localdims[i]);
        printf("%ld}, \n", (long)ary->localdims[ndims-1]);
    #endif

} /* do_CREATE_ARRAY */


/*===================================================================
 *
 * Message handler for DESTROY_ARRAY.
 * Private.
 */
static void do_DESTROY_ARRAY(npy_intp uid)
{
    //Print debug info.
    #ifdef DISTNUMPY_DEBUG
        printf("DESTROY_ARRAY - uid: %ld\n", (long) uid);
    #endif
    rm_dndview(uid);

} /* do_DESTROY_ARRAY */


/*===================================================================
 *
 * Message handler for CREATE_VIEW.
 * Private.
 */
static void do_CREATE_VIEW(npy_intp org_view_uid, dndview *newview)
{
    dndview *orgview = get_dndview(org_view_uid);
    //Print debug info.
    #ifdef DISTNUMPY_DEBUG
        int i;
        printf("CREATE_VIEW - adding view id: %ld, ndims: %d, slices:",
               (long)newview->uid, newview->ndims);
        for(i=0; i<newview->nslice; i++)
            printf(" (%ld, %ld, %ld)",(long)newview->slice[i].start,
                                      (long)newview->slice[i].step,
                                      (long)newview->slice[i].nsteps);
        printf(", based on view id: %ld, ndims: %d, slices:",
               (long)orgview->uid, orgview->ndims);
        for(i=0; i<orgview->nslice; i++)
            printf(" (%ld, %ld, %ld)",(long)orgview->slice[i].start,
                                      (long)orgview->slice[i].step,
                                      (long)orgview->slice[i].nsteps);
        printf("\n");
    #endif

    //Add the base to the new view.
    newview->base = orgview->base;
    newview->base->refcount++;

    //Save the view.
    put_dndview(newview);
} /* do_CREATE_VIEW */


/*===================================================================
 *
 * Message handler for SET_ITEM.
 * Private.
 */
static void do_SET_ITEM(dndview *view, char* value,
                        npy_intp coord[NPY_MAXDIMS])
{
    dndarray *base = view->base;
    int i;

    //Get element size.
    int elsize = base->elsize;

    //Print debug info.
    #ifdef DISTNUMPY_DEBUG
        if(myrank == 0)
        {
            printf("SET_ITEM - uid: %ld, coord: (", (long)view->uid);
            for(i=0; i < view->ndims; i++)
                printf(" %ld", (long)coord[i]);
            printf(")\n");
        }
    #endif

    if(myrank == 0)
    {
        npy_intp newcoord[NPY_MAXDIMS];
        view2base_globalcoord(coord, view, newcoord);

        //Find process cartesian coords and local offset.
        int pcoords[NPY_MAXDIMS];
        npy_intp offset = 0;
        for(i=base->ndims-1; i>=0; i--)
        {
            npy_intp B = newcoord[i] / blocksize;
            int P = cart_dim_sizes[base->ndims-1][i];
            pcoords[i] = B % P;
            npy_intp block_index = B / P;
            npy_intp index = newcoord[i] % blocksize;

            npy_intp stride = 1;
            int s;
            for(s=i+1; s<base->ndims; s++)
                stride *= dnumroc(base->dims[s], blocksize,
                                  pcoords[s],
                                  cart_dim_sizes[base->ndims-1][s]);

            offset += stride * (block_index * blocksize + index);
        }

        //Convert to a MPI rank number.
        int rank = cart2rank(base->ndims, pcoords);

        //Transfer element.
        MPI_Win_lock(MPI_LOCK_SHARED, rank, MPI_MODE_NOCHECK, base->comm_win);
        MPI_Put(value, elsize, MPI_BYTE, rank, offset*elsize, elsize,
                MPI_BYTE, base->comm_win);
        MPI_Win_unlock(rank, base->comm_win);
    }
    MPI_Barrier(MPI_COMM_WORLD);
} /* do_SET_ITEM */

/*===================================================================
 *
 * Message handler for GET_ITEM.
 * Private.
 */
static void do_GET_ITEM(char *retdata, dndview *view,
                        npy_intp coord[NPY_MAXDIMS])
{
    dndarray *base = view->base;
    int i;

    //Get element size.
    int elsize = base->elsize;

    //Print debug info.
    #ifdef DISTNUMPY_DEBUG
        if(myrank == 0)
        {
            printf("GET_ITEM - uid: %ld, coord: (", (long)view->uid);
            for(i=0; i < view->ndims; i++)
                printf(" %ld", (long)coord[i]);
            printf(")\n");
        }
    #endif

    if(myrank == 0)
    {
        npy_intp newcoord[NPY_MAXDIMS];
        view2base_globalcoord(coord, view, newcoord);

        //Find process cartesian coords and local offset.
        int pcoords[NPY_MAXDIMS];
        npy_intp offset = 0;
        for(i=base->ndims-1; i>=0; i--)
        {
            npy_intp B = newcoord[i] / blocksize;
            int P = cart_dim_sizes[base->ndims-1][i];
            pcoords[i] = B % P;
            npy_intp block_index = B / P;
            npy_intp index = newcoord[i] % blocksize;

            npy_intp stride = 1;
            int s;
            for(s=i+1; s<base->ndims; s++)
                stride *= dnumroc(base->dims[s], blocksize,
                                  pcoords[s],
                                  cart_dim_sizes[base->ndims-1][s]);

            offset += stride * (block_index * blocksize + index);
        }

        //Convert to a MPI rank number.
        int rank = cart2rank(base->ndims, pcoords);

        //Transfer element.
        MPI_Win_lock(MPI_LOCK_SHARED, rank, MPI_MODE_NOCHECK, base->comm_win);
        MPI_Get(retdata, elsize, MPI_BYTE, rank, offset*elsize, elsize,
                MPI_BYTE, base->comm_win);
        MPI_Win_unlock(rank, base->comm_win);
    }
    MPI_Barrier(MPI_COMM_WORLD);

} /* do_GET_ITEM */

/*===================================================================
 *
 * Message handler for UFUNC.
 * Private.
 */
static void do_UFUNC(npy_intp arylist[NPY_MAXARGS], int narys,
                     int nout_arys, int appropriate_function,
                     int scalar_dtype, int scalar_size,
                     char* scalar, char* op)
{
    int i, j;
#ifdef DNPY_TIME
    gettimeofday(&tv, &tz);
    total_time_tmp = (long long)tv.tv_usec + ((long long)tv.tv_sec)*1000000;
#endif

    //Get arrray structs.
    dndview *views[NPY_MAXARGS];
    int nin = narys - nout_arys;

    if(nout_arys != 1)
    {
        fprintf(stderr, "At the moment distnumpy only supports "
                        "ufunc with one output array\n");
        MPI_Abort(MPI_COMM_WORLD, -1);
    }

    for(i=0; i<narys; i++)
        if(arylist[i])
        {
            views[i] = get_dndview(arylist[i]);
        }
        else
        {    //Create a dummy view for the scalar.
            views[i] = malloc(sizeof(dndview));
            views[i]->uid = 0;
            views[i]->base = malloc(sizeof(dndarray));
            views[i]->base->ndims = 0;
            views[i]->base->dtype = scalar_dtype;
            views[i]->base->nelements = 1;
            views[i]->base->data = scalar;
            views[i]->base->elsize = scalar_size;
        }

    dndview *lastary = views[narys-1];

    //Print debug info.
    #ifdef DISTNUMPY_DEBUG
        printf("UFUNC (%s) on - in:", op);
        for(i=0; i<nin; i++)
            printf(" %ld", (long)views[i]->uid);
        printf(" out:");
        for(i=nin; i<narys; i++)
            printf(" %ld", (long)views[i]->uid);
        printf(", scalar-size: %d\n", scalar_size);
    #endif

    //Get Python function.
    PyObject *PyOp = PyObject_GetAttrString(ufunc_module, op);
    if(PyOp == NULL)
    {
        fprintf(stderr, "UFUNC : unknown operator: %s\n", op);
        MPI_Abort(MPI_COMM_WORLD, -1);
    }
    //ufunc's objects.
    PyUFuncObject *ufunc_obj = (PyUFuncObject*) PyOp;

    //Analyse if it is possible to use a simplified computation method.
    //* If all array have the same number of dimensions and if the views
    //  are all identical we can use a very simple end efficient
    //  computation method which do not require any communication..
    //* If the only view-alteration is concerning the number of visible
    //  dimensions we can use a block-oriented computation method.
    int nocomm_method = 1;
    int block_method = 1;
    for(i=0; i<narys; i++)
    {
        //A scalar is also compatible with all methods.
        if(views[i]->base->ndims == 0)
            continue;

        if(views[i]->base->ndims != lastary->base->ndims ||
           views[i]->nslice != lastary->nslice)
        {
            nocomm_method = 0;
        }
        else if(memcmp(views[i]->slice, lastary->slice,
                 views[i]->nslice * sizeof(dndslice)))
        {
             nocomm_method = 0;
        }
        if(views[i]->alterations & ~DNPY_NDIMS)
        {
            block_method = 0;
        }
    }
    //At the moment the block-method requires that the output array
    //has no alterations.
    //TODO: This shouldn't be a requirement.
    for(i=nin; i<narys; i++)
        if(views[i]->alterations != 0)
            block_method = 0;

    if(nocomm_method)
    {
//        printf("UFUNC - nocomm_method\n");
        npy_intp dims[NPY_MAXDIMS];
        npy_intp strides[NPY_MAXDIMS];
        PyObject *arytuple = PyTuple_New(narys);
        for(i=0; i < narys; i++)
        {
            char *data = views[i]->base->data;

            //A scalar is very simple.
            if(views[i]->base->ndims == 0)
            {
                PyTuple_SET_ITEM(arytuple, i,
                                 PyArray_SimpleNewFromData(0, NULL,
                                        views[i]->base->dtype, data));
                continue;
            }

            //Compute the strides and dims from the view.
            int dj=0; int bj=0; int sj=0;
            while(sj < views[i]->nslice)
            {
                if(views[i]->slice[sj].nsteps == PseudoIndex)
                {
                    dims[dj] = 1;
                    strides[dj] = 0;
                    dj++;
                }
                else
                {
                    int stride = 1;
                    int s;
                    for(s=bj+1; s<views[i]->base->ndims; s++)
                        stride *= views[i]->base->localdims[s];

                    if(views[i]->slice[sj].nsteps == SingleIndex)
                    {
                        //Forwarding - invisible dimension.
                        data += views[i]->slice[sj].start * stride *
                                views[i]->base->elsize;
                    }
                    else
                    {
                        dims[dj] = views[i]->base->localdims[bj];
                        strides[dj] = stride * views[i]->base->elsize;
                        dj++;
                    }
                    bj++;
                }
                sj++;
            }
            PyTuple_SET_ITEM(arytuple, i, PyArray_New(&PyArray_Type,
                             views[i]->ndims, dims,
                             views[i]->base->dtype, strides, data,
                             views[i]->base->elsize,
                             NPY_WRITEABLE | NPY_CARRAY, NULL));
        }

        //Call the python operator.
        PyObject *t = PyObject_CallObject(PyOp, arytuple);
        Py_XDECREF(t);

        //Clean up
        Py_DECREF(arytuple);
        Py_DECREF(PyOp);
        for(i=0; i<narys; i++)
            if(views[i]->base->ndims == 0)//Scalar dummy.
            {
                free(views[i]->base);
                free(views[i]);
            }
        return;
    }//End of the simple method.

    npy_intp coords[NPY_MAXDIMS];
    memset(coords, 0, NPY_MAXDIMS * sizeof(npy_intp));
    if(block_method)
    {
        if(appropriate_function < 0)
        {
            fprintf(stderr, "do_UFUNC was called on a view but no "
                            "appropriate_function was provided\n");
            MPI_Abort(MPI_COMM_WORLD, -1);
        }

        //ufunc objects.
        PyUFuncGenericFunction ufunc_function =
                               ufunc_obj->functions[appropriate_function];
        void *ufunc_funcdata = ufunc_obj->data[appropriate_function];

        //Receive buffers. Two buffers per array (double buffering).
        char *buffer[2][NPY_MAXARGS];

        //Allocate buffers.
        for(i=0; i<nin; i++)
        {
            npy_intp s = views[i]->base->elsize;
            for(j=0; j<views[i]->base->ndims; j++)
            {
                npy_intp d = dnumroc(views[i]->base->dims[j], blocksize,
                         0, cart_dim_sizes[views[i]->base->ndims-1][j]);
                s *= (blocksize < d)?blocksize:d;
            }
            s = (s < views[i]->base->elsize)?views[i]->base->elsize:s;
            buffer[0][i] = malloc(s);
            buffer[1][i] = malloc(s);
            if(buffer[0][i] == NULL || buffer[1][i] == NULL)
            {
                fprintf(stderr, "Out of memory!\n");
                MPI_Abort(MPI_COMM_WORLD, -1);
            }
        }

        //Find the order of which the dimensions should be iterated.
        int dim_iter_order[NPY_MAXDIMS];
        int sc = 0; //start counter.
        int ec = lastary->ndims-1; //end counter.
        for(i=ec; i>=0; i--)
        {
            int broadcast = 0;
            if(lastary->base->dims[i] != 1)
            {
                for(j=0; j<nin; j++)
                {
                    int lastcount = i - (lastary->ndims -
                                         views[j]->ndims);
                    int s, si=views[j]->ndims-1, gotdim=0;
                    if(lastcount >= 0)
                    {
                        for(s=views[j]->nslice-1; s>=0; s--)
                        {
                            if(views[j]->slice[s].nsteps !=
                               SingleIndex)
                                if(si-- == lastcount)
                                {
                                    gotdim = 1;
                                    break;
                                }
                        }
                    }
                    if(gotdim == 0 || //no more dims.
                       views[j]->slice[s].nsteps == 1)//dim size is 1.
                    {
                        broadcast = 1;
                        break;
                    }
                }
            }
            //Dims we need to broadcast should be ordered first.
            if(broadcast)
                dim_iter_order[sc++] = i;
            else
                dim_iter_order[ec--] = i;
        }
        /*
        //Init coords. SingleIndex should be set here and never again
        //touched and the rest is zero.
        j=0;
        for(i=0; i < lastary->nslice; i++)
        {
            if(lastary->slice[i].nsteps == PseudoIndex)
                continue;

            if(lastary->slice[i].nsteps == SingleIndex)
                coords[j] = lastary->slice[i].start;
            j++;
        }
        */

        //Coordinate of the next block we need to remote-get.
        npy_intp next_get_coords[NPY_MAXARGS][NPY_MAXDIMS];
        memset(next_get_coords, 0, NPY_MAXARGS*NPY_MAXDIMS*
                                   sizeof(npy_intp));

        //List of intermediate array blocks (PyArrays) we should reuse.
        PyObject *obj_blocklist[NPY_MAXARGS];
        //List of intermediate array blocks we are receiving.
        PyObject *apply_blocklist[NPY_MAXARGS];
        //List of intermediate array blocks we should apply on.
        PyObject *recv_blocklist[NPY_MAXARGS];
        for(j=0; j<narys; j++)
        {
            recv_blocklist[j] = NULL;
            apply_blocklist[j] = NULL;
            obj_blocklist[j] = NULL;
        }

        //Ranks that were locked by the previous remote_blockget.
        int locked_ranks[NPY_MAXARGS];
        for(j=0;j<narys;j++)
            locked_ranks[j] = -1;

        //'e' indicates in which buffer we are receiving.
        int e[NPY_MAXARGS];// Even/odd switcher for
        memset(e, 0, NPY_MAXARGS * sizeof(int));

        //1. iteration - fill the recv_blocklist.
        for(j=0; j<nin; j++)
        {
            remote_blockget(buffer[0][j], views[j],
                            lastary,
                            coords,
                            &locked_ranks[j],
                            &recv_blocklist[j]);
        }

        //Do the rest of the iterations.
        int notfinished = 1;
        do{
            //Update apply_blocklist with the new blocks in
            //recv_blocklist.
            for(j=0; j<nin; j++)
                if(recv_blocklist[j] != NULL &&
                   apply_blocklist[j] == NULL)
                {
                    if(locked_ranks[j] >= 0)
                    {
                        MPI_Win_unlock(locked_ranks[j], views[j]->base->comm_win);
                        locked_ranks[j] = -1;
                    }
                    apply_blocklist[j] = recv_blocklist[j];
                    recv_blocklist[j] = NULL;
                }

            //Retrieve remote blocks if there is room in the
            //recv_blocklist.
            for(j=0; j<nin; j++)
                if(recv_blocklist[j] == NULL)
                {
                    //Update next_get_coords.
                    find_next_get_coords(next_get_coords[j], views[j],
                                         lastary, dim_iter_order);
                    //Flip
                    e[j] = (e[j]+1)%2;

                    //Now get the block we will need in the next
                    //iteration.
                    remote_blockget(buffer[e[j]][j],
                                    views[j],
                                    lastary,
                                    next_get_coords[j],
                                    &locked_ranks[j],
                                    &obj_blocklist[j]);
                    recv_blocklist[j] = obj_blocklist[j];
                    obj_blocklist[j] = NULL;
                }

            //Apply the universial function.
            apply_ufunc(narys, apply_blocklist, ufunc_function,
                        ufunc_funcdata, lastary, coords);

            //Iterate one block.
            for(j=0; j < lastary->base->ndims; j++)
            {
                int o = dim_iter_order[j];//Using the computed order.
                if(++coords[o] >= lastary->base->localblockdims[o])
                {
                    //We a finished, if wrapping around.
                    if(j == lastary->base->ndims-1)
                    {
                        notfinished = 0;
                        break;
                    }
                    coords[o] = 0;
                }
                else
                    break;
            }
            for(j=0; j<nin; j++)
            {
                //Use the previous block if we are broadcasting.
                int use_prev_block = 0;

                //If coords is not equal to next_get_coords we
                //should use the previous block.
                for(i=0; i<lastary->ndims; i++)
                    if(coords[i] != next_get_coords[j][i])
                    {
                        use_prev_block = 1;
                        break;
                    }
                if(!use_prev_block)
                {
                    //Reusing the PyObject.
                    obj_blocklist[j] = apply_blocklist[j];
                    apply_blocklist[j] = NULL;
                }
            }
        }while(notfinished);

        //Finally we call apply on the last block.
        for(j=0; j<lastary->base->ndims; j++)
            coords[j] = lastary->base->localblockdims[j] - 1;
        apply_ufunc(narys, apply_blocklist, ufunc_function,
                    ufunc_funcdata, lastary, coords);

        //Cleanup.
        for(j=0; j<narys; j++)
        {
            Py_XDECREF(recv_blocklist[j]);
            Py_XDECREF(apply_blocklist[j]);
            Py_XDECREF(obj_blocklist[j]);
        }
        for(j=0; j<nin; j++)
        {
            free(buffer[0][j]);
            free(buffer[1][j]);
        }
    }//End of the block method.
    else
    {
        //Receive buffers. One per array (single buffering).
        char *buffer[NPY_MAXARGS];

        //Allocate a buffer for every array.
        for(i=0; i<nin; i++)
            buffer[i] = malloc(views[i]->base->elsize);

        for(i=0; i<lastary->base->nelements; i++)
        {
            int not_inside_view = 0;
            for(j=0; j<nin; j++)
                not_inside_view += remote_get(buffer[j], views[j],
                                              lastary, coords);
            //If just one is outside we jump to next element.
            if(!not_inside_view)
            {
                npy_intp one = 1;
                PyObject *arytuple = PyTuple_New(narys);
                for(j=0; j<nin; j++)
                    PyTuple_SET_ITEM(arytuple, j,
                                     PyArray_SimpleNewFromData(1,
                                     &one, views[j]->base->dtype,
                                     buffer[j]));

                for(j=nin; j<narys; j++)
                    PyTuple_SET_ITEM(arytuple, j,
                                     PyArray_SimpleNewFromData(1,
                                     &one, views[j]->base->dtype,
                                     views[j]->base->data +
                                     i*views[j]->base->elsize));

                //Call the python operator.
                PyObject *t = PyObject_CallObject(PyOp, arytuple);
                Py_XDECREF(t);
                Py_DECREF(arytuple);
            }

            //Iterate coords one element.
            for(j=lastary->base->ndims-1; j >= 0; j--)
            {
                if(++coords[j] >= lastary->base->localdims[j])
                    coords[j] = 0;
                else
                    break;
            }
        }
        //Cleanup
        for(i=0; i<nin; i++)
            free(buffer[i]);
    }

    //Clean up
    Py_DECREF(ufunc_obj);
    for(i=0; i<narys; i++)
    {
        if(views[i]->base->ndims == 0)//Scalar dummy.
        {
            free(views[i]->base);
            free(views[i]);
        }
    }
#ifdef DNPY_TIME
    gettimeofday(&tv, &tz);
    total_time = (long long)tv.tv_usec + ((long long)tv.tv_sec)*1000000 - total_time_tmp;

    printf("total time: %llu us\n", total_time);
    printf("total remote get: %llu us\n", total_rget_time);
    printf("total apply: %llu us\n", total_apply_time);
    printf("total admin: %llu us\n", total_time-total_rget_time-total_apply_time);
#endif

} /* do_UFUNC */

/*===================================================================
 *
 * Message handler for UFUNC_REDUCE.
 * out_ary_uid is the output array uid or scalar datatype.
 * Private.
 */
static void do_UFUNC_REDUCE(npy_intp in_ary_uid,
                            npy_intp out_ary_uid,
                            int out_scalar_dtype,
                            int out_scalar_dtypesize,
                            char* out_scalar_address,
                            int axis, char* op)
{
    int i,j;
    dndview *in_view = get_dndview(in_ary_uid);
    dndarray *in_ary = in_view->base;
    PyObject *local_scalar, *tmpobj, *tmp_scalar;

    //Get Python function.
    PyObject *PyOp = PyObject_GetAttrString(ufunc_module, op);
    if(PyOp == NULL)
    {
        fprintf(stderr, "UFUNC_REDUCE: unknown operator: %s\n", op);
        MPI_Abort(MPI_COMM_WORLD, -1);
    }

    //At the moment we don't support views.
    if(in_view->alterations != 0)
    {
        fprintf(stderr, "UFUNC_REDUCE: No view support\n");
        MPI_Abort(MPI_COMM_WORLD, -1);
    }

    //If output is a scalar we just use this simple method.
    if(in_ary->ndims == 1)
    {
        //Print debug info.
        #ifdef DISTNUMPY_DEBUG
            printf("UFUNC_REDUCE (%s) on - in: %ld, axis: %d, "
                   "returning a scalar at address: %p\n", op,
                   (long)in_ary_uid, axis, out_scalar_address);
        #endif

        //The master should use the final output address.
        if(myrank == 0)
            local_scalar = PyArray_SimpleNewFromData(0, NULL,
                           out_scalar_dtype, out_scalar_address);
        else//And the rest just use a new location.
            local_scalar = PyArray_SimpleNew(0, NULL, out_scalar_dtype);

        if(in_ary->localdims[0] > 0)
        {
            tmpobj = PyArray_SimpleNewFromData(1, &in_ary->nelements,
                                               in_ary->dtype,
                                               in_ary->data);
            //Compute local reduce.
            PyObject *t = PyObject_CallMethod(PyOp, "reduce",
                                              "(O,i,O,O)", tmpobj, axis,
                                              Py_None, local_scalar);
            Py_XDECREF(t);

            //Send local output scalar to the master process.
            if(myrank > 0)
                MPI_Send(PyArray_DATA(local_scalar),
                         out_scalar_dtypesize, MPI_BYTE, 0, 0,
                         MPI_COMM_WORLD);
            Py_DECREF(tmpobj);
        }
        if(myrank == 0)
        {
            //Get init scalar. If the scalar is not located on the
            //master we have to get it.
            if(in_ary->localdims[0] == 0)
            {
                int zero = 0;
                MPI_Recv(PyArray_DATA(local_scalar),
                         out_scalar_dtypesize, MPI_BYTE,
                         cart2rank(1, &zero), 0, MPI_COMM_WORLD,
                         MPI_STATUS_IGNORE);
            }

            //We need a tmp scalar for receiving scalars.
            tmp_scalar = PyArray_SimpleNew(0, NULL, out_scalar_dtype);

            //For all ranks in dimension 'axis', we reduces the scalars
            //to local_scalar at the master.
            for(i=1; i < cart_dim_sizes[0][0]; i++)
            {
                if(0 == dnumroc(in_ary->dims[0], blocksize,
                                i, cart_dim_sizes[0][0]))
                    continue;

                MPI_Recv(PyArray_DATA(tmp_scalar), out_scalar_dtypesize,
                         MPI_BYTE, cart2rank(1, &i), 0, MPI_COMM_WORLD,
                         MPI_STATUS_IGNORE);
                PyObject *t = PyObject_CallFunction(PyOp, "(O,O,O)",
                                                    local_scalar,
                                                    tmp_scalar,
                                                    local_scalar);
                Py_DECREF(t);
            }
            Py_DECREF(tmp_scalar);
        }
        Py_DECREF(local_scalar);
        Py_DECREF(PyOp);
        return;
    }

    dndview *out_view = get_dndview(out_ary_uid);
    dndarray *out_ary = out_view->base;

    //Print debug info.
    #ifdef DISTNUMPY_DEBUG
        printf("UFUNC_REDUCE (%s) on - in: %ld, out: %ld, axis: %d\n", op,
               (long)in_ary_uid, (long)out_ary_uid, axis);
    #endif

    //At the moment we don't support views.
    if(out_view->alterations != 0)
    {
        fprintf(stderr, "UFUNC_REDUCE: No view support\n");
        MPI_Abort(MPI_COMM_WORLD, -1);
    }

    //There is two kind of roles for MPI-processes. They can be
    //receivers or senders. The receivers are receiving the sub-results
    //computed by the senders.
    //All MPI-processes with zero-coordinate in the axis-dimension are
    //receivers and all other MPI-processes are senders.
    //The senders should send its sub-results to the receiver with
    //identical coordinates beside zero in the axis-dimension.

    PyObject *tmp = PyArray_SimpleNewFromData(in_ary->ndims,
                                              in_ary->localdims,
                                              in_ary->dtype,
                                              in_ary->data);

    //All MPI-processes computes its own sub-result.
    PyObject *subres = PyObject_CallMethod(PyOp, "reduce", "(O,i)",
                                           tmp, axis);
    //Clean up.
    Py_DECREF(tmp);

    int cart_coords[NPY_MAXDIMS];
    rank2cart(in_ary->ndims, myrank, cart_coords);

    if(cart_coords[axis] == 0)//MPI-process is a receiver.
    {
        npy_intp recv_dims[NPY_MAXDIMS];
        memset(recv_dims, 0, out_ary->ndims*sizeof(intp));

        //Handle one sender at a time.
        for(cart_coords[axis]=1;
            cart_coords[axis] < cart_dim_sizes[in_ary->ndims-1][axis];
            cart_coords[axis]++)
        {
            int rank = cart2rank(in_ary->ndims, cart_coords);

            //Compute the dim of the receiving array.
            int count = 0;
            for(i=0; i < in_ary->ndims; i++)
                if(i != axis)
                    recv_dims[count++] = dnumroc(in_ary->dims[i],
                                blocksize, cart_coords[i],
                                cart_dim_sizes[in_ary->ndims-1][i]);

            //Receive the sub-result from the sender into 'tmpres'.
            PyObject *tmpres = PyArray_SimpleNew(out_ary->ndims,
                               recv_dims, out_ary->dtype);
            MPI_Recv(PyArray_DATA(tmpres), PyArray_SIZE(tmpres) *
                     out_ary->elsize, MPI_BYTE, rank, 0,
                     MPI_COMM_WORLD, MPI_STATUS_IGNORE);

            //We use ufunc to "reduce" the two arrays over the
            //non-existing 'axis' dimension.
            PyObject *t = PyObject_CallFunction(PyOp, "(O,O,O)", subres,
                                                tmpres, subres);
            Py_DECREF(t);
            Py_DECREF(tmpres);
        }

        npy_intp lcoords[NPY_MAXDIMS];
        memset(lcoords, 0, in_ary->ndims * sizeof(npy_intp));
        int notfinished = 1;
        if(PyArray_SIZE(subres) > 0)
            while(notfinished)
            {
                remote_blockput(subres, in_ary, out_ary, axis, lcoords);

                //Iterate coords one element.
                for(j=in_ary->ndims-1; j >= 0; j--)
                {
                    //We ignore the 'axis' dimension - it stays on zero.
                    if(j != axis)
                    {
                        if(++lcoords[j] >= in_ary->localblockdims[j])
                            lcoords[j] = 0;
                        else
                            break;
                    }
                }
                notfinished = 0;
                for(i=0; i<in_ary->ndims; i++)
                    if(lcoords[i] > 0)
                        notfinished = 1;
            }
    }
    else//MPI-process is a sender.
    {
        //Send subresult to receiver.
        cart_coords[axis] = 0;
        MPI_Send(PyArray_DATA(subres),
                 PyArray_SIZE(subres) * PyArray_ITEMSIZE(subres),
                 MPI_BYTE, cart2rank(in_ary->ndims, cart_coords), 0,
                 MPI_COMM_WORLD);
    }

    //Cleanup
    MPI_Barrier(MPI_COMM_WORLD);
    Py_DECREF(subres);
    Py_DECREF(PyOp);

} /* do_UFUNC_REDUCE */

/*===================================================================
 *
 * Message handler for ZERO_FILL.
 * Private.
 */
static void do_ZEROFILL(dndview *view)
{
    //Print debug info.
    #ifdef DISTNUMPY_DEBUG
        printf("ZEROFILL - uid: %ld\n", (long)view->uid);
    #endif

    //Check if the view covers the whole array.
    if(view->alterations != 0)
    {
        fprintf(stderr, "ZEROFILL with views not implemented\n");
        MPI_Abort(MPI_COMM_WORLD, -1);
    }

    memset(view->base->data, 0, view->base->nelements *
                                view->base->elsize);
} /* do_ZEROFILL */

/*===================================================================
 *
 * Message handler for DIAGONAL.
 * Private.
 */
static void do_DIAGONAL(dndview *in, dndview *out, int offset,
                        int axis1, int axis2)
{
    int mypcoord, pcoord[2];
    npy_intp lcoord, gcoord, bsize;
    MPI_Datatype arraytype;

    //Print debug info.
    #ifdef DISTNUMPY_DEBUG
        printf("DIAGONAL - in: %ld, out: %ld, offset: %d, axis: "
               "(%d, %d)\n",(long)in->uid, (long)out->uid, offset,
                            axis1, axis2);
    #endif

    //Find process coord in output vector.
    rank2cart(1, myrank, &mypcoord);

    //Iterate over the output vector.
    npy_intp nblocks = out->base->localblockdims[0];
    char *out_data = out->base->data;
    npy_intp ldims[2], index[2];
    for(lcoord=0; lcoord < nblocks; lcoord++)
    {

        //Convert local coord to global coord.
        gcoord = mypcoord +           //process offset
                 lcoord *             //block index
                 cart_dim_sizes[0][0];//jump per block

        //Find process cartesian coords of remote MPI process.
        pcoord[0] = gcoord % cart_dim_sizes[1][0];
        pcoord[1] = gcoord % cart_dim_sizes[1][1];

        //Find remote local size.
        ldims[0] = dnumroc(in->base->dims[0], blocksize, pcoord[0],
                           cart_dim_sizes[1][0]);
        ldims[1] = dnumroc(in->base->dims[1], blocksize, pcoord[1],
                           cart_dim_sizes[1][1]);

        //Find remote index (non-block index).
        index[0] = gcoord / cart_dim_sizes[1][0] * blocksize;
        index[1] = gcoord / cart_dim_sizes[1][1] * blocksize;

        //Find blocksize.
        bsize = MIN(ldims[0] - index[0], ldims[1] - index[1]);
        if(bsize > blocksize)
            bsize = blocksize;
        else if(bsize <= 0)
            break;

        //Find remote offset from window start.
        npy_intp offset = index[0] * ldims[1] + index[1];

        //Convert to a MPI rank number.
        int rank = cart2rank(2, pcoord);

        //Setup a matching MPI datatype.
        MPI_Type_vector(bsize, 1, ldims[1]+1, in->base->mpi_dtype, 
                        &arraytype);
        MPI_Type_commit(&arraytype);

        //Transfer element.
        MPI_Win_lock(MPI_LOCK_SHARED, rank, MPI_MODE_NOCHECK, in->base->comm_win);
        MPI_Get(out_data, bsize, in->base->mpi_dtype, rank, 
                offset * in->base->elsize, 1, arraytype, 
                in->base->comm_win);
        MPI_Win_unlock(rank, in->base->comm_win);

        //Clean up
        MPI_Type_free(&arraytype);

        //Iterate output data one block.
        out_data += bsize * in->base->elsize;
    }
    //Sync.
    MPI_Barrier(MPI_COMM_WORLD);
}

/*===================================================================
 *
 * Message handler for MATMUL.
 * Private.
 */
static void do_MATMUL(dndview *A, dndview *B, dndview *C)
{
    int i, j;
    //Print debug info.
    #ifdef DISTNUMPY_DEBUG
        printf("do_MATMUL - A: %ld, B: %ld, C: %ld\n", (long)A->uid,
                                                       (long)B->uid,
                                                       (long)C->uid);
    #endif

    //Check if the views covers the whole array.
    if(A->alterations != 0 || B->alterations != 0 ||
       C->alterations != 0)
    {
        fprintf(stderr, "Matrix multiplication with views not "
                        "implemented\n");
        MPI_Abort(MPI_COMM_WORLD, -1);
    }
    //We only support doubles at the moment.
    if(A->base->dtype != NPY_DOUBLE || B->base->dtype != NPY_DOUBLE ||
       C->base->dtype != NPY_DOUBLE)
    {
        fprintf(stderr, "At the moment matrix multiplication only "
                        "supports doubles.\n");
        MPI_Abort(MPI_COMM_WORLD, -1);
    }

    //Global index of row and column that hold current row and column
    //for rank-1 update.
    int icur[2] = {0,0};
    //local index of row and column for rank-1 update
    int ii = 0;
    int jj = 0;
    //Current blocksize.
    int iwrk = MIN(blocksize, A->base->dims[1]);
    int kk;

    //Find by MPI-process coord.
    int mycoord[2], tcoord[2];
    rank2cart(2, myrank, mycoord);
    int *cart_dim = cart_dim_sizes[1];

    //malloc temp space
    double *temp  = malloc(A->base->elsize);
    double *work0 = malloc(A->base->localdims[0] * iwrk *
                           A->base->elsize);
    double *work1 = malloc(iwrk * B->base->localdims[1] *
                           A->base->elsize);

    //Leading dimensions.
    int lda = MAX(A->base->localdims[1], 1);
    int ldb = MAX(B->base->localdims[1], 1);
    int ldc = MAX(C->base->localdims[1], 1);

    for (kk=0; kk<A->base->dims[1]; kk+=iwrk)
    {
        iwrk = MIN(blocksize, A->base->dims[1] - kk);

        //Copy iwrk'th column to work-array.
        if(mycoord[1] == icur[1])
        {
            double *Adata = ((double *)A->base->data)+jj;
            for(i=0; i<A->base->localdims[0]; i++)
                for(j=0; j<iwrk; j++)
                    work0[i*iwrk+j] = Adata[i*lda+j];
        }
        //Copy iwrk'th row to work-array.
        if(mycoord[0] == icur[0])
        {
                double *Bdata = ((double *)B->base->data)+ldb*ii;
                for(i=0; i<iwrk; i++)
                    for(j=0; j<B->base->localdims[1]; j++)
                        work1[i*B->base->localdims[1]+j] = Bdata[i*ldb+j];
        }
        if(cart_dim[1] > 1)
        {
            //Ring-broadcast horizontal.
            if(mycoord[1] != icur[1])
            {
                //We have to recv since we are not icur.
                tcoord[0] = mycoord[0];
                tcoord[1] = (mycoord[1]-1+cart_dim[1])%cart_dim[1];
                MPI_Recv(work0, A->base->localdims[0] * iwrk,
                         A->base->mpi_dtype, cart2rank(2,tcoord),
                         MPI_ANY_TAG, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            }
            //Send the iwrk'th column to the right neighbour but
            //make sure we don't wrap around.
            if((mycoord[1]+1)%cart_dim[1] != icur[1])
            {
                tcoord[0] = mycoord[0];
                tcoord[1] = (mycoord[1]+1)%cart_dim[1];
                MPI_Send(work0, A->base->localdims[0] * iwrk,
                         A->base->mpi_dtype, cart2rank(2,tcoord), 0,
                         MPI_COMM_WORLD);
            }
        }
        if(cart_dim[0] > 1)
        {
            //Ring-broadcast vertical.
            if(mycoord[0] != icur[0])
            {
                //We have to recv, since we a not icur.
                tcoord[0] = (mycoord[0]-1+cart_dim[0])%cart_dim[0];
                tcoord[1] = mycoord[1];
                MPI_Recv(work1, iwrk * B->base->localdims[1],
                         A->base->mpi_dtype, cart2rank(2,tcoord),
                         MPI_ANY_TAG, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            }
            //Send the iwrk'th row to the lower neighbour but
            //make sure we don't wrap around.
            if((mycoord[0]+1)%cart_dim[0] != icur[0])
            {
                tcoord[0] = (mycoord[0]+1)%cart_dim[0];
                tcoord[1] = mycoord[1];
                MPI_Send(work1, iwrk * B->base->localdims[1],
                         A->base->mpi_dtype, cart2rank(2,tcoord), 0,
                         MPI_COMM_WORLD);
            }
        }

        //If available we use the BLAS library else we fall back to the
        //dot-function in numpy.
        if(cblas_dgemm_func != NULL)
        {
            cblas_dgemm_func(CblasRowMajor, CblasNoTrans, CblasNoTrans,
                             C->base->localdims[0],C->base->localdims[1],
                             iwrk, 1.0,
                             work0, iwrk,
                             work1, MAX(B->base->localdims[1],1),
                             1.0, (double *)C->base->data, ldc);
        }
        else
        {
            PyArray_Descr *descr = PyArray_DescrFromType(C->base->dtype);
            PyArray_DotFunc *dot = descr->f->dotfunc;
            npy_intp m,n;
            for(n=0; n < C->base->localdims[0]; n++)
                for(m=0; m < C->base->localdims[1]; m++)
                {
                    double tmp;
                    dot(work0 + n * iwrk, sizeof(double), work1 + m,
                        MAX(B->base->localdims[1],1) * sizeof(double),
                        &tmp, iwrk, NULL);
                    ((double *)C->base->data)[m+n*C->base->localdims[1]] += tmp;
                }
            Py_DECREF(descr);
        }

        //Iterate indexes.
        if(mycoord[0] == icur[0])
            ii += iwrk;
        if(mycoord[1] == icur[1])
            jj += iwrk;
        icur[0] = (icur[0]+1)%cart_dim[0];
        icur[1] = (icur[1]+1)%cart_dim[1];
    }
    //Cleanup
    free(work0);
    free(work1);
    free(temp);
} /* do_MATMUL */


/*NUMPY_API
 *===================================================================
 * Create dndarray.
 * Public
 */
static intp
dnumpy_create_dndarray(int ndims, intp dims[NPY_MAXDIMS], int dtype,
                       intp dtypesize)
{
    int i;
    assert(ndims > 0);
    assert(ndims < NPY_MAXDIMS);

    //Create dndarray.
    dndarray *newarray = malloc(sizeof(dndarray));

    newarray->dtype = dtype;
    newarray->elsize = dtypesize;
    newarray->ndims = ndims;
    newarray->refcount = 1;
    for(i=0; i<ndims; i++)
        newarray->dims[i] = dims[i];

    //Create dndview. NB: the base will have to be set when 'newarray'
    //has found its final resting place. (Done by put_dndarray).
    dndview *newview = malloc(sizeof(dndview));
    newview->uid = ++uid_count;
    newview->nslice = ndims;
    newview->ndims = ndims;
    newview->alterations = 0;
    for(i=0; i<ndims; i++)
    {
        //Default the view will span over the whole array.
        newview->slice[i].start = 0;
        newview->slice[i].step = 1;
        newview->slice[i].nsteps = dims[i];
    }

    //Tell slaves about the new array
    msg[0] = DNPY_CREATE_ARRAY;
    memcpy(&msg[1], newarray, sizeof(dndarray));
    memcpy(((char *) &msg[1]) + sizeof(dndarray), newview,
           sizeof(dndview));

    *(((char *) &msg[1])+sizeof(dndarray)+sizeof(dndview)) = DNPY_MSG_END;

    msg2slaves(msg,2*sizeof(npy_intp)+sizeof(dndarray)+sizeof(dndview));

    //The master should also do the operation
    #ifdef DISTNUMPY_DEBUG
        printf("Rank 0 received msg: ");
    #endif
    do_CREATE_ARRAY(newarray, newview);

    //Freeup memory.
    free(newarray);
    free(newview);

    return uid_count;
} /* dnumpy_create_dndarray */

/*NUMPY_API
 *===================================================================
 * Destroy dndarray.
 * Public
 */
static void
dnumpy_destroy_dndarray(intp uid)
{
    //Get arrray structs.
    dndview *ary = get_dndview(uid);

    //Tell slaves about the destruction
    msg[0] = DNPY_DESTROY_ARRAY;
    msg[1] = ary->uid;
    msg[2] = DNPY_MSG_END;
    msg2slaves(msg,3 * sizeof(npy_intp));

    //The master should also do the operation
    #ifdef DISTNUMPY_DEBUG
        printf("Rank 0 received msg: ");
    #endif
    do_DESTROY_ARRAY(ary->uid);

} /* dnumpy_destroy_dndarray */


/*NUMPY_API
 *===================================================================
 * Create a new view based on the 'org_view'.
 * Public
 */
static int
dnumpy_create_dndview(intp org_view_uid, int nslice,
                      dndslice slice[NPY_MAXDIMS])
{
    //Get arrray structs.
    dndview *orgview = get_dndview(org_view_uid);

    //Create new view based on 'org_view' and the 'slice'.
    dndview newview;
    newview.uid = ++uid_count;
    newview.ndims = 0;
    newview.alterations = 0;

    //Merging the two views.
    int si = 0; //slice index.
    int ni = 0; //new index.
    int oi = 0; //old index.
    int di = 0; //dim index.
    while(si < nslice || oi < orgview->nslice)
    {
        //If we come to the end of the slices, that happens
        //if not all dimensions is included in the 'slices', we will
        //use the whole dimension.
        int vs = (si < nslice)?1:0;//Valid slice.

        //If dimension is invisible we will just copy it to 'newview'.
        if(oi < orgview->nslice &&
           orgview->slice[oi].nsteps == SingleIndex)
        {
            memcpy(&newview.slice[ni], &orgview->slice[oi],
                   sizeof(dndslice));
            ni++; oi++; di++;
            newview.alterations |= DNPY_NDIMS;
        }
        //A single index makes the dimension invisible.
        else if(vs && slice[si].nsteps == SingleIndex)
        {
            //If dimension is a Pseudo-dimension then just go to next
            //dimension.
            if(orgview->slice[oi].nsteps == PseudoIndex)
            {
                si++; oi++;
            }
            else
            {//Copy single index to 'newview'.
                newview.slice[ni].step = 0;
                newview.slice[ni].nsteps = SingleIndex;
                newview.slice[ni].start = orgview->slice[oi].start +
                                          (vs?slice[si].start:0) *
                                          orgview->slice[oi].step;
                si++; ni++; oi++; di++;
                newview.alterations |= DNPY_NDIMS;
            }
        }
        //If a extra pseudo index should be added we just copy the
        //slice to 'newview'.
        else if(vs && slice[si].nsteps == PseudoIndex)
        {
            memcpy(&newview.slice[ni], &slice[si],
                   sizeof(dndslice));
            ni++; si++;
            newview.ndims++;
            newview.alterations |= DNPY_NDIMS;
        }
        else if(orgview->slice[oi].nsteps == PseudoIndex)
        {
            memcpy(&newview.slice[ni], &orgview->slice[oi],
                   sizeof(dndslice));
            ni++; oi++; si++;
            newview.ndims++;
            newview.alterations |= DNPY_NDIMS;
        }
        //If no special slices we just merge the two views.
        else
        {
            if(vs)
            {
                newview.slice[ni].start = orgview->slice[oi].start +
                                           slice[si].start *
                                           orgview->slice[oi].step;
                newview.slice[ni].step = slice[si].step *
                                          orgview->slice[oi].step;
                newview.slice[ni].nsteps = slice[si].nsteps;
            }
            else
                memcpy(&newview.slice[ni], &orgview->slice[oi],
                       sizeof(dndslice));

            if(newview.slice[ni].step > 1)
                newview.alterations |= DNPY_STEP | DNPY_NSTEPS;
            else if(newview.slice[ni].nsteps < orgview->base->dims[di])
            {
                newview.alterations |= DNPY_NSTEPS;
            }
            newview.ndims++;
            si++; ni++; oi++; di++;
        }
    }
    //Save the total number of sliceses for the new view.
    newview.nslice = ni;

    //Tell slaves about the new view.
    //NB: It is up to the slaves to add the newview.base adresse.
    msg[0] = DNPY_CREATE_VIEW;
    msg[1] = orgview->uid;
    memcpy(&msg[2], &newview, sizeof(dndview));
    *(((char *) &msg[2])+sizeof(dndview)) = DNPY_MSG_END;

    msg2slaves(msg, 3 * sizeof(npy_intp) + sizeof(dndview));

    //The master should also do the operation
    #ifdef DISTNUMPY_DEBUG
        printf("Rank 0 received msg: ");
    #endif

    do_CREATE_VIEW(orgview->uid, &newview);
    return uid_count;
} /* dnumpy_destroy_dndarray */


/*NUMPY_API
 *===================================================================
 * Assign the value to array at 'coordinate'.
 * 'coordinate' size must be the same as view->ndims.
 * Steals all reference to item. (Item is lost).
 * Public
 */
static int
dnumpy_dndarray_setitem(intp uid, intp coordinate[NPY_MAXDIMS],
                        PyObject *item)
{
    //Get arrray structs.
    dndview *view = get_dndview(uid);
    int ndims = view->base->ndims;
    int elsize = view->base->elsize;

    //Convert item to a compatible type.
    PyObject *item2 = PyArray_FROM_O(item);
    PyObject *citem2 = PyArray_Cast((PyArrayObject*)item2,
                                    view->base->dtype);

    //Cleanup and return error if the cast failed.
    if(citem2 == NULL)
    {
        Py_DECREF(item2);
        return -1;
    }

    //Tell slaves about the new item.
    msg[0] = DNPY_SET_ITEM;
    msg[1] = view->uid;
    memcpy(&msg[2], PyArray_DATA(citem2), elsize);
    memcpy(((char *) &msg[2]) + elsize, coordinate,
           sizeof(npy_intp) * ndims);
    *(((char *) &msg[2]) + elsize + sizeof(npy_intp) * ndims) = DNPY_MSG_END;

    msg2slaves(msg, 3 * sizeof(npy_intp) + elsize +
                    ndims * sizeof(npy_intp));

    //The master should also do the operation.
    #ifdef DISTNUMPY_DEBUG
        printf("Rank 0 received msg: ");
    #endif
    do_SET_ITEM(view, PyArray_DATA(citem2), coordinate);

    //Clean up.
    Py_DECREF(citem2);
    Py_DECREF(item2);

    return 0;//Succes
} /* dnumpy_dndarray_setitem */

/*NUMPY_API
 *===================================================================
 * Get a single value specified by 'coordinate' from the array.
 * 'coordinate' size must be the same as view->ndims.
 * Public
 */
static void
dnumpy_dndarray_getitem(char *retdata, intp uid,
                        intp coordinate[NPY_MAXDIMS])
{
    //Get arrray structs.
    dndview *view = get_dndview(uid);
    int ndims = view->base->ndims;

    //Tell slaves to send item.
    msg[0] = DNPY_GET_ITEM;
    msg[1] = view->uid;
    memcpy(&msg[2], coordinate, sizeof(npy_intp) * ndims);
    *(((char *) &msg[2]) + sizeof(npy_intp) * ndims) = DNPY_MSG_END;

    msg2slaves(msg, 3 * sizeof(npy_intp) + ndims * sizeof(npy_intp));

    //The master will retrive the whole array.
    #ifdef DISTNUMPY_DEBUG
        printf("Rank 0 received msg: ");
    #endif

    do_GET_ITEM(retdata, view, coordinate);
} /* dnumpy_dndarray_getitem */

/*NUMPY_API
 *===================================================================
 * Public
 * Apply ufunc on distributed arrays in arylist.
 * op is the python name of the ufunction.
 * Returns -1 on failure.
 */
static int
dnumpy_ufunc(PyArrayObject *arylist[NPY_MAXARGS], int narys,
             int nout_arys, char *op, int appropriate_function)
{
    int i;
    //Tell slaves about the ufunc operations.
    msg[0] = DNPY_UFUNC;
    msg[1] = narys;
    msg[2] = nout_arys;
    msg[3] = appropriate_function;
    msg[4] = 0; //Scalar datatype.
    msg[5] = 0; //Scalar datasize, zero if no scalar is in arylist.

    //Copy scalar if one is in the arylist.
    for(i=0; i<narys;i++)
        if(arylist[i]->dnduid == 0)
        {
            if(arylist[i]->nd != 0)
            {
                PyErr_SetString(PyExc_RuntimeError,
                                "ufunc - distributed and non-"
                                "distributed arrays do not mix. "
                                "Scalars is allowed though.\n");
                return -1;
            }
            msg[4] = arylist[i]->descr->type_num;
            msg[5] = arylist[i]->descr->elsize;
            memcpy(msg+6, arylist[i]->data, msg[5]);
            break;
        }

    //Copy arylist's uid to msg.
    for(i=0; i<narys;i++)
        memcpy(((char*)(msg+6))+msg[5]+i*sizeof(npy_intp),
                &arylist[i]->dnduid, sizeof(npy_intp));

    //Copy the string op.
    strcpy(((char*)(msg+6+msg[1]))+msg[5], op);

    *(((char*)(msg+6+msg[1]))+msg[5]+strlen(op)+1) = DNPY_MSG_END;

    msg2slaves(msg, (7+msg[1])*sizeof(npy_intp)+msg[5]+strlen(op)+1);

    //The master should also do the operation.
    #ifdef DISTNUMPY_DEBUG
        printf("Rank 0 received msg: ");
    #endif

    do_UFUNC((npy_intp*)(((char*)(msg+6))+msg[5]), msg[1], msg[2], msg[3],
             msg[4], msg[5], (char*) (msg+6), op);
    return 0;
} /* dnumpy_ufunc */

/*NUMPY_API
 *===================================================================
 * Public
 * Apply ufunc method "reduce" on distributed array in_ary over axis.
 * op is the python name of the ufunction.
 * Returns -1 on failure.
 */
static int
dnumpy_ufunc_reduce(PyArrayObject *in_ary, PyArrayObject *out_ary,
                    int axis, char *op)
{
    int i;
    dndview *in = get_dndview(in_ary->dnduid);

    //We do not support reduce on views.
    for(i=0; i<in->nslice;i++)
    {
        if(in->slice[i].nsteps == SingleIndex ||
           in->slice[i].nsteps == PseudoIndex ||
           in->slice[i].start != 0 || in->slice[i].step != 1 ||
           in->slice[i].nsteps != in->base->dims[i])
        {
            PyErr_SetString(PyExc_RuntimeError, "reduce on distributed "
                            "arrays must be whole arrays and not a "
                            "partial view of a underlying array.");
            return -1;
        }
    }
    if(out_ary->dnduid > 0)
    {
        dndview *out = get_dndview(out_ary->dnduid);
        //Check if the views covers the whole array.
        if(in->alterations != 0 || out->alterations != 0)
        {
            PyErr_SetString(PyExc_RuntimeError,
                            "reduce on distributed arrays must"
                            "be whole arrays and not a partial"
                            " view of a underlying array.");
            return -1;
        }
    }

    //Tell slaves
    msg[0] = DNPY_UFUNC_REDUCE;
    msg[1] = in_ary->dnduid;
    msg[2] = out_ary->dnduid;
    //If out_ary is a scalar we will send the datatype and datatypesize.
    if(out_ary->dnduid == 0)
    {
        msg[3] = PyArray_TYPE(out_ary);
        msg[4] = PyArray_ITEMSIZE(out_ary);
    }
    msg[5] = axis;
    strcpy((char*)(msg+6), op);
    *(((char*)(msg+6))+strlen(op)+1) = DNPY_MSG_END;

    msg2slaves(msg, 7*sizeof(npy_intp)+strlen(op)+1);

    //The master should also do the operation
    #ifdef DISTNUMPY_DEBUG
        printf("Rank 0 received msg: ");
    #endif

    do_UFUNC_REDUCE(msg[1], msg[2], msg[3], msg[4],
                    out_ary->data, axis, op);
    return 0;
} /* dnumpy_ufunc_reduce */

/*NUMPY_API
 *===================================================================
 * Public
 * Fill the distributed array/view with zeroes.
 */
static void
dnumpy_zerofill(intp ary_uid)
{
    //Get arrray structs.
    dndview *view = get_dndview(ary_uid);

    //Tell slaves
    msg[0] = DNPY_ZEROFILL;
    msg[1] = view->uid;
    msg[2] = DNPY_MSG_END;
    msg2slaves(msg, 3 * sizeof(npy_intp));

    //The master should also do the operation.
    #ifdef DISTNUMPY_DEBUG
        printf("Rank 0 received msg: ");
    #endif
    do_ZEROFILL(view);
} /* dnumpy_zerofill */


/*NUMPY_API
 *===================================================================
 * Public
 * Returns a new 1-dim dist array filled with the diagonal from 'uid'.
 * Returns NULL on failure.
 */
static PyObject*
dnumpy_diagonal(intp uid, int offset, int axis1, int axis2)
{
    PyObject *ret;
    intp n1, n2, start, stop, step, count;
    //Get arrray structs.
    dndview *view = get_dndview(uid);

    if(axis1 != 0 && axis2 != 1)
    {
        PyErr_Format(PyExc_ValueError, "axis1 and axis2 must be the "
                     "defualt values 0 and 1 on a distributed array.");
        return NULL;
    }
    if(view->ndims != 2)
    {
        PyErr_Format(PyExc_ValueError, "if array is distributed it "
                     "must have exactly two dimensions.");
        return NULL;
    }

    if(offset != 0)
    {
        PyErr_Format(PyExc_ValueError, "if array is distributed offset "
                     "must be zero.");
        return NULL;
    }

    if(view->alterations != 0)
    {
        PyErr_SetString(PyExc_RuntimeError, "diagonal on a distributed "
                        "array must be a whole array and not a "
                        "partial view of a underlying array.");
        return NULL;
    }

    //Compute the size of the diagonal.
    n1 = view->base->dims[0];
    n2 = view->base->dims[1];
    step = n2 + 1;
    if (offset < 0) {
        start = -n2 * offset;
        stop = MIN(n2, n1+offset)*(n2+1) - n2*offset;
    }
    else {
        start = offset;
        stop = MIN(n1, n2-offset)*(n2+1) + offset;
    }
    //count = ceil((stop-start)/step)
    count = ((stop-start) / step) + (((stop-start) % step) != 0);

    //Create the returning distributed vector.
    ret = PyArray_New(&PyArray_Type, 1, &count, view->base->dtype,
                      NULL, NULL, 0,
                      NPY_WRITEABLE | NPY_CARRAY | DNPY_DISTRIBUTED,
                      NULL);

    //Tell slaves
    msg[0] = DNPY_DIAGONAL;
    msg[1] = view->uid;
    msg[2] = PyArray_DNDUID(ret);
    msg[3] = offset;
    msg[4] = axis1;
    msg[5] = axis2;
    msg[6] = DNPY_MSG_END;
    msg2slaves(msg, 7 * sizeof(npy_intp));

    //The master should also do the operation.
    #ifdef DISTNUMPY_DEBUG
        printf("Rank 0 received msg: ");
    #endif
    do_DIAGONAL(view, get_dndview(PyArray_DNDUID(ret)), offset, axis1,
                axis2);

    return ret;
} /* dnumpy_diagonal */


/*NUMPY_API
 * ===================================================================
 * Public
 * Initialization of distnumpy.
 */
static void
dnumpy_init(void)
{
    int i,j,s;
    //Initilization of MPI.
    MPI_Init(NULL, NULL);

    MPI_Comm_rank(MPI_COMM_WORLD, &myrank);
    MPI_Comm_size(MPI_COMM_WORLD, &worldsize);

    //Initilization of cart_dim_strides.
    for(i=0; i<NPY_MAXDIMS; i++)
    {
        //Find a balanced distributioin of processes per direction
        //and save it in the a global struct.
        cart_dim_sizes[i] = malloc(NPY_MAXDIMS*sizeof(int));
        memset(cart_dim_sizes[i], 0, sizeof(int)*NPY_MAXDIMS);
        MPI_Dims_create(worldsize, i+1, cart_dim_sizes[i]);

        //Find number of used cartesian dims.
        cart_used_ndims[i] = 1;
        for(j=1; j<=i;j++)
            if(cart_dim_sizes[i][j] > 1)
                cart_used_ndims[i]++;

        //Allocate cartesian information.
        cart_dim_strides[i] = malloc(cart_used_ndims[i]*sizeof(int));
        memset(cart_dim_strides[i], 0, cart_used_ndims[i]*sizeof(int));

        //Compute strides for all dims. Using row-major like MPI.
        //A 2x2 process grid looks like:
        //    coord (0,0): rank 0.
        //    coord (0,1): rank 1.
        //    coord (1,0): rank 2.
        //    coord (1,1): rank 3.
        for(j=0; j<cart_used_ndims[i]; j++)
        {
            int stride = 1;
            for(s=j+1; s<cart_used_ndims[i]; s++)
                stride *= cart_dim_sizes[i][s];
            cart_dim_strides[i][j] = stride;
        }
    }
} /* dnumpy_init */


/*NUMPY_API
 *===================================================================
 * Public
 * C = A * B.
 * All elements in C must be zero.
 */
static void
dnumpy_matrix_multiplication(intp Auid, intp Buid, intp Cuid)
{
    //Get arrray structs.
    dndview *A = get_dndview(Auid);
    dndview *B = get_dndview(Buid);
    dndview *C = get_dndview(Cuid);

    //Tell slaves
    msg[0] = DNPY_MATMUL;
    msg[1] = Auid;
    msg[2] = Buid;
    msg[3] = Cuid;
    msg[4] = DNPY_MSG_END;
    msg2slaves(msg, 5 * sizeof(npy_intp));

    //The master should also do the operation.
    #ifdef DISTNUMPY_DEBUG
        printf("Rank 0 received msg: ");
    #endif

    do_MATMUL(A, B, C);
} /* dnumpy_matrix_multiplication */


/*NUMPY_API
 * ===================================================================
 * Public
 * De-initialization of distnumpy.
 */
static void
dnumpy_exit()
{
    int i;
    if(myrank == 0)
    {
        //Shutdown slaves
        msg[0] = DNPY_SHUTDOWN;
        msg[1] = DNPY_MSG_END;
        msg2slaves(msg, 2 * sizeof(npy_intp));
        #ifdef DISTNUMPY_DEBUG
            printf("Rank 0 received msg: SHUTDOWN\n");
        #endif
    }
    //Free Cartesian Information.
    for(i=0; i < NPY_MAXDIMS; i++)
    {
        free(cart_dim_strides[i]);
        free(cart_dim_sizes[i]);
    }

    int nleaks = 0;
    for(i=0; i < DNPY_MAX_NARRAYS; i++)
        if(dndviews_uid[i] != 0)
            nleaks++;

    if(nleaks > 0)
        printf("DistNumPy - Warning %d distributed arrays didn't get "
               "deallocated.\n", nleaks);

    MPI_Finalize();
} /* dnumpy_exit */

/*NUMPY_API
 * ===================================================================
 * Public
 * From this point on the master will continue with the pyton code
 * and the slaves will stay in C.
 * If returning False the Python must call sys.exit(0) immediately.
 */
static PyObject *
dnumpy_master_slave_split()
{
    if(myrank == 0)
    {
        //Check for user-defined block size.
        char *bs = getenv("DNPY_BLOCKSIZE");
        if(bs == NULL)
            blocksize = DNPY_BLOCKSIZE;
        else
            blocksize = atoi(bs);

        if(blocksize <= 0)
        {
            fprintf(stderr, "User-defined blocksize must be greater "
                            "than zero\n");
            MPI_Abort(MPI_COMM_WORLD, -1);
        }

        //Send block size to clients.
        npy_intp msg[3];
        msg[0] = DNPY_INIT_BLOCKSIZE;
        msg[1] = blocksize;
        msg[2] = DNPY_MSG_END;
        msg2slaves(msg, 3 * sizeof(npy_intp));

        #ifdef DISTNUMPY_DEBUG
        printf("Rank 0 received msg: ");
        #endif
        do_INIT_BLOCKSIZE(msg[1]);
        return Py_True;
    }

    int shutdown = 0;
    while(shutdown == 0)//Work loop
    {
        char *t1, *t2, *t3;
        npy_intp d1, d2, d3, d4, d5;
        dndview *ary, *ary2, *ary3;
        MPI_Barrier(MPI_COMM_WORLD);
        //Receive message from master.
        MPI_Bcast(msg, DNPY_MAX_MSG_SIZE, MPI_BYTE, 0, MPI_COMM_WORLD);
        char *msg_data = (char *) &msg[1];
        #ifdef DISTNUMPY_DEBUG
            printf("Rank %d received msg: ", myrank);
        #endif
        switch(msg[0])
        {
            case DNPY_INIT_BLOCKSIZE:
                do_INIT_BLOCKSIZE(*((npy_intp*)msg_data));
                break;
            case DNPY_CREATE_ARRAY:
                t1 = msg_data + sizeof(dndarray);
                do_CREATE_ARRAY((dndarray*) msg_data, (dndview*) t1);
                break;
            case DNPY_DESTROY_ARRAY:
                do_DESTROY_ARRAY(*((npy_intp*)msg_data));
                break;
            case DNPY_CREATE_VIEW:
                d1 = *((npy_intp*)msg_data);
                t1 = msg_data+sizeof(npy_intp);
                do_CREATE_VIEW(d1, (dndview*) t1);
                break;
            case DNPY_SHUTDOWN:
                #ifdef DISTNUMPY_DEBUG
                    printf("SHUTDOWN\n");
                #endif
                shutdown = 1;
                break;
            case DNPY_SET_ITEM:
                ary = get_dndview(*((npy_intp*)msg_data));
                t1 = msg_data+sizeof(npy_intp);
                t2 = t1+ary->base->elsize;
                do_SET_ITEM(ary, t1, (npy_intp*) t2);
                break;
            case DNPY_GET_ITEM:
                ary = get_dndview(*((npy_intp*)msg_data));
                t1 = msg_data+sizeof(npy_intp);
                do_GET_ITEM(NULL, ary, (npy_intp*) t1);
                break;
            case DNPY_UFUNC:
                d1 = *((npy_intp*)msg_data);
                d2 = *(((npy_intp*)msg_data)+1);
                d3 = *(((npy_intp*)msg_data)+2);
                d4 = *(((npy_intp*)msg_data)+3);
                d5 = *(((npy_intp*)msg_data)+4);
                t1 = msg_data+sizeof(npy_intp)*5;
                t2 = t1+d5;
                t3 = t2+d1*sizeof(npy_intp);
                do_UFUNC((npy_intp *)t2,d1,d2,d3,d4,d5,t1,t3);
                break;
            case DNPY_UFUNC_REDUCE:
                d1 = *((npy_intp*)msg_data);
                d2 = *(((npy_intp*)msg_data)+1);
                d3 = *(((npy_intp*)msg_data)+2);
                d4 = *(((npy_intp*)msg_data)+3);
                d5 = *(((npy_intp*)msg_data)+4);
                t1 = msg_data+sizeof(npy_intp)*5;
                do_UFUNC_REDUCE(d1, d2, d3, d4, NULL, d5, t1);
                break;
            case DNPY_ZEROFILL:
                do_ZEROFILL(get_dndview(*((npy_intp*)msg_data)));
                break;
            case DNPY_DIAGONAL:
                ary  = get_dndview(((npy_intp*)msg_data)[0]);
                ary2 = get_dndview(((npy_intp*)msg_data)[1]);
                d1 = ((npy_intp*)msg_data)[2];
                d2 = ((npy_intp*)msg_data)[3];
                d3 = ((npy_intp*)msg_data)[4];
                do_DIAGONAL(ary, ary2, d1, d2, d3);
                break;
            case DNPY_MATMUL:
                ary  = get_dndview(((npy_intp*)msg_data)[0]);
                ary2 = get_dndview(((npy_intp*)msg_data)[1]);
                ary3 = get_dndview(((npy_intp*)msg_data)[2]);
                do_MATMUL(ary, ary2, ary3);
                break;
            default:
                fprintf(stderr, "Unknown msg: %ld\n", (long)msg[0]);
                MPI_Abort(MPI_COMM_WORLD, -1);
        }
    }
    return Py_False;

} /* dnumpy_master_slave_split */

/*NUMPY_API
 * ===================================================================
 * Public
 * Registrete the python module in which ufunction's operators can be
 * found.
 */
static void
dnumpy_reg_ufunc_module(PyObject *module)
{
    ufunc_module = module;
} /* dnumpy_reg_ufunc_module */

/*NUMPY_API
 * ===================================================================
 * Public
 * Registrete cblas. Used in _dotblas.c
 */
static void
dnumpy_reg_cblas(void *dgemm, void *sgemm, void *zgemm, void *cgemm)
{
    cblas_dgemm_func = dgemm;
    cblas_sgemm_func = sgemm;
    cblas_zgemm_func = zgemm;
    cblas_cgemm_func = cgemm;
} /* dnumpy_reg_Xgemm */

#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <math.h>

#ifndef CBLAS_HEADER
#define CBLAS_HEADER "../blasdot/cblas.h"
#endif
#include CBLAS_HEADER

/*===================================================================
 *
 * MPI process variables.
*/
static int myrank, worldsize;
static npy_intp blocksize;
static npy_intp msg[DNPY_MAX_MSG_SIZE];
//The work buffer and its next free slot.
static void *workbuf;
static void *workbuf_nextfree;
//Unique identification counter
static npy_intp uid_count=0;
//Array-views belonging to local MPI process
static dndview dndviews[DNPY_MAX_NARRAYS];
static npy_intp dndviews_uid[DNPY_MAX_NARRAYS];
//Cartesian dimension information - one for every dimension-order.
static int *cart_dim_strides[NPY_MAXDIMS];
static int *cart_dim_sizes[NPY_MAXDIMS];
static int cart_used_ndims[NPY_MAXDIMS];
//Pointer to the python module who has the ufunc operators.
static PyObject *ufunc_module;
//Number of view block operations in the sub-view-block DAG.
static npy_intp nvb_in_svb_dag=0;
//The ready queue for the sub-view-block DAG and its current size.
static dndnode *ready_queue[DNPY_RDY_QUEUE_MAXSIZE];
static npy_intp ready_queue_size=0;
//Unique MPI tag.
npy_intp mpi_tag=0;
//Pointers to the cblas functions.
typedef void (cblas_func_type)(const enum CBLAS_ORDER Order,...);
static cblas_func_type *cblas_dgemm_func = NULL;
static cblas_func_type *cblas_sgemm_func = NULL;
static cblas_func_type *cblas_zgemm_func = NULL;
static cblas_func_type *cblas_cgemm_func = NULL;
//Prototype for the DAGs.
static void dag_svb_flush(void);
static void dag_svb_add(void *node);
//Variables for statistics.
#ifdef DNPY_STATISTICS
    static int node_uid_count = 0;
#endif
//Variables for timing.
#ifdef DNDY_TIME
    #include <sys/time.h>
    struct timeval tv;
    struct timezone tz;
    static unsigned long long t_app_total=0, t_dag_svb_flush=0,     \
        t_dag_svb_add=0, t_calc_vblock=0, t_ufunc_svb=0,            \
        t_apply_ufunc=0, t_ufunc_comm=0;
    #define DNDTIME(output)                                         \
        gettimeofday(&tv, &tz);                                     \
        output = (unsigned long long) tv.tv_usec +                  \
                 (unsigned long long) tv.tv_sec * 1000000;
    #define DNDTIME_SUM(in,sum)                                     \
        gettimeofday(&tv, &tz);                                     \
        sum += ((unsigned long long) tv.tv_usec +                   \
                (unsigned long long) tv.tv_sec * 1000000) - in;
#endif

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
 * Returns a string describing the operation type.
 * Private
 */
#ifdef DNPY_STATISTICS
static char *optype2str(int optype)
{
    switch(optype)
    {
        case DNPY_CREATE_ARRAY:
            return "DNPY_CREATE_ARRAY";
        case DNPY_DESTROY_ARRAY:
            return "del";
        case DNPY_CREATE_VIEW:
            return "DNPY_CREATE_VIEW";
        case DNPY_PUT_ITEM:
            return "DNPY_PUT_ITEM";
        case DNPY_GET_ITEM:
            return "DNPY_GET_ITEM";
        case DNPY_UFUNC:
            return "ufunc";
        case DNPY_RECV:
            return "recv";
        case DNPY_SEND:
            return "send";
        case DNPY_BUF_RECV:
            return "Brecv";
        case DNPY_BUF_SEND:
            return "Bsend";
        case DNPY_APPLY:
            return "apply";
        case DNPY_UFUNC_REDUCE:
            return "DNPY_UFUNC_REDUCE";
        case DNPY_ZEROFILL:
            return "DNPY_ZEROFILL";
        case DNPY_DATAFILL:
            return "DNPY_DATAFILL";
        case DNPY_DIAGONAL:
            return "DNPY_DIAGONAL";
        case DNPY_MATMUL:
            return "DNPY_MATMUL";
        default:
            return "\"Unknown data type\"";
    }
} /* stat_op2str */

/*===================================================================
 *
 * Dumping the sub-view-block DAG to a file using the DOT language.
 * Private
 */
static void dag_svb_dump(void)
{
    FILE *pFile;
    dndnode *tnode;
    int i;
    char filename[20];

    if(ready_queue_size < 1)
        return;

    sprintf(filename, "dndstat_p%d", myrank);
    remove(filename);

    pFile = fopen(filename, "a");
    assert(pFile != NULL);

    //Write the DOT header.
    fputs("digraph {\n", pFile);

    tnode = ready_queue[0];
    while(tnode != NULL)
    {
        fprintf(pFile, "n%d [label=\"%s", tnode->uid,
                optype2str(tnode->op));
        if(tnode->optype == DNPY_COMM)
            fprintf(pFile, "-%ld: ", ((dndnode_comm*)tnode)->mpi_tag);
        else
            fprintf(pFile, ": ");

        for(i=0; i<tnode->narys; i++)
            fprintf(pFile, "a%ld ", (long) tnode->views[i]->uid);
        fprintf(pFile, "\"]\n");

        for(i=0; i<tnode->ndepend; i++)
            fprintf(pFile, "n%d -> n%d;\n",
                    tnode->uid, tnode->depend[i]->uid);

        //Go to next node.
        tnode = tnode->next;
    }

    //Write the DOT tail.
    fputs("}\n", pFile);

    fclose(pFile);
}/* dag_svb_dump */
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
    assert(rank < worldsize);
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
    if(uid)
        for(i=0; i < DNPY_MAX_NARRAYS; i++)
            if(dndviews_uid[i] == uid)
                return &dndviews[i];
    fprintf(stderr, "get_dndview, uid %ld does not exist\n", (long) uid);
    MPI_Abort(MPI_COMM_WORLD, -1);
    return NULL;
}
static dndview *put_dndview(dndview *view)
{
    npy_intp i;
    npy_intp ones[NPY_MAXDIMS];

    //Initiate the Python Array Object.
    for(i=0; i<view->ndims; i++)
        ones[i] = 1; //Just need ones for the dim.
    view->PyAry = PyArray_SimpleNewFromData(view->ndims, ones,
                                            view->base->dtype,
                                            view->base->data);

    for(i=0; i < DNPY_MAX_NARRAYS; i++)
        if(dndviews_uid[i] == 0)
        {
            memcpy(&dndviews[i], view, sizeof(dndview));
            dndviews_uid[i] = view->uid;
            return &dndviews[i];
        }
    fprintf(stderr, "put_dndarray, MAX_NARRAYS is exceeded\n");
    MPI_Abort(MPI_COMM_WORLD, -1);
    return NULL;
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
                //Deref the Python Array Object.
                Py_DECREF(dndviews[i].PyAry);
                //Cleanup base.
                dndviews_uid[i] = 0;
                if(dndviews[i].base->ndims == 0)//Dummy Scalar.
                {
                    free(dndviews[i].base->data);
                    free(dndviews[i].base);
                }
                else if(--dndviews[i].base->refcount == 0)
                {
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

 */
static MPI_Datatype
calc_svb_MPIdatatype(const dndview *view, dndsvb *svb)
{
    npy_intp i,j,stride;
    MPI_Datatype comm_viewOLD, comm_viewNEW;
    npy_intp start[NPY_MAXDIMS];
    npy_intp step[NPY_MAXDIMS];
    npy_intp nsteps[NPY_MAXDIMS];

    //Convert vcoord to coord, which have length view->base->ndims.
    j=0;
    for(i=0; i < view->nslice; i++)
    {
        if(view->slice[i].nsteps == PseudoIndex)
        {
            continue;
        }
        if(view->slice[i].nsteps == SingleIndex)
        {
            nsteps[j] = 1;
            step[j] = 1;
        }
        else
        {
            nsteps[j] = view->slice[i].nsteps;
            step[j] = view->slice[i].step;
        }
        start[j++] = view->slice[i].start;
    }

    //Compute the MPI datatype for communication.
    MPI_Type_dup(view->base->mpi_dtype, &comm_viewOLD);
    for(i=view->base->ndims-1; i >= 0; i--)//Row-major.
    {
        //Compute the MPI datatype for the view.
        stride = svb->stride[i] * step[i] * view->base->elsize;
        MPI_Type_create_hvector(svb->nsteps[i], 1, stride,
                                comm_viewOLD, &comm_viewNEW);

        //Cleanup and iterate comm types.
        MPI_Type_free(&comm_viewOLD);
        comm_viewOLD = comm_viewNEW;
    }
    MPI_Type_commit(&comm_viewNEW);
    return comm_viewNEW;
}/* calc_svb_MPIdatatype */

/*===================================================================
 *
 * Calculate the view block at the specified block-coordinate.
 * NB: vcoord is the visible coordinates and must therefore have
 * length view->ndims.
 * Private.
 */
static void
calc_vblock(const dndview *view, const npy_intp vcoord[NPY_MAXDIMS],
            dndvb *vblock)
{
    npy_intp i, j, B, item_idx, s, offset, goffset, voffset, boffset;
    npy_intp notfinished, stride, vitems, vvitems, vblocksize;
    npy_intp comm_offset, nelem;
    npy_intp coord[NPY_MAXDIMS];
    npy_intp scoord[NPY_MAXDIMS];
    npy_intp ncoord[NPY_MAXDIMS];
    int pcoord[NPY_MAXDIMS];
    int *cdims = cart_dim_sizes[view->base->ndims-1];
    npy_intp start[NPY_MAXDIMS];
    npy_intp step[NPY_MAXDIMS];
    npy_intp nsteps[NPY_MAXDIMS];
    dndsvb *svb;

    //Convert vcoord to coord, which have length view->base->ndims.
    j=0;s=0;
    for(i=0; i < view->nslice; i++)
    {
        if(view->slice[i].nsteps == PseudoIndex)
        {
            assert(vcoord[s] == 0);
            s++;
            continue;
        }
        if(view->slice[i].nsteps == SingleIndex)
        {
            nsteps[j] = 1;
            step[j] = 1;
            coord[j] = 0;
        }
        else
        {
            coord[j] = vcoord[s];
            nsteps[j] = view->slice[i].nsteps;
            step[j] = view->slice[i].step;
            s++;
        }
        start[j++] = view->slice[i].start;
    }
    assert(j == view->base->ndims);

    vblock->sub = workbuf_nextfree;
    svb = vblock->sub;

    //Init number of sub-view-block in each dimension.
    memset(vblock->svbdims, 0, view->base->ndims * sizeof(npy_intp));

    //Sub-vblocks coordinate.
    memset(scoord, 0, view->base->ndims * sizeof(npy_intp));
    //Compute all sub-vblocks associated with the n'th vblock.
    notfinished=1; s=0;
    while(notfinished)
    {
        for(i=view->base->ndims-1; i >= 0; i--)//Row-major.
        {
            //Non-block coordinates.
            ncoord[i] = coord[i] * blocksize;
            //View offset relative to array-view (non-block offset).
            voffset = ncoord[i] + scoord[i];
            //Global offset relative to array-base.
            goffset = voffset * step[i] + start[i];
            //Global block offset relative to array-base.
            B = goffset / blocksize;
            //Process rank of the owner in the i'th dimension.
            pcoord[i] = B % cdims[i];
            //Local block offset relative to array-base.
            boffset = B / cdims[i];
            //Item index local to the block.
            item_idx = goffset % blocksize;
            //Local offset relative to array-base.
            offset = boffset * blocksize + item_idx;
            //Save offset.
            svb[s].start[i] = offset;
            //Viewable items left in the block.
            vitems = MAX((blocksize - item_idx) / step[i], 1);
            //Size of current view block.
            vblocksize = MIN(blocksize, nsteps[i] - ncoord[i]);
            //Viewable items left in the view-block.
            vvitems = vblocksize - (voffset % blocksize);
            //Compute nsteps.
            svb[s].nsteps[i] = MIN(blocksize, MIN(vvitems, vitems));
            //Debug check.
            assert(svb[s].nsteps[i] >= 0);
        }
        //Find rank.
        svb[s].rank = cart2rank(view->base->ndims, pcoord);
        assert(svb[s].rank >= 0);
        //Data has not been fetched.
        svb[s].data = NULL;
        //Communication has not been handled.
        svb[s].comm_received_by = -1;

        //Compute the strides (we need the rank to do this).
        stride = 1;
        for(i=view->base->ndims-1; i >= 0; i--)//Row-major.
        {
            svb[s].stride[i] = stride;
            stride *= dnumroc(view->base->dims[i], blocksize,
                              pcoord[i], cdims[i]);
            assert(svb[s].stride[i] > 0);
        }

        //Compute the MPI datatype for communication.
        comm_offset = 0;
        nelem = 1;
        for(i=view->base->ndims-1; i >= 0; i--)//Row-major.
        {
            //Compute offsets.
            comm_offset += svb[s].start[i] * svb[s].stride[i];
            //Computing total number of elements.
            nelem *= svb[s].nsteps[i];
        }
        //Save offsets.
        svb[s].comm_offset = comm_offset * view->base->elsize;

        //Save total number of elements.
        svb[s].nelem = nelem;

        //Save data pointer if local data.
        if(svb[s].rank == myrank)
        {
            vblock->sub[s].data = view->base->data +
                                  svb[s].comm_offset;
        }

        //Iterate Sub-vblocks coordinate (Row-major).
        for(j=view->base->ndims-1; j >= 0; j--)
        {
            //Count svbdims.
            vblock->svbdims[j]++;

            scoord[j] += svb[s].nsteps[j];
            if(scoord[j] >= MIN(blocksize, nsteps[j] - ncoord[j]))
            {
                //We a finished, if wrapping around.
                if(j == 0)
                {
                    notfinished = 0;
                    break;
                }
                scoord[j] = 0;
            }
            else
                break;
        }
        //Reset svbdims because we need the last iteration.
        if(notfinished)
            for(i=view->base->ndims-1; i > j; i--)
                vblock->svbdims[i] = 0;

        s++;
    }
    //Save number of sub-vblocks.
    vblock->nsub = s;
    assert(vblock->nsub > 0);
    //And the next free work buffer slot.
    workbuf_nextfree += s * sizeof(dndsvb);
} /* calc_vblock */

/*===================================================================
 *
 * Convert visible vblock dimension index to base vblock
 * dimension index.
 * Private.
 */
static npy_intp
idx_v2b(const dndview *view, npy_intp vindex)
{
    assert(vindex < view->ndims);
    npy_intp i, bindex=0;
    for(i=0; i < view->nslice; i++)
    {
        if(view->slice[i].nsteps == SingleIndex)
        {
            if(view->base->ndims > 1)
                bindex++;
            continue;
        }
        if(vindex == 0)
            break;
        if(view->slice[i].nsteps != PseudoIndex)
            bindex++;

        vindex--;
    }
    assert(bindex < view->base->ndims);
    return bindex;
} /* idx_v2b */

/*===================================================================
 *
 * Convert visible vblock dimension index to slice dimension index.
 * Private.
 */
static npy_intp
idx_v2s(const dndview *view, npy_intp vindex)
{
    npy_intp i;
    assert(vindex < view->ndims);
    for(i=0; i < view->nslice; i++)
    {
        if(view->slice[i].nsteps == SingleIndex)
            continue;
        if(vindex == 0)
            break;
        vindex--;
    }
    assert(i < view->nslice);
    return i;
} /* idx_v2s */

/*===================================================================
 *
 * Apply the universal function decribed in 'apply'.
 * Private.
 */
static void
apply_ufunc(dndnode_ufunc *apply)
{
    int j, i;
    PyObject *iters[NPY_MAXARGS];
    npy_intp *lastdims, lastndim;
    int narys = apply->narys;

    #ifdef DNDY_TIME
        unsigned long long tdelta;
        DNDTIME(tdelta);
    #endif

    for(i=0; i<narys; i++)
    {
        PyObject *ary = apply->views[i]->PyAry;
        assert(ary != NULL);
        //Scalars are already finished.
        if(PyArray_NDIM(ary) == 0)
            continue;

        int s = PyArray_NDIM(ary) * sizeof(npy_intp);
        memcpy(PyArray_DIMS(ary), apply->viewup[i].dims, s);
        memcpy(PyArray_STRIDES(ary), apply->viewup[i].stride, s);
        PyArray_BYTES(ary) = apply->svbs[i]->data +
                             apply->viewup[i].offset;
    }

    lastdims = PyArray_DIMS(apply->views[narys-1]->PyAry);
    lastndim = PyArray_NDIM(apply->views[narys-1]->PyAry);

    if(apply->func == NULL)//Compatible Mode.
    {
        PyObject *tuple = PyTuple_New(narys);
        for(i=0; i<narys; i++)
        {
            PyObject *ary = apply->views[i]->PyAry;
            PyObject *tary = PyArray_New(&PyArray_Type,
                                         PyArray_NDIM(ary),
                                         PyArray_DIMS(ary),
                                         PyArray_TYPE(ary),
                                         PyArray_STRIDES(ary),
                                         PyArray_DATA(ary),
                                         PyArray_ITEMSIZE(ary),
                                         NPY_BEHAVED, NULL);
            assert(tary != NULL);
            PyTuple_SET_ITEM(tuple, i, tary);
        }
        PyObject *t = PyObject_CallObject(apply->PyOp, tuple);
        assert(t != NULL);
        Py_DECREF(t);
        Py_DECREF(tuple);
        return;
    }

    //Create broadcasted iterators.
    for(j=0; j<narys; j++)
        iters[j] = PyArray_BroadcastToShape(apply->views[j]->PyAry,
                                            lastdims, lastndim);

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
        //Apply operation.
        apply->func(tdata, &lastdims[axis], steps, apply->funcdata);
        //Go to next block.
        for(j=0; j<narys; j++)
            PyArray_ITER_NEXT(iters[j]);
    }
    //Cleanup
    for(j=0; j<narys; j++)
        Py_DECREF(iters[j]);

    #ifdef DNDY_TIME
        DNDTIME_SUM(tdelta, t_apply_ufunc)
    #endif
}/* apply_ufunc */


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
                           dndarray *dest_ary, MPI_Win comm_win,
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
    MPI_Win_lock(MPI_LOCK_EXCLUSIVE, rank, 0, comm_win);
    MPI_Put(PyArray_BYTES(src), 1, src_ary_type, rank, 0, 1,
            dest_ary_type, comm_win);
    MPI_Win_unlock(rank, comm_win);

    //Clean up
    MPI_Type_free(&src_ary_type);
    MPI_Type_free(&dest_ary_type);
    MPI_Type_free(&elem_type);
    return 0;
} /* remote_blockput */

/*===================================================================
 *
 * Create a MPI data type to match the given view for a
 * N-dimensional block cyclic distribution.
 * Private.
 */
static int create_mpi_dtype(dndview *view, MPI_Datatype *dtype)
{
    /* Get cartesian coords */
    int coords[NPY_MAXDIMS];
    rank2cart(view->base->ndims, myrank, coords);
    int *stride = cart_dim_sizes[view->base->ndims-1];

    /* Counters */
    int i, j, n, extent;

    /* Convenience */
    int ndims = view->base->ndims;

    /* Holds arrays of displacements and blocklengths for each dimension */
    int *displacements[ndims];
    int *blocklengths[ndims];

    /* Copies for reverse enumeration */
    int mydims[ndims];
    int mylocalblockdims[ndims];
    int mycoords[ndims];
    int mystride[ndims];

    /* We want to reverse the dimension enumeration for convenience */
    i = 0;
    for(n = ndims-1; n >= 0; n--) {
        mydims[i] = view->base->dims[n];
        mylocalblockdims[i] = view->base->localblockdims[n];
        mycoords[i] = coords[n];
        mystride[i++] = stride[n];
    }

    /* Allocate room for mylocalblockdims[n] elements in each dimension */
    for(n = 0; n < ndims; n++) {
        if(NULL == (displacements[n] = malloc(sizeof(int)*mylocalblockdims[n]))) {
            fprintf(stderr, "Failed to allocate memory for displacements.\n");
            return -1;
        }
        if(NULL == (blocklengths[n] = malloc(sizeof(int)*mylocalblockdims[n]))) {
            fprintf(stderr, "Failed to allocate memory for block lengths.\n");
            return -1;
        }
    }

    /* Iterate over dimensions */
    for(n = 0; n < ndims; n++) {
        j = 0; // Reset internal counter

        /* Iterate over a dimension to calculate displacements */
        for(i = mycoords[n]*blocksize; i < mydims[n]; ) {
            displacements[n][j] = i;

            if(blocksize+i > mydims[n]) {
                blocklengths[n][j] = mydims[n] % blocksize;
            } else {
                blocklengths[n][j] = blocksize;
            }

            i += mystride[n]*blocksize;
            j++;
        }
    }

    /* Stores derived data types */
    MPI_Datatype derived_type[ndims];
    MPI_Datatype tmp_type;

    /* Setup derived data type for elements */
    MPI_Type_dup(view->base->mpi_dtype, &(derived_type[0]));

    /* Setup derived data types for rest of dimensions */
    for(n = 1; n <= ndims; n++) {
        extent = view->base->elsize;
        for(i = 0; i < n; i++) {
            extent *= mydims[i];
        }

        MPI_Type_indexed(mylocalblockdims[n-1], blocklengths[n-1],
                         displacements[n-1], derived_type[n-1], &tmp_type);
        if(n < ndims) {
            MPI_Type_create_resized(tmp_type, 0, extent, &derived_type[n]);
        } else {
            MPI_Type_create_resized(tmp_type, 0, extent, dtype);
        }
    }

    /* Free allocated data */
    MPI_Type_free(&tmp_type);
    for(n = 1; n < ndims; n++) {
        MPI_Type_free(&derived_type[n]);
        free(displacements[n]);
    }

    MPI_Type_commit(dtype);
    return 0;

} /* create_mpi_dtype */


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
    int *cdims = cart_dim_sizes[ndims-1];
    npy_intp i;
    int cartcoord[NPY_MAXDIMS];

    //Save array uid.
    ary->uid = view->uid;

    //Allocate and copy the array to the views's base.
    view->base = malloc(sizeof(dndarray));
    memcpy(view->base, ary, sizeof(dndarray));
    ary = view->base;//Use the new pointer.

    //Get cartesian coords.
    rank2cart(ndims, myrank, cartcoord);

    //Acummulate total number of local sizes and save it.
    npy_intp nelements = 1;
    for(i=0; i < ary->ndims; i++)
    {
        ary->localdims[i] = dnumroc(ary->dims[i],
                            blocksize, cartcoord[i], cdims[i]);
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

    //Create a MPI-datatype that correspond to an array element.
    MPI_Type_contiguous(ary->elsize, MPI_BYTE, &ary->mpi_dtype);
    MPI_Type_commit(&ary->mpi_dtype);

    //Compute number of blocks.
    view->nblocks = 1;
    for(i=0; i<ndims;i++)
    {
        view->blockdims[i] = ceil(ary->dims[i] / (double) blocksize);
        view->nblocks *= view->blockdims[i];
    }

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

    dndnode *node = workbuf_nextfree;
    workbuf_nextfree += sizeof(dndnode);
    node->op = DNPY_DESTROY_ARRAY;
    node->optype = DNPY_NONCOMM;
    node->narys = 1;
    node->views[0] = get_dndview(uid);
    node->svbs[0] = NULL;
    node->accesstype[0] = DNPY_WRITE;
    dag_svb_add(node);

} /* do_DESTROY_ARRAY */


/*===================================================================
 *
 * Message handler for CREATE_VIEW.
 * Private.
 */
static void do_CREATE_VIEW(npy_intp org_view_uid, dndview *newview)
{
    npy_intp n, i, j;
    dndview *orgview = get_dndview(org_view_uid);

    //Print debug info.
    #ifdef DISTNUMPY_DEBUG
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

    dag_svb_flush();

    //Add the base to the new view.
    newview->base = orgview->base;
    newview->base->refcount++;

    //Save the view and change pointer to its new location.
    newview = put_dndview(newview);

    //Compute size of view-blocks.
    newview->nblocks = 1;
    n=0;
    for(i=0; i < newview->nslice; i++)
    {
        if(newview->slice[i].nsteps != SingleIndex)
        {
            j = 1;//SingleIndex has length one.
            if(newview->slice[i].nsteps != PseudoIndex)
                j = newview->slice[i].nsteps;

            newview->blockdims[n] = ceil(j / (double) blocksize);
            newview->nblocks *= newview->blockdims[n];
            n++;
        }
    }
} /* do_CREATE_VIEW */

/*===================================================================
 *
 * Message handler for PUT_ITEM and GET_ITEM.
 * Direction: 0=Get, 1=Put.
 * Private.
 */
static void do_PUTGET_ITEM(int Direction, dndview *view, char* item,
                           npy_intp coord[NPY_MAXDIMS])
{
    npy_intp i,j,b,offset;
    npy_intp n, s;
    npy_intp tcoord[NPY_MAXDIMS];
    npy_intp rcoord[NPY_MAXDIMS];
    npy_intp nsteps[NPY_MAXDIMS];
    npy_intp step[NPY_MAXDIMS];
    dndvb vblock;

    //Print debug info.
    #ifdef DISTNUMPY_DEBUG
        if(myrank == 0)
        {   if(Direction)
                printf("PUT_ITEM");
            else
                printf("GET_ITEM");
            printf(" - uid: %ld, coord: (",(long)view->uid);
            for(i=0; i < view->ndims; i++)
                printf(" %ld", (long)coord[i]);
            printf(")\n");
        }
    #endif

    dag_svb_flush();

    //Create the window for one-sided communication.
    MPI_Win comm_win;
    MPI_Win_create(view->base->data, view->base->nelements *
                   view->base->elsize, 1, MPI_INFO_NULL, MPI_COMM_WORLD,
                   &comm_win);

    if(myrank == 0)
    {
        //Convert to block coordinates.
        for(i=0; i<view->ndims; i++)
            tcoord[i] = coord[i] / blocksize;

        //Get view block info.
        calc_vblock(view, tcoord, &vblock);

        //Convert PseudoIndex and SingleIndex.
        j=0;n=0;
        for(i=0; i < view->nslice; i++)
        {
            if(view->slice[i].nsteps == PseudoIndex)
            {
                assert(view->slice[i].start == 0);
                n++;
            }
            else if(view->slice[i].nsteps == SingleIndex)
            {
                rcoord[j] = 0;//The offset is already incl. in the svb.
                nsteps[j] = 1;
                step[j] = 1;
                j++;
            }
            else
            {
                rcoord[j] = coord[n];
                nsteps[j] = view->slice[i].nsteps;
                step[j] = view->slice[i].step;
                j++; n++;
            }
        }

        //Convert global coordinate to index coordinates
        //relative to the view.
        for(i=0; i<view->base->ndims; i++)
            tcoord[i] = rcoord[i] % blocksize;

        //Find sub view block and convert icoord to coordinate
        //relative to the sub view block.
        s=1;b=0;
        for(i=view->base->ndims-1; i>=0; i--)//Row-major.
        {
            j = vblock.sub[b].nsteps[i];
            while(tcoord[i] >= vblock.sub[b].nsteps[i])
            {
                tcoord[i] -= vblock.sub[b].nsteps[i];
                j += vblock.sub[b].nsteps[i];
                b += s;
            }
            while(j < MIN(blocksize, nsteps[i] - rcoord[i]))
            {
                j += vblock.sub[b].nsteps[i];
            }
            s *= vblock.svbdims[i];
        }

        //Compute offset.
        offset = 0;
        for(i=view->base->ndims-1; i>=0; i--)//Row-major.
            offset += (vblock.sub[b].start[i] + tcoord[i] * step[i]) *
                      vblock.sub[b].stride[i];

        //Do the communication.
        MPI_Win_lock(MPI_LOCK_EXCLUSIVE, vblock.sub[b].rank,
                     MPI_MODE_NOCHECK, comm_win);
        if(Direction)
            MPI_Put(item, 1, view->base->mpi_dtype, vblock.sub[b].rank,
                    offset*view->base->elsize, 1, view->base->mpi_dtype,
                    comm_win);
        else
            MPI_Get(item, 1, view->base->mpi_dtype, vblock.sub[b].rank,
                    offset*view->base->elsize, 1, view->base->mpi_dtype,
                    comm_win);
        MPI_Win_unlock(vblock.sub[b].rank, comm_win);
    }

    dag_svb_flush();//Will cleanup the used sub-view-blocks.

    MPI_Win_free(&comm_win);

} /* do_PUTGET_ITEM */


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
    npy_intp i, j, s, n, v;
    npy_intp coord[NPY_MAXDIMS], bcoord[NPY_MAXDIMS];
    npy_intp vdims[NPY_MAXARGS][NPY_MAXDIMS];
    npy_intp offset, scoord[NPY_MAXDIMS], sdims[NPY_MAXDIMS];
    npy_intp sstride[NPY_MAXDIMS], csdims[NPY_MAXDIMS];
    npy_intp svboffset[NPY_MAXARGS][NPY_MAXDIMS];
    npy_intp csvb[NPY_MAXARGS][NPY_MAXDIMS];
    int dim_iter_order[NPY_MAXDIMS];
    int pcoord[NPY_MAXDIMS];
    int notfinished=1;
    dndvb vblocks[NPY_MAXARGS];
    int nin = narys - nout_arys;
    dndview *views[NPY_MAXARGS];
    PyUFuncGenericFunction func = NULL;
    void *funcdata = NULL;

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
        {   //Create a dummy view for the scalar.
            dndview dummy;
            //Use negated uid since positive uids are generated
            //by the master process only.
            dummy.uid = -(++uid_count);
            dummy.nslice = 0;
            dummy.base = malloc(sizeof(dndarray));
            dummy.base->uid = dummy.uid;
            dummy.base->ndims = 0;
            dummy.ndims = 0;
            dummy.base->dtype = scalar_dtype;
            dummy.base->nelements = 1;
            dummy.base->data = malloc(scalar_size);
            memcpy(dummy.base->data, scalar, scalar_size);
            dummy.base->elsize = scalar_size;
            views[i] = put_dndview(&dummy);
        }

    dndview *lastary = views[narys-1];

    //Print debug info.
    #ifdef DISTNUMPY_DEBUG
        printf("UFUNC on - in:");
        for(i=0; i<nin; i++)
            printf(" %ld", (long)views[i]->uid);
        printf(" out:");
        for(i=nin; i<narys; i++)
            printf(" %ld", (long)views[i]->uid);
        printf("\n");
    #endif

    //Get Python function.
    PyObject *PyOp = PyObject_GetAttrString(ufunc_module, op);
    assert(PyOp != NULL);
    PyUFuncObject *ufunc_obj = (PyUFuncObject*) PyOp;
    if(appropriate_function >= 0)
    {
        func = ufunc_obj->functions[appropriate_function];
        funcdata = ufunc_obj->data[appropriate_function];
    }

    //Find the order of which the dimensions should be iterated.
    {
        int sc = 0; //start counter.
        int ec = lastary->ndims-1; //end counter.
        for(i=ec; i>=0; i--)
        {
            int broadcast = 0;
            for(j=0; j<nin; j++)
            {
                int lastcount = i - (lastary->ndims - views[j]->ndims);
                int gotdim=0;
                if(lastcount >= 0)
                {
                    int si=views[j]->ndims-1;
                    for(s=views[j]->nslice-1; s>=0; s--)
                    {
                        if(views[j]->slice[s].nsteps != SingleIndex)
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
            //Dims we need to broadcast should be ordered first.
            if(broadcast)
                dim_iter_order[sc++] = i;
            else
                dim_iter_order[ec--] = i;
        }
    }
    //Reset vblocks.
    for(i=0; i<narys; i++)
        vblocks[i].uid = -1;

    //Iterate over all output vblocks.
    //NB: coord consist of only visible coordinates.
    memset(coord, 0, lastary->ndims * sizeof(npy_intp));
    for(; notfinished; mpi_tag++)
    {
        #ifdef DNDY_TIME
            unsigned long long tdelta;
            DNDTIME(tdelta);
        #endif

        //Get one vblock from all arrays.
        for(v=0; v<narys; v++)
        {
            //Scalars are already located locally
            if(views[v]->base->ndims == 0)
                continue;

            //Compute broadcasted coordinate 'bcoord' and the
            //dimensions of vdims.
            //NB: bcoord consist of only visible coordinates.
            assert(lastary->ndims >= views[v]->ndims);
            n = lastary->ndims-1;
            for(j=views[v]->ndims-1; j>=0; j--,n--)
            {
                s = views[v]->slice[idx_v2s(views[v],j)].nsteps;
                if(s == PseudoIndex)
                    s = 1;
                if(s == 1)
                    bcoord[j] = 0;// 1-dim broadcast.
                else
                    bcoord[j] = coord[n];
                vdims[v][j] = MIN(blocksize, s - bcoord[j]*blocksize);
            }
            #ifdef DNDY_TIME
                unsigned long long tdelta2;
                DNDTIME(tdelta2);
            #endif

            //Calculate the current coord's uid.
            npy_intp uid = 0;s=1;
            for(i=views[v]->ndims-1; i>=0; i--)
            {
                uid += bcoord[i] * s;
                s *= views[v]->blockdims[i];
            }
            //Check of we need to calulate a new view-block.
            if(vblocks[v].uid != uid)
            {
                vblocks[v].uid = uid;
                calc_vblock(views[v], bcoord, &vblocks[v]);
            }

            #ifdef DNDY_TIME
                DNDTIME_SUM(tdelta2, t_calc_vblock)
            #endif
        }

        //Compute the owner of the current coord.
        int coord_owner;
        for(i=0; i<lastary->ndims; i++)
            pcoord[i] = coord[i] % cart_dim_sizes[lastary->ndims-1][i];
        coord_owner = cart2rank(lastary->ndims, pcoord);

        //Do computation for view-block.
        memset(scoord, 0, lastary->ndims * sizeof(npy_intp));
        memset(svboffset, 0, NPY_MAXARGS * NPY_MAXDIMS * sizeof(npy_intp));
        memset(csvb, 0, NPY_MAXARGS * NPY_MAXDIMS * sizeof(npy_intp));
        int svb_notfinished = 1;
        for(; svb_notfinished; mpi_tag++)
        {
            //Find current sub-view-block for all arrays.
            dndsvb *svbs[NPY_MAXARGS];
            for(v=0; v<narys; v++)
            {
                if(views[v]->base->ndims == 0)
                    continue;

                s=1;j=0;
                for(i=views[v]->ndims-1; i>=0; i--)
                {
                    j += csvb[v][i] * s;
                    s *= vblocks[v].svbdims[idx_v2b(views[v],i)];
                }
                assert(j < vblocks[v].nsub);
                svbs[v] = &vblocks[v].sub[j];
            }

            //Find biggest overlapping chunk of sub-view-blocks.
            for(i=0; i<lastary->ndims; i++)
                sdims[i] = blocksize;
            for(i=lastary->ndims-1; i>=0; i--)
            {
                for(v=0; v<narys; v++)
                {
                    if(views[v]->base->ndims == 0)
                        continue;
                    j = i - (lastary->ndims - views[v]->ndims);

                    //Check for broadcast dimension.
                    if(j<0)//Fewer ndims than lastary.
                        continue;
                    //1-dimensions.
                    s = views[v]->slice[idx_v2s(views[v],j)].nsteps;
                    if(s == 1 || s == PseudoIndex)
                        continue;
                    //Save the smallest sdims.
                    s = svbs[v]->nsteps[idx_v2b(views[v],j)] -
                        svboffset[v][j];
                    sdims[i] = MIN(sdims[i], s);
                }
                assert(sdims[i] >= 0 &&
                       sdims[i] <= blocksize);
            }

            if(coord_owner == myrank)
            {
                //Add receive request to the svb-DAG.
                for(v=0; v<narys; v++, mpi_tag++)
                {
                    if(views[v]->base->ndims == 0)
                        continue;
                    //We did the comm already.
                    if(svbs[v]->comm_received_by == myrank)
                        continue;
                    //The data is located locally.
                    if(svbs[v]->rank == myrank)
                        continue;

                    assert(svbs[v]->data == NULL);

                    //Communication has now been handled.
                    svbs[v]->comm_received_by = myrank;

                    //Add a new sub-view block DAG node.
                    dndnode_comm *node = workbuf_nextfree;
                    workbuf_nextfree += sizeof(dndnode_comm);
                    node->op = DNPY_BUF_RECV;
                    node->optype = DNPY_COMM;
                    node->narys = 1;
                    node->views[0] = views[v];
                    node->svbs[0] = svbs[v];
                    node->accesstype[0] = DNPY_WRITE;
                    node->mpi_tag = mpi_tag;
                    node->remote_rank = svbs[v]->rank;
                    dag_svb_add(node);
                }
                //Add the apply request to the svb-DAG.
                dndnode_ufunc *node = workbuf_nextfree;
                workbuf_nextfree += sizeof(dndnode_ufunc);
                node->op = DNPY_UFUNC;
                node->optype = DNPY_NONCOMM;
                node->narys = narys;
                node->nout = nout_arys;
                memcpy(node->views, views, narys * sizeof(dndview*));
                for(v=0; v<nin; v++)
                    node->accesstype[v] = DNPY_READ;
                for(v=nin; v<narys; v++)
                    node->accesstype[v] = DNPY_WRITE;
                node->func = func;
                node->funcdata = funcdata;
                Py_INCREF(PyOp);
                node->PyOp = PyOp;

                //Update the PyArray Objects to match the current
                //sub-view-block.
                for(v=0; v<narys; v++)
                {
                    //Scalars is already updated.
                    if(views[v]->base->ndims == 0)
                    {
                        node->svbs[v] = NULL;
                        continue;
                    }

                    //Convert sdims to match current view ndims.
                    for(i=0; i<views[v]->ndims; i++)
                        csdims[views[v]->ndims-1-i] =
                                        MIN(vdims[v][views[v]->ndims-1-i],
                                            sdims[lastary->ndims-1-i]);

                    //Compute offset & stride for the Object.
                    //When the data is local it may be non-contiguous
                    //and when the data is from a remote source it is
                    //contiguous.
                    offset=0; s=views[v]->base->elsize;
                    if(svbs[v]->rank == myrank)
                        for(i=views[v]->ndims-1; i>=0; i--)
                        {
                            //PseudoIndex should have 0 in stride.
                            if(views[v]->slice[idx_v2s(views[v],i)].nsteps ==
                               PseudoIndex)
                            {
                                sstride[i] = 0;
                                continue;
                            }
                            sstride[i] = svbs[v]->stride[idx_v2b(views[v],i)] *
                                         views[v]->slice[idx_v2s(views[v],i)].step *
                                         views[v]->base->elsize;
                            offset += svboffset[v][i] * sstride[i];
                        }
                    else
                        for(i=views[v]->ndims-1; i>=0; i--)
                        {
                            //PseudoIndex should have 0 in stride.
                            if(views[v]->slice[idx_v2s(views[v],i)].nsteps ==
                               PseudoIndex)
                            {
                                sstride[i] = 0;
                                continue;
                            }
                            sstride[i] = s;
                            offset += svboffset[v][i] * s;
                            s *= svbs[v]->nsteps[idx_v2b(views[v],i)];
                        }
                    //Apply update to PyArray object.
                    //Which corresponds to the following:
                    //    PyArray_New(&PyArray_Type,
                    //                views[v]->ndims, csdims,
                    //                views[v]->base->dtype, sstride,
                    //                svbs[v]->data + offset,
                    //                views[v]->base->elsize,
                    //                NPY_BEHAVED, NULL);
                    s = views[v]->ndims * sizeof(npy_intp);
                    memcpy(node->viewup[v].dims, csdims, s);
                    memcpy(node->viewup[v].stride, sstride, s);
                    node->viewup[v].offset = offset;
                    node->svbs[v] = svbs[v];
                }
                //Add a new sub-view block DAG node.
                dag_svb_add(node);
            }
            else
            {   //Add send request to the svb-DAG.
                for(v=0; v<narys; v++, mpi_tag++)
                {
                    if(views[v]->base->ndims == 0)
                        continue;
                    //The data is not located on this proc.
                    if(svbs[v]->rank != myrank)
                        continue;
                    if(svbs[v]->comm_received_by == coord_owner)
                        continue;

                    //Communication has now been handled.
                    svbs[v]->comm_received_by = coord_owner;

                    //Add a new sub-view block DAG node.
                    dndnode_comm *node = workbuf_nextfree;
                    workbuf_nextfree += sizeof(dndnode_comm);
                    node->op = DNPY_SEND;
                    node->optype = DNPY_COMM;
                    node->narys = 1;
                    node->views[0] = views[v];
                    node->svbs[0] = svbs[v];
                    node->accesstype[0] = DNPY_READ;
                    node->mpi_tag = mpi_tag;
                    node->remote_rank = coord_owner;
                    dag_svb_add(node);
                }
            }

            //Iterate coord one chunk.
            for(i=lastary->ndims-1; i>=0; i--)
            {
                //Update svboffset and csvb.
                for(v=0; v<narys; v++)
                {
                    j = i - (lastary->ndims - views[v]->ndims);
                    if(j<0)
                        continue;
                    svboffset[v][j] += sdims[j];
                    if(svboffset[v][j] >=
                       svbs[v]->nsteps[idx_v2b(views[v],j)])
                    {
                        svboffset[v][j] = 0;
                    }
                }

                scoord[i] += sdims[i];
                n = lastary->slice[idx_v2s(lastary,i)].nsteps;
                if(n == PseudoIndex)
                    n = 0;
                if(scoord[i] >= MIN(blocksize, n - coord[i] * blocksize))
                {
                    //We a finished, if wrapping around.
                    if(i == 0)
                    {
                        svb_notfinished = 0;
                        break;
                    }
                    scoord[i] = 0;
                }
                else
                    break;
            }
            //Iterate csvb one sub-view-block.
            for(v=0; v<narys; v++)
            {
                for(i=views[v]->ndims-1; i >= 0; i--)
                {
                    if(svboffset[v][i] == 0)
                        csvb[v][i]++;

                    if(csvb[v][i] >=
                       vblocks[v].svbdims[idx_v2b(views[v],i)])
                        csvb[v][i] = 0;
                    else
                        break;
                }
            }
        }

        //Write vblocks to output arrays.
        for(v=nin; v<narys; v++)
            for(j=0; j<vblocks[v].nsub; j++, mpi_tag++)
            {
                dndsvb *svb = &vblocks[v].sub[j];

                //Add a new sub-view block DAG node.
                if(coord_owner == myrank && svb->rank != myrank)
                {
                    dndnode_comm *node = workbuf_nextfree;
                    workbuf_nextfree += sizeof(dndnode_comm);
                    node->narys = 1;
                    node->views[0] = views[v];
                    node->svbs[0] = svb;
                    node->accesstype[0] = DNPY_READ;
                    node->mpi_tag = mpi_tag;
                    node->op = DNPY_BUF_SEND;
                    node->optype = DNPY_COMM;
                    node->remote_rank = svb->rank;
                    dag_svb_add(node);
                }
                else if(coord_owner != myrank && svb->rank == myrank)
                {
                    dndnode_comm *node = workbuf_nextfree;
                    workbuf_nextfree += sizeof(dndnode_comm);
                    node->narys = 1;
                    node->views[0] = views[v];
                    node->svbs[0] = svb;
                    node->accesstype[0] = DNPY_WRITE;
                    node->mpi_tag = mpi_tag;
                    node->op = DNPY_RECV;
                    node->optype = DNPY_COMM;
                    node->remote_rank = coord_owner;
                    dag_svb_add(node);
                }
            }
        #ifdef DNDY_TIME
            DNDTIME_SUM(tdelta, t_ufunc_svb)
        #endif

        //We have added a whole view-block operation to the svb-DAG
        //and we may therefore have to flush it before calculating
        //new view-blocks.
        if(++nvb_in_svb_dag >= DNPY_MAX_VB_IN_SVB_DAG)
        {
            dag_svb_flush();
            //Reset vblocks.
            for(i=0; i<narys; i++)
                vblocks[i].uid = -1;
        }

        //Iterate coord one vblock.
        for(j=lastary->ndims-1; j >= 0; j--)
        {
            int o = dim_iter_order[j];//Using the computed order.
            coord[o]++;
            if(coord[o] >= lastary->blockdims[o])
            {
                //We are finished, if wrapping around.
                if(j == 0)
                {
                    notfinished = 0;
                    break;
                }
                coord[o] = 0;//Start coord.
            }
            else
                break;
        }
    }
    Py_DECREF(PyOp);
    //Clean up Scalar dummy.
    for(i=0; i<narys; i++)
        if(views[i]->base->ndims == 0)//Scalar dummy.
            do_DESTROY_ARRAY(views[i]->uid);
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

    dag_svb_flush();

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

    //Create the window for one-sided communication.
    MPI_Win comm_win;
    MPI_Win_create(out_ary->data, out_ary->nelements * out_ary->elsize,
                   1, MPI_INFO_NULL, MPI_COMM_WORLD, &comm_win);


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
                remote_blockput(subres, in_ary, out_ary, comm_win,
                                axis, lcoords);

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
    MPI_Win_free(&comm_win);

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

    dag_svb_flush();

    memset(view->base->data, 0, view->base->nelements *
                                view->base->elsize);
} /* do_ZEROFILL */

/*===================================================================
 *
 * Generic message handler for DATAFILL and DATADUMP.
 * Private.
 */
static void do_FILEIO(dndview *view, const char *filename, long filepos,
                      int op)
{
    /* MPI data type */
    MPI_Datatype dtype;

    //Check if the views covers the whole array.
    if(view->alterations != 0)
    {
        fprintf(stderr, "FILEIO with views not implemented\n");
        MPI_Abort(MPI_COMM_WORLD, -1);
    }

    dag_svb_flush();

    /* File access mode */
    int mode = 0;

    /* Setup file handle */
    MPI_File fh;
    switch(op) {
        case DNPY_DATAFILL:
            mode = MPI_MODE_RDONLY;
            break;
        case DNPY_DATADUMP:
            mode = MPI_MODE_WRONLY | MPI_MODE_APPEND;
            break;
        default:
            fprintf(stderr, "Unknown I/O operation\n");
            MPI_Abort(MPI_COMM_WORLD, -1);
    }

    //MPI_File_open discards const
    MPI_File_open(MPI_COMM_WORLD, (char *) filename,
                  mode, MPI_INFO_NULL, &fh);

    /* Setup MPI data type */
    create_mpi_dtype(view, &dtype);

    /* Set file view with derived data type */
    MPI_File_set_view(fh, filepos, view->base->mpi_dtype, dtype,
                      "native", MPI_INFO_NULL);

    /* Do operation */
    MPI_Status status;
    switch(op) {
        case DNPY_DATAFILL:
            MPI_File_read_all(fh, view->base->data, view->base->nelements,
                              view->base->mpi_dtype, &status);
            break;
        case DNPY_DATADUMP:
            MPI_File_write_all(fh, view->base->data, view->base->nelements,
                               view->base->mpi_dtype, &status);
            break;
        default:
            fprintf(stderr, "Huh? Unknown operation requested.\n");
            MPI_Abort(MPI_COMM_WORLD, -1);
    }

    /* Close file handle */
    MPI_File_close(&fh);

    /* Free derived data type */
    MPI_Type_free(&dtype);

} /* do_FILEIO */

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

    dag_svb_flush();

    //Create the window for one-sided communication.
    MPI_Win comm_win;
    MPI_Win_create(in->base->data, in->base->nelements *
                   in->base->elsize, 1, MPI_INFO_NULL, MPI_COMM_WORLD,
                   &comm_win);

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
        MPI_Win_lock(MPI_LOCK_SHARED, rank, MPI_MODE_NOCHECK, comm_win);
        MPI_Get(out_data, bsize, in->base->mpi_dtype, rank,
                offset * in->base->elsize, 1, arraytype,
                comm_win);
        MPI_Win_unlock(rank, comm_win);

        //Clean up
        MPI_Type_free(&arraytype);

        //Iterate output data one block.
        out_data += bsize * in->base->elsize;
    }
    //Free win.
    MPI_Win_free(&comm_win);

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

    dag_svb_flush();

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

/*===================================================================
 *
 * Flush the sub-view block DAG and the work buffer.
 * Private.
 */
static void dag_svb_flush(void)
{
    npy_intp i, j, f, commsize, ncommsize;
    int fcomm[DNPY_RDY_QUEUE_MAXSIZE];
    int fcommsize;
    MPI_Request reqs[DNPY_RDY_QUEUE_MAXSIZE];
    MPI_Status reqstatus[DNPY_RDY_QUEUE_MAXSIZE];
    dndnode *comm[DNPY_RDY_QUEUE_MAXSIZE];
    dndnode *ncomm[DNPY_RDY_QUEUE_MAXSIZE];
    MPI_Datatype dtype[DNPY_RDY_QUEUE_MAXSIZE];
    npy_intp dtypesize=0;

    #ifdef DNDY_TIME
        unsigned long long tdelta;
        DNDTIME(tdelta);
    #endif
    #ifdef DNPY_STATISTICS
        dag_svb_dump();
    #endif

    commsize=0; ncommsize=0;
    while(ready_queue_size + commsize + ncommsize > 0)
    {
        assert(ready_queue_size <= DNPY_RDY_QUEUE_MAXSIZE);

        //Sort the queue into two queues - one for communication and
        //one for non-communication nodes.
        //Furthermore, initiate the communication nodes.
        for(i=0; i<ready_queue_size; i++)
        {
            //Init. all communication nodes in the ready queue.
            if(ready_queue[i]->optype == DNPY_COMM)
            {
                dndnode_comm *C = (dndnode_comm*) ready_queue[i];
                MPI_Datatype comm_dtype;
                switch(C->op)
                {
                    case DNPY_RECV:
                        assert(C->narys == 1);
                        comm_dtype = calc_svb_MPIdatatype(C->views[0],
                                                          C->svbs[0]);
                        MPI_Irecv(C->views[0]->base->data +
                                  C->svbs[0]->comm_offset,
                                  1, comm_dtype,
                                  C->remote_rank, C->mpi_tag,
                                  MPI_COMM_WORLD, &reqs[commsize]);
                        MPI_Type_free(&comm_dtype);
                        break;
                    case DNPY_BUF_RECV:
                        assert(C->narys == 1);
                        assert(C->svbs[0]->data == NULL);
                        C->svbs[0]->data = workbuf_nextfree;
                        workbuf_nextfree += C->svbs[0]->nelem *
                                            C->views[0]->base->elsize;
                        MPI_Irecv(C->svbs[0]->data, C->svbs[0]->nelem,
                                  C->views[0]->base->mpi_dtype,
                                  C->remote_rank, C->mpi_tag,
                                  MPI_COMM_WORLD, &reqs[commsize]);
                        break;
                    case DNPY_SEND:
                        assert(C->narys == 1);
                        comm_dtype = calc_svb_MPIdatatype(C->views[0],
                                                          C->svbs[0]);
                        MPI_Isend(C->views[0]->base->data +
                                  C->svbs[0]->comm_offset,
                                  1, comm_dtype,
                                  C->remote_rank, C->mpi_tag,
                                  MPI_COMM_WORLD, &reqs[commsize]);
                        dtype[dtypesize++] = comm_dtype;
                        //At the moment we have to delay this freeing to
                        //the end of the flush. I’m not sure if this is
                        //a bug in DistNumPy or the MPICH-2 implementa-
                        //tion.
                        //MPI_Type_free(&comm_dtype);
                        break;
                    case DNPY_BUF_SEND:
                        assert(C->narys == 1);
                        assert(C->svbs[0]->data != NULL);
                        MPI_Isend(C->svbs[0]->data, C->svbs[0]->nelem,
                                  C->views[0]->base->mpi_dtype,
                                  C->remote_rank, C->mpi_tag,
                                  MPI_COMM_WORLD, &reqs[commsize]);
                        break;
                    default:
                        fprintf(stderr, "Unknown DAG operation: %d.\n",
                                (int)C->op);
                        MPI_Abort(MPI_COMM_WORLD, -1);
                }
                comm[commsize++] = ready_queue[i];
            }
            else
            {
                assert(ready_queue[i]->optype == DNPY_NONCOMM);
                ncomm[ncommsize++] = ready_queue[i];
            }
        }
        //The ready queue is now empty.
        ready_queue_size = 0;

        //Apply all non-communication nodes and move new non-depend
        //nodes to the ready queue.
        for(i=0; i<ncommsize; i++)
        {
            dndnode *node = ncomm[i];
            switch(node->op)
            {
                case DNPY_UFUNC:
                    apply_ufunc((dndnode_ufunc*)node);
                    Py_DECREF(((dndnode_ufunc*)node)->PyOp);
                    break;
                case DNPY_DESTROY_ARRAY:
                    assert(node->narys == 1);
                    assert(node->views[0] != NULL);
                    rm_dndview(node->views[0]->uid);
                    break;
                default:
                    fprintf(stderr, "Unknown DAG operation: %d.\n",
                            (int)node->op);
                    MPI_Abort(MPI_COMM_WORLD, -1);
            }
            for(j=0; j<node->ndepend; j++)
            {
                assert(node->depend[j]->mydepend > 0);
                node->depend[j]->mydepend--;
                if(node->depend[j]->mydepend == 0)//The node is ready.
                    ready_queue[ready_queue_size++] = node->depend[j];
            }
        }

        //Wait for communication when only communication nodes are
        //unhandled and move new non-depend nodes to the ready queue.
        if(ncommsize == 0 && ready_queue_size == 0 && commsize > 0)
        {
            #ifdef DNDY_TIME
                unsigned long long tdelta2;
                DNDTIME(tdelta2);
            #endif

            MPI_Waitsome(commsize, reqs, &fcommsize, fcomm, reqstatus);
            assert(fcommsize != MPI_UNDEFINED);

            #ifdef DNDY_TIME
                DNDTIME_SUM(tdelta2, t_ufunc_comm)
            #endif

            for(f=0; f<fcommsize; f++)
            {
                dndnode *node = comm[fcomm[f]];
                for(i=0; i<node->ndepend; i++)
                {
                    assert(node->depend[i]->mydepend > 0);
                    node->depend[i]->mydepend--;
                    if(node->depend[i]->mydepend == 0)//The node is ready.
                        ready_queue[ready_queue_size++] = node->depend[i];
                }
            }
            for(f=0; f<fcommsize; f++)
            {
                j=0;
                for(i=0; i<commsize; i++)
                    if(fcomm[f]-f != i)
                    {
                        comm[j] = comm[i];
                        reqs[j++] = reqs[i];
                    }
            }
            commsize -= fcommsize;
        }
        //Non-communication node queue is now empty.
        ncommsize = 0;
    }

    //Do the delayed MPI type freeing.
    for(i=0; i<dtypesize; i++)
        MPI_Type_free(&dtype[i]);

    //Resetting.
    nvb_in_svb_dag = 0;
    workbuf_nextfree = workbuf;
    assert(ready_queue_size == 0);

    #ifdef DNDY_TIME
        DNDTIME_SUM(tdelta, t_dag_svb_flush)
    #endif
}/* dag_svb_flush */

/*===================================================================
 *
 * Returns True if there is data conflict/overlap.
 * A is the node that is being inserted.
 * Private.
 */
static char dndnode_conflict(dndnode *A, dndnode *B)
{
    npy_intp i,j,d;
    //At the moment we only check conflicts at the array level.
    for(i=0; i<A->narys; i++)
        for(j=0; j<B->narys; j++)
            if(A->views[i]->base->uid == B->views[j]->base->uid)
                if(A->accesstype[i] == DNPY_WRITE ||
                   B->accesstype[j] == DNPY_WRITE)
                {
                    if(A->svbs[0] == NULL || B->svbs[0] == NULL)
                        return 1;//Depend on the whole array.

                    for(d=0; d<A->views[i]->base->ndims; d++)
                    {
                        if(B->svbs[j]->start[d] >
                           A->svbs[i]->start[d] + A->svbs[i]->nsteps[d]
                           ||
                           A->svbs[i]->start[d] >
                           B->svbs[j]->start[d] + B->svbs[j]->nsteps[d])
                            break;//No conflict at the svb level.
                        else
                            return 1;
                    }
                }
    return 0;
}/* dndnode_conflict */

/*===================================================================
 *
 * Add a node to the sub-view block DAG.
 * Private.
 */
static void dag_svb_add(void *node_input)
{
    dndnode *node = node_input;

    if(nvb_in_svb_dag >= DNPY_MAX_VB_IN_SVB_DAG)//Check for overflow.
        dag_svb_flush();

    #ifdef DNDY_TIME
        unsigned long long tdelta;
        DNDTIME(tdelta);
    #endif

    //Init DAG pointers.
    node->ndepend = 0;
    node->mydepend = 0;
    node->next = NULL;
    #ifdef DNPY_STATISTICS
        node->uid = ++node_uid_count;
    #endif

    //Check dependencies for all nodes in the svb DAG.
    if(ready_queue_size == 0)
        ready_queue[ready_queue_size++] = node;
    else
    {
        dndnode *tnode = ready_queue[0];
        while(1)
        {
            if(dndnode_conflict(node, tnode))
            {
                //Add the new found dependency.
                assert(tnode->ndepend < DNPY_MAX_DEPENDENCY);
                assert(node->mydepend < DNPY_MAX_DEPENDENCY);
                tnode->depend[tnode->ndepend++] = node;
                node->mydepend++;
            }
            //Go to next node.
            if(tnode->next == NULL)//We are finished.
            {
                tnode->next = node;
                break;
            }
            tnode = tnode->next;
        }
        if(node->mydepend == 0)
            ready_queue[ready_queue_size++] = node;
    }

    #ifdef DNDY_TIME
        DNDTIME_SUM(tdelta, t_dag_svb_add)
    #endif

} /* dag_svb_add */


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
} /* dnumpy_create_dndview */

/*NUMPY_API
 *===================================================================
 * Public
 * Executes all appending operations.
 */
static PyObject *
dnumpy_evalflush(void)
{
    //Tell slaves
    msg[0] = DNPY_EVALFLUSH;
    msg[1] = DNPY_MSG_END;
    msg2slaves(msg, 2 * sizeof(npy_intp));

    //The master should also do the operation.
    #ifdef DISTNUMPY_DEBUG
        printf("Rank 0 received msg: EVALFLUSH\n");
    #endif
    dag_svb_flush();
    Py_RETURN_NONE;
} /* dnumpy_evalflush */


/*NUMPY_API
 *===================================================================
 * Assign the value to array at 'coordinate'.
 * 'coordinate' size must be the same as view->ndims.
 * Steals all reference to item. (Item is lost).
 * Public
 */
static int
dnumpy_dndarray_putitem(intp uid, intp coordinate[NPY_MAXDIMS],
                        PyObject *item)
{
    //Get arrray structs.
    dndview *view = get_dndview(uid);
    int ndims = view->ndims;
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
    msg[0] = DNPY_PUT_ITEM;
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
    do_PUTGET_ITEM(1, view, PyArray_DATA(citem2), coordinate);

    //Clean up.
    Py_DECREF(citem2);
    Py_DECREF(item2);

    return 0;//Succes
} /* dnumpy_dndarray_putitem */

/*NUMPY_API
 *===================================================================
 * Get a single value specified by 'coordinate' from the array.
 * 'coordinate' size must be the same as view->ndims.
 * Public
 */
static void
dnumpy_dndarray_getitem(char *retdata, intp uid,
                        const intp coordinate[NPY_MAXDIMS])
{
    //Get arrray structs.
    dndview *view = get_dndview(uid);

    //Tell slaves to send item.
    msg[0] = DNPY_GET_ITEM;
    msg[1] = view->uid;
    memcpy(&msg[2], coordinate, sizeof(npy_intp)*view->ndims);
    *(((char *) &msg[2]) + sizeof(npy_intp)*view->ndims) = DNPY_MSG_END;

    msg2slaves(msg, 3*sizeof(npy_intp) + view->ndims*sizeof(npy_intp));

    //The master will retrive the whole array.
    #ifdef DISTNUMPY_DEBUG
        printf("Rank 0 received msg: ");
    #endif

    do_PUTGET_ITEM(0, view, retdata, &msg[2]);
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
 * Fill the distributed array with data from file.
 */
static void
dnumpy_datafill(intp ary_uid, const char *filename, long datapos)
{
    //Get arrray structs.
    dndview *view = get_dndview(ary_uid);

    //Get filename length
    int filename_length = strlen(filename);

    //Tell slaves
    msg[0] = DNPY_DATAFILL;
    msg[1] = view->uid;
    memcpy(&msg[2], &datapos, sizeof(long));
    strcpy((char*)&msg[2]+sizeof(long), filename);
    *(&msg[2]+sizeof(long)+(filename_length+1)*sizeof(char)) = DNPY_MSG_END;
    msg2slaves(msg, 3*sizeof(npy_intp) + sizeof(long) + (filename_length+1)*sizeof(char));

    //The master should also do the operation.
    do_FILEIO(view, filename, datapos, DNPY_DATAFILL);
} /* dnumpy_datafill */

/*NUMPY_API
 *===================================================================
 * Public
 * Save the distributed array to file.
 */
static void
dnumpy_datadump(intp ary_uid, const char *filename, long datapos)
{
    //Get array structs.
    dndview *view = get_dndview(ary_uid);

    //Get filename length
    int filename_length = strlen(filename);

    //Tell slaves
    msg[0] = DNPY_DATADUMP;
    msg[1] = view->uid;
    memcpy(&msg[2], &datapos, sizeof(long));
    strcpy((char*)&msg[2]+sizeof(long), filename);
    *(&msg[2]+sizeof(long)+(filename_length+1)*sizeof(char)) = DNPY_MSG_END;
    msg2slaves(msg, 3*sizeof(npy_intp) + sizeof(long) + (filename_length+1)*sizeof(char));

    //The master should also do the operation.
    do_FILEIO(view, filename, datapos, DNPY_DATADUMP);
} /* dnumpy_datadump */

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
                      NPY_BEHAVED | NPY_CARRAY | DNPY_DISTRIBUTED,
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

    //Allocate buffers.
    workbuf = malloc(DNPY_WORK_BUFFER_MAXSIZE);
    workbuf_nextfree = workbuf;
    assert(workbuf != NULL);

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
    //Make sure that the sub-view-block DAG is flushed.
    dag_svb_flush();

    #ifdef DNDY_TIME
        #ifdef DNDY_TIME_NODE
        if(DNDY_TIME_NODE == myrank)
        #endif
        {
        unsigned long long total=0;
        DNDTIME_SUM(t_app_total, total);
        printf("app_total:         %6.llu ms\n", total/1000);
        printf("  dag_svb_flush:   %6.llu ms\n", t_dag_svb_flush/1000);
        printf("    apply_ufunc:   %6.llu ms\n", t_apply_ufunc/1000);
        printf("    ufunc_comm:    %6.llu ms\n", t_ufunc_comm/1000);
        printf("  ufunc_svb:       %6.llu ms\n", t_ufunc_svb/1000);
        printf("    dag_svb_add:   %6.llu ms\n", t_dag_svb_add/1000);
        printf("    calc_vblock:   %6.llu ms\n", t_calc_vblock/1000);
        }
    #endif

    //Free buffers.
    free(workbuf);
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
dnumpy_master_slave_split(void)
{
    #ifdef DNDY_TIME
        DNDTIME(t_app_total)
    #endif

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
        long l1;
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
            case DNPY_EVALFLUSH:
                #ifdef DISTNUMPY_DEBUG
                    printf("EVALFLUSH\n");
                #endif
                dag_svb_flush();
                break;
            case DNPY_PUT_ITEM:
                ary = get_dndview(*((npy_intp*)msg_data));
                t1 = msg_data+sizeof(npy_intp);
                t2 = t1+ary->base->elsize;
                do_PUTGET_ITEM(1, ary, t1, (npy_intp*) t2);
                break;
            case DNPY_GET_ITEM:
                ary = get_dndview(*((npy_intp*)msg_data));
                t1 = msg_data+sizeof(npy_intp);
                do_PUTGET_ITEM(0, ary, NULL, (npy_intp*) t1);
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
            case DNPY_DATAFILL:
                d1 = ((npy_intp*)msg_data)[0]; // view uid
                l1 = (long) ((npy_intp*)msg_data)[1]; // filepos
                t1 = msg_data+sizeof(npy_intp)+sizeof(long); // get filename
                do_FILEIO(get_dndview(d1), t1, l1, DNPY_DATAFILL);
                break;
            case DNPY_DATADUMP:
                d1 = ((npy_intp*)msg_data)[0]; // view uid
                l1 = (long) ((npy_intp*)msg_data)[1]; // filepos
                t1 = msg_data+sizeof(npy_intp)+sizeof(long); // get filename
                do_FILEIO(get_dndview(d1), t1, l1, DNPY_DATADUMP);
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
 * Registrete the python module in which ufunction's operators can be
 * found.
 */
static PyObject*
dnumpy_get_reged_ufunc_module()
{
    return ufunc_module;
} /* dnumpy_get_reged_ufunc_module */

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

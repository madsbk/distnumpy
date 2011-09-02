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
#ifndef DNPY_SPMD
static npy_intp msg[DNPY_MAX_MSG_SIZE];
#endif
static npy_intp initmsg_not_handled=1;
//The work buffer and its next free slot.
static void *workbuf;
static void *workbuf_nextfree;
static void *workbuf_max;
//Unique identification counter
static npy_intp uid_count=0;
//Array-views belonging to local MPI process
static dndview dndviews[DNPY_MAX_NARRAYS];
static npy_intp ndndarrays=0;//Current number of dndviews allocated.
static npy_intp dndviews_uid[DNPY_MAX_NARRAYS];
//Cartesian dimension information - one for every dimension-order.
static int *cart_dim_strides[NPY_MAXDIMS];
static int *cart_dim_sizes[NPY_MAXDIMS];
//Pointer to the python module who has the ufunc operators.
static PyObject *ufunc_module;
//The ready queue for operations and its current size.
static dndop *ready_queue[DNPY_RDY_QUEUE_MAXSIZE];
static npy_intp ready_queue_size=0;
//Unique MPI tag.
static int mpi_tag=0;
//Memory pool.
static dndmem *mem_pool = NULL;
//Pointers to the cblas functions.
typedef void (cblas_func_type)(const enum CBLAS_ORDER Order,...);
static cblas_func_type *cblas_dgemm_func = NULL;
static cblas_func_type *cblas_sgemm_func = NULL;
static cblas_func_type *cblas_zgemm_func = NULL;
static cblas_func_type *cblas_cgemm_func = NULL;
//Prototype for the DAGs.
static void dag_svb_flush(int free_workbuf);
static void dag_svb_add(dndnode *nodes, int nnodes, int force_laziness);
//Variables for statistics.
#ifdef DNPY_STATISTICS
    static int node_uid_count = 0;
    static int op_uid_count = 0;
    static dndarray *rootarray = NULL;
#endif

//Variables for timing.
#include <sys/time.h>
struct timeval tv;
struct timezone tz;
static dndtime dndt;
unsigned long long totaldelta;

#define DNDTIME(output)                                         \
    gettimeofday(&tv, &tz);                                     \
    output = (unsigned long long) tv.tv_usec +                  \
             (unsigned long long) tv.tv_sec * 1000000;
#define DNDTIME_SUM(in,sum)                                     \
    gettimeofday(&tv, &tz);                                     \
    sum += ((unsigned long long) tv.tv_usec +                   \
            (unsigned long long) tv.tv_sec * 1000000) - in;

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
        case DNPY_REDUCE_SEND:
            return "reduce_send";
        case DNPY_REDUCE_RECV:
            return "DNPY_REDUCE_RECV";
        default:
            return "\"Unknown data type\"";
    }
} /* optype2str */

#ifdef DNPY_STATISTICS
/*===================================================================
 *
 * Dumping the sub-view-block DAG to a file using the DOT language.
 * Private
 */
static void dag_svb_dump(void)
{
    FILE *pFile;
    dndnode *tnode;
    dndarray *tary;
    npy_intp i, j;
    char filename[20];

    if(ready_queue_size < 1)
        return;

    sprintf(filename, "dndstat_p%d", myrank);
    remove(filename);

    pFile = fopen(filename, "a");
    assert(pFile != NULL);

    //Write the DOT header.
    fputs("digraph {\n", pFile);

    //Write the ready queue.
    fprintf(pFile, "ready_queue [label=\"Ready");
    for(i=0; i<ready_queue_size; i++)
        fprintf(pFile, "|<f%ld>%ld", (long)i, (long)i);
    fprintf(pFile, "\" shape=\"record\"];\n");
    for(i=0; i<ready_queue_size; i++)
        fprintf(pFile, "ready_queue:<f%ld> -> op%ld"
                       "[color=\"0.650 0.700 0.700\"];\n", (long)i,
                       (long)ready_queue[i]->uid);

    //Iterate through all arrays.
    tary = rootarray;
    while(tary != NULL)
    {
        //Write the root nodes.
        if(tary->uid < 0)//DOT don't support negative IDs.
            fprintf(pFile, "arrayn%ld", (long)tary->uid * -1);
        else
            fprintf(pFile, "array%ld", (long)tary->uid);
        fprintf(pFile, " [label=\"array %ld(r%d)", (long)tary->uid,
                tary->refcount);
        for(i=0; i < tary->nblocks; i++)
            if(tary->rootnodes[i] != NULL)
                fprintf(pFile, "|<f%ld>%ld", (long)i, (long)i);
        fprintf(pFile, "\" shape=\"record\"];\n");

        //Write the nodes.
        for(i=0; i < tary->nblocks; i++)
            if(tary->rootnodes[i] != NULL)
            {
                tnode = tary->rootnodes[i];
                if(tary->uid < 0)//DOT don't support negative IDs.
                    fprintf(pFile, "arrayn%ld", (long)tary->uid * -1);
                else
                    fprintf(pFile, "array%ld", (long)tary->uid);
                fprintf(pFile, ":<f%ld> -> node%ld;\n", (long)i,
                                (long)tnode->uid);
                while(tnode != NULL)
                {
                    //Write the linked list of nodes.
                    fprintf(pFile, "node%ld", (long)tnode->uid);
                    if(tnode->next != NULL)
                    {
                        assert(tnode->next->uid > 0);
                        fprintf(pFile, " -> node%ld",
                                       (long)tnode->next->uid);
                    }
                    fprintf(pFile, ";\n");

                    //Write the node-to-operation relation.
                    fprintf(pFile, "node%ld -> op%ld [color=\"0.650"
                                   " 0.700 0.700\"];\n",
                                  (long)tnode->uid,
                                  (long)tnode->op->uid);
                    fprintf(pFile, "op%ld [label=\"%s-r%ld:",
                                  (long)tnode->op->uid,
                                  optype2str(tnode->op->op),
                                  (long)tnode->op->refcount);
                    for(j=0; j<tnode->op->narys; j++)
                        fprintf(pFile, " a%ld(%ld)",
                                  (long)tnode->op->views[j]->uid,
                                  (long)tnode->op->views[j]->base->uid);
                    fprintf(pFile, "\" shape=\"note\"];\n");
                    tnode = tnode->next;
                }
            }
        //Go to the next ary.
        tary = tary->next;
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
                        int proc_dim_rank, int nproc_in_dim,
                        int first_process)
{
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
    for(i=0; i<ndims; i++)
        rank += coords[i] * strides[i];
    assert(rank < worldsize);
    return rank;
}
static void rank2cart(int ndims, int rank, int coords[NPY_MAXDIMS])
{
    int i;
    int *strides = cart_dim_strides[ndims-1];
    memset(coords, 0, ndims*sizeof(int));
    for(i=0; i<ndims; i++)
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
                dndview *view = &dndviews[i];
                //Cleanup base.
                dndviews_uid[i] = 0;
                if(view->base->ndims == 0)//Dummy Scalar.
                {
                    //Remove the array from to the lisked list used
                    //when statistic is defined.
                    #ifdef DNPY_STATISTICS
                        if(view->base->next != NULL)
                            view->base->next->prev = view->base->prev;
                        if(view->base->prev != NULL)
                            view->base->prev->next = view->base->next;
                        else
                            rootarray = view->base->next;
                    #endif
                    free(view->base->rootnodes);
                    free(view->base->data);
                    free(view->base);
                }
                else if(--view->base->refcount == 0)
                {
                    //Remove the array from to the lisked list used
                    //when statistic is defined.
                    #ifdef DNPY_STATISTICS
                        if(view->base->next != NULL)
                            view->base->next->prev = view->base->prev;
                        if(view->base->prev != NULL)
                            view->base->prev->next = view->base->next;
                        else
                            rootarray = view->base->next;
                    #endif
                    MPI_Type_free(&view->base->mpi_dtype);
                    if(view->base->data != NULL)
                    {
                        #ifdef DNDY_TIME
                            unsigned long long tdelta;
                            DNDTIME(tdelta);
                        #endif
                        dndmem *mem = (dndmem*) (view->base->data -
                                                 sizeof(dndmem));

                        //Put the allocated memory in front of the pool.
                        mem->next = mem_pool;
                        mem_pool = mem;

                        #ifdef DNDY_TIME
                            DNDTIME_SUM(tdelta, dndt.arydata_free)
                        #endif
                    }
                    free(view->base->rootnodes);
                    free(view->base);
                    --ndndarrays;
                    assert(ndndarrays >= 0);
                }
                return;
            }
    fprintf(stderr, "rm_dndarray, uid %ld does not exist\n", (long)uid);
    MPI_Abort(MPI_COMM_WORLD, -1);
    return;
}/* Put, get & rm dndview */

#ifndef DNPY_SPMD
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

    #ifdef DNDY_TIME
        unsigned long long tdelta;
        DNDTIME(tdelta);
    #endif

    MPI_Bcast(msg, DNPY_MAX_MSG_SIZE, MPI_BYTE, 0, MPI_COMM_WORLD);

    #ifdef DNDY_TIME
        DNDTIME_SUM(tdelta, dndt.msg2slaves)
    #endif

} /* msg2slaves */
#endif

/*===================================================================
 *  Returns a MPI data type that match the specified sub-view-block.
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
 * Frees the memory pool from a given memory allocation.
 * Private.
 */
static void
free_mem_pool(dndmem *mem)
{
    while(mem != NULL)
    {
        dndmem *next = mem->next;
        MPI_Free_mem(mem);
        mem = next;
    }
} /* free_mem_pool */



/*===================================================================
 *
 * Makes sure that the array's memory has been allocated.
 * Private.
 */
static void
delayed_array_allocation(dndarray *ary)
{
    #ifdef DNDY_TIME
        unsigned long long tdelta;
        DNDTIME(tdelta);
    #endif
    npy_intp size = ary->localsize * ary->elsize;
    npy_intp count = 0;
    dndmem *free = NULL;

    if(ary->data != NULL)//Already allocated.
        return;

    //Check if there is some free memory in the memory pool.
    if(mem_pool != NULL)
    {
        dndmem *prev = mem_pool;
        dndmem *next = mem_pool->next;

        //Handle first iteration as a special case.
        if(mem_pool->size == size)
        {
            ary->data = ((char*)mem_pool) + sizeof(dndmem);
            mem_pool = mem_pool->next;//Remove from pool.
            #ifdef DNDY_TIME
                ++dndt.mem_reused;
            #endif
        }
        else//Handle the rest.
        {
            while(next != NULL)
            {
                assert(prev->next == next);
                if(next->size == size)
                {
                    ary->data = ((char*)next) + sizeof(dndmem);
                    prev->next = next->next;//Remove from pool.
                    #ifdef DNDY_TIME
                        ++dndt.mem_reused;
                    #endif
                    break;
                }
                if(++count == DNPY_MAX_MEM_POOL)
                {
                    //Will remove all mem after this one.
                    prev->next = NULL;
                    free = next;
                }

                //Go to next memory allocation.
                prev = next;
                next = next->next;
            }
        }
    }

    if(ary->data == NULL)//Need to allocate new memory.
    {
        dndmem *mem;
        if(MPI_Alloc_mem(size + sizeof(dndmem), MPI_INFO_NULL,
                         &mem) != MPI_SUCCESS)
        {
            fprintf(stderr, "Out of memory!\n");
            MPI_Abort(MPI_COMM_WORLD, -1);
        }
        mem->size = size;
        ary->data = (char*) (mem + 1);
    }

    //Reduce the pool size to DNPY_MAX_MEM_POOL.
    if(free != NULL)
        free_mem_pool(free);

    #ifdef DNDY_TIME
        DNDTIME_SUM(tdelta, dndt.arydata_malloc)
    #endif
}/* delayed_array_allocation */


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
        assert(nsteps[j] > 0);
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
        dndnode **rootnode = view->base->rootnodes;
        stride = 1;
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
            //Compute this sub-view-block's root node.
            rootnode += B * stride;
            stride *= view->base->blockdims[i];
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
            assert(svb[s].nsteps[i] > 0);
        }
        //Find rank.
        if(view->base->onerank < 0)
            svb[s].rank = cart2rank(view->base->ndims,pcoord);
        else
            svb[s].rank = view->base->onerank;

        assert(svb[s].rank >= 0);
        //Data has not been fetched.
        svb[s].data = NULL;
        //Communication has not been handled.
        svb[s].comm_received_by = -1;
        //Save rootnode.
        svb[s].rootnode = rootnode;

        //Compute the strides (we need the rank to do this).
        stride = 1;
        if(view->base->onerank < 0)
            for(i=view->base->ndims-1; i >= 0; i--)
            {
                svb[s].stride[i] = stride;
                stride *= dnumroc(view->base->dims[i], blocksize,
                                  pcoord[i], cdims[i], 0);
                assert(svb[s].stride[i] > 0);
            }
        else//All on one rank.
            for(i=view->base->ndims-1; i >= 0; i--)
            {
                svb[s].stride[i] = stride;
                stride = view->base->dims[i];
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
            delayed_array_allocation(view->base);
            vblock->sub[s].data = view->base->data + svb[s].comm_offset;
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
    WORKBUF_INC(s * sizeof(dndsvb));
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
    //We need the MIN since bindex is too high when PseudoIndex is
    //used at the end of the view.
    return MIN(bindex, view->base->ndims-1);
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
apply_ufunc(dndop_ufunc *apply)
{
    int j, i;
    int narys = apply->narys;
    npy_intp *lastdims = apply->asvb[narys-1].dims;
    npy_intp lastndims = apply->views[narys-1]->ndims;

    //Pack all data.
    char *tdata[NPY_MAXARGS];
    for(j=0; j<narys; j++)
    {
        if(apply->views[j]->ndims == 0)
            tdata[j] = apply->views[j]->base->data;
        else
            tdata[j] = apply->svbs[j]->data + apply->asvb[j].offset;
        assert(tdata[j] != NULL);
    }

    if(apply->func == NULL)//Compatible Mode.
    {
        PyObject *tuple = PyTuple_New(narys);
        for(i=0; i<narys; i++)
        {
            PyObject *tary = PyArray_New(&PyArray_Type,
                                         apply->views[i]->ndims,
                                         apply->asvb[i].dims,
                                         apply->views[i]->base->dtype,
                                         apply->asvb[i].stride,
                                         tdata[i],
                                         apply->views[i]->base->elsize,
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
    int axis = lastndims-1;
    npy_intp strides[NPY_MAXARGS][NPY_MAXDIMS];
    npy_intp backstrides[NPY_MAXARGS][NPY_MAXDIMS];
    for(j=0; j<narys; j++)
    {
        for (i = 0; i < lastndims; i++)
        {
            int k = i - (lastndims - apply->views[j]->ndims);
            if((k < 0) || apply->asvb[j].dims[k] != lastdims[i])
            {
                strides[j][i] = 0;
            }
            else
            {
                strides[j][i] = apply->asvb[j].stride[k];
            }
            backstrides[j][i] = strides[j][i] * (lastdims[i] - 1);
        }
    }

    //'steps' is the axis-dimension's stride.
    npy_intp steps[NPY_MAXARGS];
    for(j=0; j<narys; j++)
        steps[j] = strides[j][axis];

    //Init coord.
    npy_intp coord[NPY_MAXDIMS];
    memset(coord, 0, lastndims * sizeof(npy_intp));

    #ifdef DNDY_TIME
        unsigned long long tdelta;
        DNDTIME(tdelta);
    #endif

    //Handling one-dimensional arrays are very simple.
    if(lastndims == 1)
    {
        apply->func(tdata, &lastdims[axis], steps, apply->funcdata);
        return;
    }

    int ndims2update = lastndims-2;
    npy_intp *length = &lastdims[axis];
    int notfinished = 1;
    while(notfinished)
    {
        //Apply operation.
        apply->func(tdata, length, steps, apply->funcdata);

        //Go to next block. The last coordinate is skiped.
        for(i=ndims2update; i >= 0; i--)
        {
            coord[i]++;
            if(coord[i] >= lastdims[i])
            {
                if(i==0)//Finished
                {
                    notfinished = 0;
                    break;
                }
                coord[i] = 0;
                for(j=0; j<narys; j++)
                    tdata[j] -= backstrides[j][i];
            }
            else
            {
                for(j=0; j<narys; j++)
                    tdata[j] += strides[j][i];
                break;
            }
        }
    }

    #ifdef DNDY_TIME
        DNDTIME_SUM(tdelta, dndt.apply_ufunc)
    #endif
}/* apply_ufunc */



/*===================================================================
 *
 * Return True if the coordinate can be skiped.
 * Private.
 */
static int
ufunc_skip_coord(npy_intp coord[NPY_MAXDIMS], int narys,
                 dndview *views[NPY_MAXARGS], dndview *lastary)
{
    npy_intp i, v;
    int pcoord[NPY_MAXDIMS];
    int skip_coord=0;
    int coord_owner;

    //Compute the owner of the current coord.
    for(i=0; i<lastary->ndims; i++)
        pcoord[i] = coord[i] % cart_dim_sizes[lastary->ndims-1][i];
    coord_owner = cart2rank(lastary->ndims, pcoord);

    //Lets check if the MPI-process can skip this iteration.
    if(coord_owner != myrank)
    {
        skip_coord=1;
        for(v=0; v<narys; v++)
        {
            dndview *view = views[v];
            //Scalars are ignored
            if(view->base->ndims == 0)
                continue;

            rank2cart(view->base->ndims, myrank, pcoord);
            int conflict = 1;
            for(i=0; i<view->ndims; i++)
            {
                dndslice *t = &view->slice[idx_v2s(view, i)];
                if(t->nsteps == PseudoIndex)
                    continue;
                int bdim = idx_v2b(view,i);
                int nproc = cart_dim_sizes[view->base->ndims-1][bdim];

                //Base index
                npy_intp s_base_idx = coord[i]*blocksize*t->step
                                      + t->start;
                //Process index
                npy_intp s_proc_idx = (s_base_idx/blocksize) % nproc;

                if(!((s_proc_idx <= pcoord[bdim] &&
                      pcoord[bdim] <= s_proc_idx+t->step) ||
                     (s_proc_idx > pcoord[bdim] &&
                     t->step >= nproc-(s_proc_idx-pcoord[bdim]))))
                {
                    conflict = 0;
                    break;
                }
            }
            if(conflict)
            {
                skip_coord = 0;
                break;
            }
        }
    }
    return skip_coord;

}/* ufunc_skip_coord */


/*===================================================================
 *
 * Handles the computation of one ufunc block coordinate.
 * Private.
 */
static void
ufunc_handle_coord(npy_intp coord[NPY_MAXDIMS], int narys, int nout_arys,
                   dndview *views[NPY_MAXARGS], dndvb vblocks[NPY_MAXARGS],
                   PyObject *PyOp, PyUFuncGenericFunction func, void *funcdata,
                   int *first_mpi_tag, int dim_iter_order[NPY_MAXDIMS])
{
    npy_intp i,j,v,s,n;
    dndview *lastary = views[narys-1];
    int nin = narys - nout_arys;
    int mpi_tag_svb=0;
    npy_intp csdims[NPY_MAXDIMS];
    npy_intp sstride[NPY_MAXDIMS], svboffset[NPY_MAXARGS][NPY_MAXDIMS];
    npy_intp csvb[NPY_MAXARGS][NPY_MAXDIMS];
    npy_intp vdims[NPY_MAXARGS][NPY_MAXDIMS];
    npy_intp offset, scoord[NPY_MAXDIMS], sdims[NPY_MAXDIMS];
    npy_intp bcoord[NPY_MAXDIMS];

    //This is a hack to make sure that mpi_tag is sync between
    //MPI-processes even when skipping iterations.
    //For each view-block calculation the following must be true:
    //narys * number of sub-view-blocks * 4 < 100.
    j=1;
    for(i=lastary->ndims-1; i>=0; --i)
    {
        int o = dim_iter_order[i];
        mpi_tag_svb += coord[o] * j;
        j *= lastary->blockdims[o];
    }
    mpi_tag_svb = mpi_tag_svb * 100 + mpi_tag;

    //We are adding a whole view-block operation to the svb-DAG
    //and we may therefore have to flush and free memory.
    int end_tag = *first_mpi_tag + DNPY_MAX_VB_IN_SVB_DAG * 100;
    if(mpi_tag_svb >= end_tag)
    {
        //Flush
        dag_svb_flush(1);
        //Resetting.
        for(i=0; i<narys; i++)
            vblocks[i].uid = -1;
        *first_mpi_tag = end_tag + DNPY_MAX_VB_IN_SVB_DAG * 100;
    }

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
            DNDTIME_SUM(tdelta2, dndt.calc_vblock)
        #endif
    }

    //Do computation for view-block.
    memset(scoord, 0, lastary->ndims * sizeof(npy_intp));
    memset(svboffset, 0, NPY_MAXARGS * NPY_MAXDIMS * sizeof(npy_intp));
    memset(csvb, 0, NPY_MAXARGS * NPY_MAXDIMS * sizeof(npy_intp));
    int svb_notfinished = 1;
    for(; svb_notfinished; mpi_tag_svb++)
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
        //The owner is the MPI-process containing the output svb.
        int owner = svbs[nin]->rank;
        if(owner == myrank)
        {
            //Add receive request to the svb-DAG.
            for(v=0; v<nin; v++, mpi_tag_svb++)
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
                dndop_comm *op = workbuf_nextfree;
                WORKBUF_INC(sizeof(dndop_comm));
                op->op = DNPY_BUF_RECV;
                op->optype = DNPY_COMM;
                op->narys = 1;
                op->refcount = 0;
                op->views[0] = views[v];
                op->svbs[0] = svbs[v];
                op->accesstypes[0] = DNPY_WRITE;
                op->mpi_tag = mpi_tag_svb;
                op->remote_rank = svbs[v]->rank;
                dndnode *node = workbuf_nextfree;
                WORKBUF_INC(sizeof(dndnode));
                node->op = (dndop *) op;
                node->op_ary_idx = 0;
                dag_svb_add(node, 1, 0);
            }
            for(v=nin; v<narys; v++)//Allocate memory for output
            {
                if(views[v]->base->ndims == 0)
                    continue;
                //The data is located locally.
                if(svbs[v]->rank == myrank)
                    continue;
                if(svbs[v]->data != NULL)
                    continue;

                svbs[v]->data = workbuf_nextfree;
                WORKBUF_INC(svbs[v]->nelem * views[v]->base->elsize);
            }
            //Add the apply request to the svb-DAG.
            dndop_ufunc *op = workbuf_nextfree;
            WORKBUF_INC(sizeof(dndop_ufunc));
            op->op = DNPY_UFUNC;
            op->optype = DNPY_NONCOMM;
            op->narys = narys;
            op->refcount = 0;
            op->nout = nout_arys;
            memcpy(op->views, views, narys * sizeof(dndview*));
            for(v=0; v<nin; v++)
                op->accesstypes[v] = DNPY_READ;
            for(v=nin; v<narys; v++)
                op->accesstypes[v] = DNPY_WRITE;
            op->func = func;
            op->funcdata = funcdata;
            Py_INCREF(PyOp);
            op->PyOp = PyOp;

            //Update the PyArray Objects to match the current
            //sub-view-block.
            for(v=0; v<narys; v++)
            {
                //Scalars is already updated.
                if(views[v]->base->ndims == 0)
                {
                    op->svbs[v] = NULL;
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
                //Apply update to apply-sub-view-block.
                s = views[v]->ndims * sizeof(npy_intp);
                memcpy(op->asvb[v].dims, csdims, s);
                memcpy(op->asvb[v].stride, sstride, s);
                op->asvb[v].offset = offset;
                op->svbs[v] = svbs[v];
            }

            //Add a new sub-view block DAG node.
            dndnode *nodes = workbuf_nextfree;
            WORKBUF_INC(narys * sizeof(dndnode));
            for(v=0; v<narys; v++)
            {
                nodes[v].op_ary_idx = v;
                nodes[v].op = (dndop *)op;
            }
            dag_svb_add(nodes, v, 0);
        }
        else
        {   //Add send request to the svb-DAG.
            for(v=0; v<nin; v++, mpi_tag_svb++)
            {
                if(views[v]->base->ndims == 0)
                    continue;
                //The data is not located on this proc.
                if(svbs[v]->rank != myrank)
                    continue;
                if(svbs[v]->comm_received_by == owner)
                    continue;

                //Communication has now been handled.
                svbs[v]->comm_received_by = owner;

                //Add a new sub-view block DAG node.
                dndop_comm *op = workbuf_nextfree;
                WORKBUF_INC(sizeof(dndop_comm));
                op->op = DNPY_SEND;
                op->optype = DNPY_COMM;
                op->narys = 1;
                op->refcount = 0;
                op->views[0] = views[v];
                op->svbs[0] = svbs[v];
                op->accesstypes[0] = DNPY_READ;
                op->mpi_tag = mpi_tag_svb;
                op->remote_rank = owner;
                dndnode *node = workbuf_nextfree;
                WORKBUF_INC(sizeof(dndnode));
                node->op_ary_idx = 0;
                node->op = (dndop *)op;
                dag_svb_add(node, 1, 0);
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
        for(j=0; j<vblocks[v].nsub; j++, mpi_tag_svb++)
        {
            dndsvb *svb = &vblocks[v].sub[j];
            //The owner is the MPI-process containing the output svb.
            int owner = vblocks[nin].sub[j].rank;
            //Add a new sub-view block DAG node.
            if(owner == myrank && svb->rank != myrank)
            {
                dndop_comm *op = workbuf_nextfree;
                WORKBUF_INC(sizeof(dndop_comm));
                op->narys = 1;
                op->refcount = 0;
                op->views[0] = views[v];
                op->svbs[0] = svb;
                op->accesstypes[0] = DNPY_READ;
                op->mpi_tag = mpi_tag_svb;
                op->op = DNPY_BUF_SEND;
                op->optype = DNPY_COMM;
                op->remote_rank = svb->rank;
                dndnode *node = workbuf_nextfree;
                WORKBUF_INC(sizeof(dndnode));
                node->op_ary_idx = 0;
                node->op = (dndop *)op;
                dag_svb_add(node, 1, 0);
            }
            else if(owner != myrank && svb->rank == myrank)
            {
                dndop_comm *op = workbuf_nextfree;
                WORKBUF_INC(sizeof(dndop_comm));
                op->narys = 1;
                op->refcount = 0;
                op->views[0] = views[v];
                op->svbs[0] = svb;
                op->accesstypes[0] = DNPY_WRITE;
                op->mpi_tag = mpi_tag_svb;
                op->op = DNPY_RECV;
                op->optype = DNPY_COMM;
                op->remote_rank = owner;
                dndnode *node = workbuf_nextfree;
                WORKBUF_INC(sizeof(dndnode));
                node->op_ary_idx = 0;
                node->op = (dndop *)op;
                dag_svb_add(node, 1, 0);
            }
        }

    #ifdef DNDY_TIME
        DNDTIME_SUM(tdelta, dndt.ufunc_svb)
    #endif
}

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
        MPI_Type_free(&tmp_type);
    }

    /* Free allocated data */
    for(n = 0; n < ndims; n++)
    {
        MPI_Type_free(&derived_type[n]);
        free(displacements[n]);
    }

    MPI_Type_commit(dtype);
    return 0;

} /* create_mpi_dtype */

/*===================================================================
 *
 * Message handler for TIME_RESET.
 * Private.
 */
static void do_TIME_RESET(void)
{
    //Print debug info.
    #ifdef DISTNUMPY_DEBUG
        printf("TIME_RESET\n");
    #endif

    dag_svb_flush(1);
    MPI_Barrier(MPI_COMM_WORLD);

    memset(&dndt, 0, sizeof(dndtime));
    DNDTIME(totaldelta);

} /* do_TIME_RESET */

/*===================================================================
 *
 * Message handler for TIME_GETDICT.
 * Private.
 */
static PyObject *
do_TIME_GETDICT(void)
{
    Py_ssize_t i;
    //Print debug info.
    #ifdef DISTNUMPY_DEBUG
        printf("TIME_GETDICT\n");
    #endif

    dag_svb_flush(1);

    #ifdef DNDY_TIME
        unsigned long long tdelta;
        DNDTIME(tdelta);
    #endif

    MPI_Barrier(MPI_COMM_WORLD);

    #ifdef DNDY_TIME
        DNDTIME_SUM(tdelta, dndt.final_barrier)
    #endif

    DNDTIME_SUM(totaldelta, dndt.total);

    if(myrank == 0)
    {
        dndtime tmp = dndt;
        PyObject* ret = PyTuple_New(worldsize);

        for(i=0; i<worldsize; ++i)
        {
            PyObject* dict = PyDict_New();
            if(i > 0)//tmp is already filled for rank zero.
                MPI_Recv(&tmp, sizeof(dndtime), MPI_BYTE, i, 0,
                         MPI_COMM_WORLD, MPI_STATUS_IGNORE);

            PyDict_SetItemString(dict, "total",
                   PyLong_FromUnsignedLongLong(tmp.total));
            PyDict_SetItemString(dict, "dag_svb_flush",
                   PyLong_FromUnsignedLongLong(tmp.dag_svb_flush));
            PyDict_SetItemString(dict, "dag_svb_rm",
                   PyLong_FromUnsignedLongLong(tmp.dag_svb_rm));
            PyDict_SetItemString(dict, "apply_ufunc",
                   PyLong_FromUnsignedLongLong(tmp.apply_ufunc));
            PyDict_SetItemString(dict, "ufunc_comm",
                   PyLong_FromUnsignedLongLong(tmp.ufunc_comm));
            PyDict_SetItemString(dict, "comm_init",
                   PyLong_FromUnsignedLongLong(tmp.comm_init));
            PyDict_SetItemString(dict, "arydata_free",
                   PyLong_FromUnsignedLongLong(tmp.arydata_free));
            PyDict_SetItemString(dict, "reduce_1d",
                   PyLong_FromUnsignedLongLong(tmp.reduce_1d));
            PyDict_SetItemString(dict, "reduce_nd",
                   PyLong_FromUnsignedLongLong(tmp.reduce_nd));
            PyDict_SetItemString(dict, "reduce_nd_apply",
                   PyLong_FromUnsignedLongLong(tmp.reduce_nd_apply));
            PyDict_SetItemString(dict, "zerofill",
                   PyLong_FromUnsignedLongLong(tmp.zerofill));
            PyDict_SetItemString(dict, "ufunc_svb",
                   PyLong_FromUnsignedLongLong(tmp.ufunc_svb));
            PyDict_SetItemString(dict, "dag_svb_add",
                   PyLong_FromUnsignedLongLong(tmp.dag_svb_add));
            PyDict_SetItemString(dict, "calc_vblock",
                   PyLong_FromUnsignedLongLong(tmp.calc_vblock));
            PyDict_SetItemString(dict, "arydata_malloc",
                   PyLong_FromUnsignedLongLong(tmp.arydata_malloc));
            PyDict_SetItemString(dict, "msg2slaves",
                   PyLong_FromUnsignedLongLong(tmp.msg2slaves));
            PyDict_SetItemString(dict, "final_barrier",
                   PyLong_FromUnsignedLongLong(tmp.final_barrier));
            PyDict_SetItemString(dict, "mem_reused",
                   PyLong_FromLong(tmp.mem_reused));
            PyDict_SetItemString(dict, "nconnect",
                   PyLong_FromLong(tmp.nconnect));
            PyDict_SetItemString(dict, "nconnect_max",
                   PyLong_FromLong(tmp.nconnect_max));
            PyDict_SetItemString(dict, "napply",
                   PyLong_FromLong(tmp.napply));
            PyDict_SetItemString(dict, "nflush",
                   PyLong_FromLong(tmp.nflush));
            PyTuple_SetItem(ret, i, dict);
        }
        return ret;
    }
    else
    {
        MPI_Send(&dndt, sizeof(dndtime), MPI_BYTE, 0, 0, MPI_COMM_WORLD);
    }

    Py_RETURN_NONE;
} /* do_TIME_GETDICT */

/*===================================================================
 *
 * Message handler for DNPY_INIT_PGRID.
 * Private.
 */
static void do_INIT_PGRID(int pgrid[NPY_MAXDIMS*NPY_MAXDIMS])
{
    int j,i;
    initmsg_not_handled = 0;
    assert(ndndarrays == 0);

    //Save the cart_dim_sizes and compute the cart_dim_strides.
    for(i=0; i<NPY_MAXDIMS; i++)
    {
        int ndims = i+1;
        for(j=0; j<ndims; j++)
            cart_dim_sizes[i][j] = pgrid[i*NPY_MAXDIMS+j];

        //Set cartesian information.
        memset(cart_dim_strides[i], 0, ndims*sizeof(int));

        //Compute strides for all dims. Using row-major like MPI.
        //A 2x2 process grid looks like:
        //    coord (0,0): rank 0.
        //    coord (0,1): rank 1.
        //    coord (1,0): rank 2.
        //    coord (1,1): rank 3.
        for(j=0; j<ndims; j++)
        {
            int stride = 1, s;
            for(s=j+1; s<ndims; s++)
                stride *= cart_dim_sizes[i][s];
            cart_dim_strides[i][j] = stride;
        }
    }
    //Print debug info.
    #ifdef DISTNUMPY_DEBUG
        printf("INIT_PROC_GRID - process grid (the first three): ");
        for(i=0; i<3; i++)
        {
            printf("(%d)[", i+1);
            for(j=0; j<i; j++)
                printf("%d,", cart_dim_sizes[i][j]);
            printf("%d] ", cart_dim_sizes[i][i]);
        }
        printf("\n");
    #endif

}/* do_DNPY_INITMSG */

/*===================================================================
 *
 * Message handler for CREATE_ARRAY.
 * Private.
 */
static dndview* do_CREATE_ARRAY(dndarray *ary, dndview *view)
{
    int ndims = ary->ndims;
    int *cdims = cart_dim_sizes[ndims-1];
    npy_intp i;
    int cartcoord[NPY_MAXDIMS];

    ++ndndarrays;

    //Save array uid.
    ary->uid = view->uid;

    //Allocate and copy the array to the views's base.
    view->base = malloc(sizeof(dndarray));
    memcpy(view->base, ary, sizeof(dndarray));
    ary = view->base;//Use the new pointer.

    //Append the array to the lisked list when statistic is defined.
    #ifdef DNPY_STATISTICS
        ary->prev = NULL;
        ary->next = rootarray;
        rootarray = ary;
        if(ary->next != NULL)
        {
            assert(ary->next->prev == NULL);
            ary->next->prev = rootarray;
        }
    #endif

    //Get cartesian coords.
    rank2cart(ndims, myrank, cartcoord);

    //Accumulate the total number of local sizes and save it.
    npy_intp localsize = 1;
    ary->nblocks = 1;
    for(i=0; i < ary->ndims; i++)
    {
        if(ary->onerank < 0)
            ary->localdims[i] = dnumroc(ary->dims[i], blocksize,
                                        cartcoord[i], cdims[i], 0);
        else
            ary->localdims[i] = (ary->onerank==myrank)?ary->dims[i]:0;

        localsize *= ary->localdims[i];
        ary->localblockdims[i] = ceil(ary->localdims[i] /
                                      (double) blocksize);
        ary->blockdims[i] = ceil(ary->dims[i] / (double) blocksize);
        ary->nblocks *= ary->blockdims[i];
    }
    ary->localsize = localsize;
    if(ary->localsize == 0)
    {
        memset(ary->localdims, 0, ary->ndims * sizeof(npy_intp));
        memset(ary->localblockdims, 0, ary->ndims * sizeof(npy_intp));
    }
    if(ary->nblocks == 0)
        memset(ary->blockdims, 0, ary->ndims * sizeof(npy_intp));

    //Allocate the root nodes array.
    ary->rootnodes = malloc(ary->nblocks * sizeof(dndnode*));
    for(i=0; i<ary->nblocks; i++)
        ary->rootnodes[i] = NULL;

    //The memory allocation is delayed to the point where it is used.
    ary->data = NULL;

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
    if(view->nblocks == 0)
        memset(view->blockdims, 0, ndims * sizeof(npy_intp));

    //Print debug info.
    #ifdef DISTNUMPY_DEBUG
        printf("CREATE_ARRAY - uid: %ld, dtype: %s(%d),"
               " nelem: %ld, ndims: %d, dims: {",
               (long) view->uid,
               types_numpy2str(ary->dtype), ary->elsize,
               (long)localsize, ndims);
        for(i=0; i<ndims-1; i++)
            printf("%ld ", (long)ary->dims[i]);
        printf("%ld}, localdims: {", (long)ary->dims[ndims-1]);
        for(i=0; i<ndims-1; i++)
            printf("%ld ", (long)ary->localdims[i]);
        printf("%ld}, \n", (long)ary->localdims[ndims-1]);
    #endif

    //Save the new view.
    return put_dndview(view);

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
    dndview *view = get_dndview(uid);

    dndop *op = workbuf_nextfree;
    WORKBUF_INC(sizeof(dndop));
    op->op = DNPY_DESTROY_ARRAY;
    op->optype = DNPY_NONCOMM;
    op->narys = 1;
    op->refcount = 0;
    op->views[0] = view;
    op->svbs[0] = NULL;//Whole array.
    op->accesstypes[0] = DNPY_WRITE;

    dndnode *node = workbuf_nextfree;
    WORKBUF_INC(sizeof(dndnode));
    node->op = op;
    node->op_ary_idx = 0;
    dag_svb_add(node, 1, 0);

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

    //Add the base to the new view.
    assert(orgview->base->refcount > 0);
    newview->base = orgview->base;
    newview->base->refcount++;

    //Save the view and change pointer to its new location.
    newview = put_dndview(newview);

    //Compute size of view-blocks.
    if(newview->nblocks == 0)//The view is empty
    {
        memset(newview->blockdims, 0, newview->ndims * sizeof(npy_intp));
    }
    else
    {
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
    char *data;
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

    dag_svb_flush(1);

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
    delayed_array_allocation(view->base);
    data = view->base->data + offset*view->base->elsize;

#ifndef DNPY_SPMD
    if(vblock.sub[b].rank == 0)//Local copying.
    {
        if(myrank == 0)
        {
            if(Direction)
                memcpy(data, item, view->base->elsize);
            else
                memcpy(item, data, view->base->elsize);
        }
    }
    else if(myrank == 0)
    {
        if(Direction)
            MPI_Ssend(item, 1, view->base->mpi_dtype, vblock.sub[b].rank,
                     0, MPI_COMM_WORLD);
        else
            MPI_Recv(item, 1, view->base->mpi_dtype, vblock.sub[b].rank,
                     0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    }
    else if(myrank == vblock.sub[b].rank)
    {
        if(Direction)
            MPI_Recv(data, 1, view->base->mpi_dtype, 0, 0,
                     MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        else
            MPI_Ssend(data, 1, view->base->mpi_dtype, 0, 0,
                     MPI_COMM_WORLD);
    }
#else
    if(Direction)
    {
        if(vblock.sub[b].rank == 0)//Local copying.
        {
            if(myrank == 0)
                memcpy(data, item, view->base->elsize);
        }
        else if(myrank == 0)
        {
            MPI_Ssend(item, 1, view->base->mpi_dtype, vblock.sub[b].rank,
                     0, MPI_COMM_WORLD);
        }
        else if(myrank == vblock.sub[b].rank)
        {
            MPI_Recv(data, 1, view->base->mpi_dtype, 0, 0,
                     MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        }
    }
    else
    {
        if(vblock.sub[b].rank == myrank)//Local copying.
            memcpy(item, data, view->base->elsize);

        MPI_Bcast(item, view->base->elsize, MPI_BYTE, vblock.sub[b].rank,
                  MPI_COMM_WORLD);
    }
#endif

    dag_svb_flush(1);//Will cleanup the used sub-view-blocks.

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
    npy_intp i, j, s, v;
    npy_intp coord[NPY_MAXDIMS];
    int dim_iter_order[NPY_MAXDIMS];
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
        {
            //Create a dummy view for the scalar.
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
            dummy.base->refcount = 1;
            dummy.base->localsize = 1;
            dummy.base->nblocks = 1;
            dummy.base->rootnodes = malloc(sizeof(dndnode *));
            dummy.base->rootnodes[0] = NULL;
            dummy.base->data = malloc(scalar_size);
            memcpy(dummy.base->data, scalar, scalar_size);
            dummy.base->elsize = scalar_size;
            views[i] = put_dndview(&dummy);
            //Append the array to the lisked list when statistic is defined.
            #ifdef DNPY_STATISTICS
                dummy.base->prev = NULL;
                dummy.base->next = rootarray;
                rootarray = dummy.base;
                if(dummy.base->next != NULL)
                {
                    assert(dummy.base->next->prev == NULL);
                    dummy.base->next->prev = rootarray;
                }
            #endif
        }
    dndview *lastary = views[narys-1];

    //Need to make sure that the output view is not empty.
    if(lastary->nblocks == 0)
    {
        //Clean up Scalar dummy.
        for(i=0; i<narys; i++)
            if(views[i]->base->ndims == 0)//Scalar dummy.
                do_DESTROY_ARRAY(views[i]->uid);
        return;
    }

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

    //The first MPI tag after a flush.
    mpi_tag = (1+mpi_tag) % DNPY_MAX_MPI_TAG;
    int first_mpi_tag = mpi_tag;


    //Find max span
    npy_intp max_span = 0;
    for(v=0; v<narys; v++)
    {
        dndview *view = views[v];
        for(i=0; i<view->ndims; i++)
        {
            dndslice *t = &view->slice[idx_v2s(view, i)];
            if(t->nsteps == PseudoIndex)
                continue;
            npy_intp span = ceil(t->start /(double) blocksize) + t->step-1;
            max_span = MAX(max_span,span);

            //Number of steps must be divisible by the number of
            //MPI processes in that dimension.
            if(cart_dim_sizes[view->base->ndims-1][idx_v2b(view,i)]
               % t->step != 0)
            {
                max_span = worldsize + 1000;
                break;
            }
        }
    }

    //If the process distribution is not identical for each dimension
    //we have to force a max_span that 'spans' all coordinates,
    for(i=0; i<lastary->ndims; i++)
    {
        int nproc = cart_dim_sizes[lastary->ndims-1][i];
        for(v=0; v<narys; v++)
        {

            dndview *view = views[v];
            if(i >= view->ndims ||
               cart_dim_sizes[view->base->ndims-1][idx_v2b(view,i)] != nproc)
            {
                max_span = worldsize + 1000;
                break;
            }
        }
    }

    //Check if some dimensions involves all MPI processes.
    int all_involved[NPY_MAXDIMS];
    npy_intp involvement_coord[NPY_MAXDIMS];
    for(i=0; i<lastary->ndims; i++)
    {
        if(max_span*2 >= cart_dim_sizes[lastary->ndims-1][i] - 2)
            all_involved[i] = 1; //True.
        else
            all_involved[i] = 0; //False.

        involvement_coord[i] = 0;
    }

    //Compute start coord.
    int coord_start[NPY_MAXDIMS];
    int pcoord[NPY_MAXDIMS];
    rank2cart(lastary->ndims, myrank, pcoord);
    for(i=0; i<lastary->ndims; i++)
    {
        if(all_involved[i])
        {
            coord[i] = 0;
        }
        else
        {
            coord_start[i] = MAX(pcoord[i] - max_span, 0);
            coord_start[i] = MIN(coord_start[i],lastary->blockdims[i]-1);
            coord[i] = coord_start[i];
        }
    }

    //Loop over coordinates that may involve the MPI-process
    //NB: coord consist of only visible block coordinates.
    while(notfinished)
    {
        //Lets check if the MPI-process can skip this iteration.
        if(!ufunc_skip_coord(coord, narys, views, lastary))
            //Handle the coordinate.
            ufunc_handle_coord(coord, narys, nout_arys, views, vblocks,
                               PyOp, func, funcdata, &first_mpi_tag,
                               dim_iter_order);

        //Iterate coord to the next vblock.
        for(j=lastary->ndims-1; j >= 0; j--)
        {
            int o = dim_iter_order[j];//Using the computed order.

            //We simply iterates one vblock when all are involved.
            if(all_involved[o])
            {
                coord[o] += 1;
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
            else
            {
                //The involvement of the border coordinates may have to
                //be shorten.
                npy_intp length = max_span * 2 + 1;
                if(coord[o] < max_span * 2 + 1 && pcoord[o] < max_span)
                    length -= max_span - pcoord[o];

                //First we calculate the coord increase.
                involvement_coord[o] += 1;
                if(involvement_coord[o] >= length)
                {
                    involvement_coord[o] = 0;
                    coord[o] += (cart_dim_sizes[lastary->ndims-1][o])
                                - max_span*2;
                }
                else
                    coord[o] += 1;

                //Then we check for overflow.
                if(coord[o] >= lastary->blockdims[o])
                {
                    //We are finished, if wrapping around.
                    if(j == 0)
                    {
                        notfinished = 0;
                        break;
                    }
                    coord[o] = coord_start[o];
                    involvement_coord[o] = 0;
                }
                else
                    break;
            }
        }
    }

    //Sync mpi tags
    mpi_tag = (mpi_tag + lastary->nblocks * 100) % DNPY_MAX_MPI_TAG;
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
    int i;
    npy_intp j;
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

    dag_svb_flush(1);

    //At the moment we don't support views.
    if(in_view->alterations != 0)
    {
        fprintf(stderr, "UFUNC_REDUCE: No view support\n");
        MPI_Abort(MPI_COMM_WORLD, -1);
    }

    #ifdef DNDY_TIME
        unsigned long long tdelta;
        DNDTIME(tdelta);
    #endif

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
        //NB: in SPMD all ranks are masters.
    #ifndef DNPY_SPMD
        if(myrank == 0)
    #else
        if(1)
    #endif
            local_scalar = PyArray_SimpleNewFromData(0, NULL,
                           out_scalar_dtype, out_scalar_address);
        else//And the rest just use a new location.
            local_scalar = PyArray_SimpleNew(0, NULL, out_scalar_dtype);

        if(in_ary->localdims[0] > 0)
        {
            tmpobj = PyArray_SimpleNewFromData(1, &in_ary->localsize,
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
                                i, cart_dim_sizes[0][0], 0))
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

    #ifdef DNPY_SPMD
        MPI_Bcast(out_scalar_address, out_scalar_dtypesize, MPI_BYTE, 0,
                  MPI_COMM_WORLD);
    #endif

        #ifdef DNDY_TIME
            DNDTIME_SUM(tdelta, dndt.reduce_1d)
        #endif

        return;
    }

    dndview *out_view = get_dndview(out_ary_uid);
    dndarray *out_ary = out_view->base;
    PyObject *subres = NULL;

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

    #ifdef DNDY_TIME
        unsigned long long tdelta2;
        DNDTIME(tdelta2);
    #endif

    if(in_ary->localsize > 0)
    {
        PyObject *tmp = PyArray_SimpleNewFromData(in_ary->ndims,
                                                  in_ary->localdims,
                                                  in_ary->dtype,
                                                  in_ary->data);

        //All MPI-processes computes its own sub-result.
        subres = PyObject_CallMethod(PyOp, "reduce", "(O,i)",
                                               tmp, axis);
        //Clean up.
        Py_DECREF(tmp);
    }

    #ifdef DNDY_TIME
        DNDTIME_SUM(tdelta2, dndt.reduce_nd_apply)
    #endif

    int cart_coords[NPY_MAXDIMS];
    rank2cart(in_ary->ndims, myrank, cart_coords);

    char receiver = cart_coords[axis] == 0?1:0;
    if(receiver)//MPI-process is a receiver.
    {
        npy_intp recv_dims[NPY_MAXDIMS];
        memset(recv_dims, 0, out_ary->ndims*sizeof(intp));

        //Handle one sender at a time.
        for(cart_coords[axis]=1;
            cart_coords[axis] < cart_dim_sizes[in_ary->ndims-1][axis];
            cart_coords[axis]++)
        {
            int rank = cart2rank(in_ary->ndims, cart_coords);
            int not_empty = 1;
            //Compute the dim of the receiving array.
            int count = 0;
            for(i=0; i < in_ary->ndims; i++)
            {
                npy_intp d = dnumroc(in_ary->dims[i], blocksize,
                                     cart_coords[i],
                                     cart_dim_sizes[in_ary->ndims-1][i]
                                     , 0);
                if(d <= 0)//Its empty.
                {
                    not_empty = 0;
                    break;
                }
                if(i != axis)
                    recv_dims[count++] = d;
            }

            if(not_empty)
            {
                //Receive the sub-result from the sender into 'tmpres'.
                PyObject *tmpres = PyArray_SimpleNew(out_ary->ndims,
                                   recv_dims, out_ary->dtype);

                MPI_Recv(PyArray_DATA(tmpres), PyArray_SIZE(tmpres) *
                         out_ary->elsize, MPI_BYTE, rank, 0,
                         MPI_COMM_WORLD, MPI_STATUS_IGNORE);

                //We use ufunc to "reduce" the two arrays over the
                //non-existing 'axis' dimension.
                PyObject *t = PyObject_CallFunction(PyOp, "(O,O,O)",
                                                    subres, tmpres,
                                                    subres);
                Py_DECREF(t);
                Py_DECREF(tmpres);
            }
        }
    }
    else if(subres != NULL)//MPI-process is a sender.
    {

        //Send subresult to receiver.
        cart_coords[axis] = 0;
        MPI_Send(PyArray_DATA(subres),
                 PyArray_SIZE(subres) * PyArray_ITEMSIZE(subres),
                 MPI_BYTE, cart2rank(in_ary->ndims, cart_coords), 0,
                 MPI_COMM_WORLD);
    }

    //Make sure that we are clear of previously used mpi tags.
    mpi_tag = (1+mpi_tag) % DNPY_MAX_MPI_TAG;

    //Iterate over input blocks local to the MPI process.
    //Only receivers have input blocks.
    if(receiver && in_view->base->localsize > 0)
    {
        npy_intp coord[NPY_MAXDIMS], gcoord[NPY_MAXDIMS];
        char notfinished = 1;
        memset(coord, 0, in_view->ndims * sizeof(npy_intp));
        rank2cart(in_ary->ndims, myrank, cart_coords);
        while(notfinished)
        {
            dndsvb *svb = workbuf_nextfree;
            WORKBUF_INC(sizeof(dndsvb));
            svb->rank = myrank;
            svb->data = PyArray_DATA(subres);
            svb->nelem = 1;
            svb->comm_offset = 0;
            //Compute the start, nsteps & stride.
            j=out_view->ndims-1;
            npy_intp stride = 1;//Not in bytes.
            for(i=in_view->ndims-1; i >= 0; i--)//Row-major.
            {
                svb->start[i] = coord[i] * blocksize;
                if(i == axis)
                {
                    svb->nsteps[i] = 1;
                    svb->stride[i] = 1;
                }
                else
                {
                    svb->nsteps[i] = MIN(blocksize,
                                         PyArray_DIM(subres, j) -
                                         svb->start[i]);
                    svb->stride[i] = stride;
                    stride *= in_view->base->localdims[i];
                    svb->comm_offset += svb->start[i] * svb->stride[i];
                    svb->nelem *= svb->nsteps[i];
                    j--;
                }
            }
            //Convert to bytes.
            svb->comm_offset *= in_view->base->elsize;

            //Add a new sub-view block DAG node.
            dndop_comm *op = workbuf_nextfree;
            WORKBUF_INC(sizeof(dndop_comm));
            op->narys = 1;
            op->refcount = 0;
            op->views[0] = in_view;
            op->svbs[0] = svb;
            op->accesstypes[0] = DNPY_READ;
            op->op = DNPY_REDUCE_SEND;
            op->optype = DNPY_COMM;

            //Convert local input block coordinate to global input
            //block coordinate.
            int *cdimsizes = cart_dim_sizes[in_view->ndims-1];
            for(i=0; i<in_view->ndims; i++)
                gcoord[i] = cart_coords[i] + //process offset.
                            coord[i] *       //block index.
                            cdimsizes[i];    //jump per block.

            //Compute the sub-view-block's root node.
            op->svbs[0]->rootnode = in_view->base->rootnodes;
            j = 1;
            for(i=in_view->base->ndims-1; i >= 0; i--)
            {
                op->svbs[0]->rootnode += gcoord[i] * j;
                j *= in_view->base->blockdims[i];
            }

            //Convert global output block coordinate to global output
            //block coordinate.
            j=0;
            for(i=0; i<in_view->ndims; i++)
                if(axis != i)
                    gcoord[j++] = gcoord[i];

            //Find process cartesian coordinate and its MPI rank.
            int pcoords[NPY_MAXDIMS];
            cdimsizes = cart_dim_sizes[out_view->ndims-1];
            for(i=0; i<out_view->ndims; i++)
                pcoords[i] = gcoord[i] % cdimsizes[i];
            int rank = cart2rank(out_view->ndims, pcoords);
            op->remote_rank = rank;

            //Use a mpi_tag based on the global output array block
            //coordinate. Row-Major.
            npy_intp tag = 0; j=1;
            for(i=out_view->ndims-1; i>=0; i--)
            {
                tag += gcoord[i] * j;
                j *= out_view->blockdims[i];
            }
            op->mpi_tag = tag + mpi_tag;

            dndnode *node = workbuf_nextfree;
            WORKBUF_INC(sizeof(dndnode));
            node->op_ary_idx = 0;
            node->op = (dndop *)op;
            dag_svb_add(node, 1, 1);

            //Iterate coord one local block.
            for(j=in_view->ndims-1; j >= 0; j--)
            {
                //We ignore the 'axis' dimension - it stays on zero.
                if(j != axis)
                {
                    if(++coord[j] >= in_view->base->localblockdims[j])
                    {
                        //We are finished, if wrapping around.
                        if(j == 0)
                        {
                            notfinished = 0;
                            break;
                        }
                        coord[j] = 0;//Start coord.
                    }
                    else
                        break;
                }
                else if(axis == 0)//We are finished, if wrapping around.
                {
                    notfinished = 0;
                    break;
                }
            }
        }
    }

    if(out_view->base->localsize > 0)
    {
        rank2cart(out_view->ndims, myrank, cart_coords);

        //Iterate over output blocks local to the MPI process.
        npy_intp coord[NPY_MAXDIMS], gcoord[NPY_MAXDIMS];
        char notfinished = 1;
        memset(coord, 0, out_view->ndims * sizeof(npy_intp));
        while(notfinished)
        {
            dndsvb *svb = workbuf_nextfree;
            WORKBUF_INC(sizeof(dndsvb));
            svb->rank = myrank;
            svb->data = NULL;
            svb->nelem = 1;
            svb->comm_offset = 0;
            j = 1;//For stride, not in bytes.
            //Compute the start, nsteps & stride.
            for(i=out_view->ndims-1; i >= 0; i--)//Row-major.
            {
                svb->start[i] = coord[i] * blocksize;
                svb->nsteps[i] = MIN(blocksize, out_view->base->localdims[i] - svb->start[i]);
                svb->stride[i] = j;
                j *= out_view->base->localdims[i];
                svb->comm_offset += svb->start[i] * svb->stride[i];
                svb->nelem *= svb->nsteps[i];
                assert(svb->nsteps[i] > 0);
            }
            //Convert to bytes.
            svb->comm_offset *= out_view->base->elsize;

            //Add a new sub-view block DAG node.
            dndop_comm *op = workbuf_nextfree;
            WORKBUF_INC(sizeof(dndop_comm));
            op->narys = 1;
            op->refcount = 0;
            op->views[0] = out_view;
            op->svbs[0] = svb;
            op->accesstypes[0] = DNPY_WRITE;
            op->op = DNPY_RECV;
            op->optype = DNPY_COMM;

            //Convert local output block coordinate to global output
            //block coordinate.
            int *cdimsizes = cart_dim_sizes[out_view->ndims-1];
            for(i=0; i<out_view->ndims; i++)
                gcoord[i] = cart_coords[i] + //process offset.
                            coord[i] *       //block index.
                            cdimsizes[i];    //jump per block.

            //Compute the sub-view-block's root node.
            op->svbs[0]->rootnode = out_view->base->rootnodes;
            j = 1;
            for(i=out_view->base->ndims-1; i >= 0; i--)
            {
                op->svbs[0]->rootnode += gcoord[i] * j;
                j *= out_view->base->blockdims[i];
            }

            //Convert global output block coordinate to global input
            //block coordinate.
            j=out_view->ndims-1;
            for(i=in_view->ndims-1; i>=0; i--)
                if(axis == i)
                {
                    gcoord[i] = 0;
                    break;
                }
                else
                {
                    gcoord[i] = gcoord[j];
                    j--;
                }

            //Find process cartesian coordinate and its MPI rank.
            int pcoords[NPY_MAXDIMS];
            cdimsizes = cart_dim_sizes[in_view->ndims-1];
            for(i=0; i<in_view->ndims; i++)
                pcoords[i] = gcoord[i] % cdimsizes[i];
            int rank = cart2rank(in_view->ndims, pcoords);
            op->remote_rank = rank;

            //Use a mpi_tag based on the global output array block
            //coordinate. Row-Major.
            npy_intp tag = 0; j=1;
            for(i=in_view->ndims-1; i>=0; i--)
            {
                if(i != axis)
                {
                    tag += gcoord[i] * j;
                    j *= in_view->blockdims[i];
                }
            }
            op->mpi_tag = tag + mpi_tag;

            dndnode *node = workbuf_nextfree;
            WORKBUF_INC(sizeof(dndnode));
            node->op_ary_idx = 0;
            node->op = (dndop *)op;
            dag_svb_add(node, 1, 1);

            //Iterate coord one local block.
            for(j=out_view->ndims-1; j >= 0; j--)
            {
                if(++coord[j] >= out_view->base->localblockdims[j])
                {
                    //We are finished, if wrapping around.
                    if(j == 0)
                    {
                        notfinished = 0;
                        break;
                    }
                    coord[j] = 0;//Start coord.
                }
                else
                    break;
            }
        }
    }

    //Sync mpi_tag.
    mpi_tag = (mpi_tag+out_view->nblocks) % DNPY_MAX_MPI_TAG;

    #ifdef DNDY_TIME
        DNDTIME_SUM(tdelta, dndt.reduce_nd)
    #endif

    dag_svb_flush(1);

    //Cleanup
    Py_XDECREF(subres);
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

    dag_svb_flush(1);

    #ifdef DNDY_TIME
        unsigned long long tdelta;
        DNDTIME(tdelta);
    #endif

    delayed_array_allocation(view->base);
    memset(view->base->data, 0, view->base->localsize *
                                view->base->elsize);

    #ifdef DNDY_TIME
        DNDTIME_SUM(tdelta, dndt.zerofill)
    #endif

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

    dag_svb_flush(1);

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

    delayed_array_allocation(view->base);

    /* Do operation */
    MPI_Status status;
    switch(op) {
        case DNPY_DATAFILL:
            MPI_File_read_all(fh, view->base->data, view->base->localsize,
                              view->base->mpi_dtype, &status);
            break;
        case DNPY_DATADUMP:
            MPI_File_write_all(fh, view->base->data, view->base->localsize,
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

    dag_svb_flush(1);
    delayed_array_allocation(in->base);
    delayed_array_allocation(out->base);

    //Create the window for one-sided communication.
    MPI_Win comm_win;
    MPI_Win_create(in->base->data, in->base->localsize *
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
                           cart_dim_sizes[1][0], 0);
        ldims[1] = dnumroc(in->base->dims[1], blocksize, pcoord[1],
                           cart_dim_sizes[1][1], 0);

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

    dag_svb_flush(1);

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

    delayed_array_allocation(A->base);
    delayed_array_allocation(B->base);
    delayed_array_allocation(C->base);

    //Process index of row and column that hold current row and column
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
    npy_intp localdims[2];

    //We need to compute our own local dimensions since a MPI-process
    //may be involved in the computation even though it has no elements
    //of a particular array.
    localdims[0] = dnumroc(C->base->dims[0], blocksize, mycoord[0],
                           cart_dim[0], 0);
    localdims[1] = dnumroc(C->base->dims[1], blocksize, mycoord[1],
                           cart_dim[1], 0);

    //Leading dimensions.
    int lda = MAX(dnumroc(A->base->dims[1], blocksize, mycoord[1],
                  cart_dim[1], 0), 1);
    int ldb = MAX(dnumroc(B->base->dims[1], blocksize, mycoord[1],
                  cart_dim[1], 0), 1);
    int ldc = MAX(dnumroc(C->base->dims[1], blocksize, mycoord[1],
                  cart_dim[1], 0), 1);

    //malloc temp space
    double *temp  = malloc(C->base->elsize);
    double *work0 = malloc(localdims[0] * iwrk *
                           C->base->elsize);
    double *work1 = malloc(iwrk * localdims[1] *
                           C->base->elsize);


    for (kk=0; kk<A->base->dims[1]; kk+=iwrk)
    {
        iwrk = MIN(blocksize, A->base->dims[1] - kk);

        //Copy iwrk'th column to work-array.
        if(mycoord[1] == icur[1])
        {
            double *Adata = ((double *)A->base->data)+jj;
            for(i=0; i<localdims[0]; i++)
                for(j=0; j<iwrk; j++)
                    work0[i*iwrk+j] = Adata[i*lda+j];
        }
        //Copy iwrk'th row to work-array.
        if(mycoord[0] == icur[0])
        {
            double *Bdata = ((double *)B->base->data)+ldb*ii;
            for(i=0; i<iwrk; i++)
                for(j=0; j<localdims[1]; j++)
                    work1[i*localdims[1]+j] = Bdata[i*ldb+j];
        }
        if(cart_dim[1] > 1)
        {
            //Ring-broadcast horizontal.
            if(mycoord[1] != icur[1])
            {
                //We have to recv since we are not icur.
                tcoord[0] = mycoord[0];
                tcoord[1] = (mycoord[1]-1+cart_dim[1])%cart_dim[1];
                MPI_Recv(work0, localdims[0] * iwrk,
                         A->base->mpi_dtype, cart2rank(2,tcoord),
                         MPI_ANY_TAG, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            }
            //Send the iwrk'th column to the right neighbour but
            //make sure we don't wrap around.
            if((mycoord[1]+1)%cart_dim[1] != icur[1])
            {
                tcoord[0] = mycoord[0];
                tcoord[1] = (mycoord[1]+1)%cart_dim[1];
                MPI_Send(work0, localdims[0] * iwrk,
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
                MPI_Recv(work1, iwrk * localdims[1],
                         A->base->mpi_dtype, cart2rank(2,tcoord),
                         MPI_ANY_TAG, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            }
            //Send the iwrk'th row to the lower neighbour but
            //make sure we don't wrap around.
            if((mycoord[0]+1)%cart_dim[0] != icur[0])
            {
                tcoord[0] = (mycoord[0]+1)%cart_dim[0];
                tcoord[1] = mycoord[1];
                MPI_Send(work1, iwrk * localdims[1],
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
                             work1, ldb,
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
                        ldb * sizeof(double),
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
 * Returns True if there is data conflict/overlap.
 * Private.
 */
static char dndnode_conflict(const dndop *A, const int Aidx,
                             const dndop *B, const int Bidx)
{
    npy_intp d;
    char Aaccesstype = A->accesstypes[Aidx];
    dndarray *Abase = A->views[Aidx]->base;
    char Baccesstype = B->accesstypes[Bidx];
    dndarray *Bbase = B->views[Bidx]->base;
    dndsvb *Asvb = A->svbs[Aidx];
    dndsvb *Bsvb = B->svbs[Bidx];

    if(Abase->uid == Bbase->uid)
        if(Aaccesstype == DNPY_WRITE || Baccesstype == DNPY_WRITE)
        {
            if(Asvb == NULL || Bsvb == NULL)
                return 1;//Depend on the whole array.

            char conflict = 1;
            for(d=0; d<Abase->ndims; d++)
            {
                if(Bsvb->start[d] >=
                   Asvb->start[d] + Asvb->nsteps[d]
                   ||
                   Asvb->start[d] >=
                   Bsvb->start[d] + Bsvb->nsteps[d])
                {
                    conflict = 0;
                    break;//No conflict at the svb level.
                }
            }
            if(conflict)
                return 1;
        }
    return 0;
}/* dndnode_conflict */

/*===================================================================
 *
 * Removes a operation from the sub-view-block DAG.
 * op2apply is the operations that should be applyed.
 * If op2apply is NULL then op2apply is ignored.
 * Returns number of operations in op2apply.
 * Private.
 */
static npy_intp dag_svg_remove(dndop *op, dndop *op2apply[])
{
    int j;
    npy_intp b, i, nops=1, mnops=DNPY_MAX_OP_MERGES;

    #ifdef DNDY_TIME
        unsigned long long tdelta;
        DNDTIME(tdelta);
    #endif

    if(op2apply != NULL)
        op2apply[0] = op;
    else
        mnops = 1;

    for(i=0; i<nops; i++)
    {
        if(op2apply != NULL)
            op = op2apply[i];

        for(j=0; j<op->narys; j++)
        {
            if(op->svbs[j] == NULL)//Whole array.
            {
                dndarray *ary = op->views[j]->base;
                assert(ary != NULL);
                for(b=0; b<ary->nblocks; b++)
                {
                    assert(ary->rootnodes[b] != NULL);
                    while(ary->rootnodes[b] != NULL &&
                          ary->rootnodes[b]->op == op)
                        ary->rootnodes[b] = ary->rootnodes[b]->next;

                    dndnode *n1 = ary->rootnodes[b];

                    //We are finished if the list has become empty.
                    if(n1 == NULL)
                        continue;

                    //Handle the first node in the list.
                    if(--n1->op->refcount == 0)
                    {
                        if(nops < mnops && n1->op->optype == DNPY_NONCOMM)
                            op2apply[nops++] = n1->op;
                        else
                            ready_queue[ready_queue_size++] = n1->op;
                    }
                    //Handle the rest of the nodes in the list.
                    dndnode *n2 = n1->next;
                    while(n2 != NULL)
                    {
                        assert(n1->next == n2);
                        if(n2->op == op)//Remove the node.
                            n1->next = n2->next;
                        else
                        {
                            if(--n2->op->refcount == 0)
                            {
                                if(nops < mnops && n2->op->optype == DNPY_NONCOMM)
                                    op2apply[nops++] = n2->op;
                                else
                                    ready_queue[ready_queue_size++] = n2->op;
                            }
                            n1 = n2;
                        }
                        n2 = n2->next;
                    }
                }
            }
            else
            {
                dndsvb *svb = op->svbs[j];
                while((*svb->rootnode) != NULL &&
                      (*svb->rootnode)->op == op)
                    *svb->rootnode = (*svb->rootnode)->next;
                dndnode *n1 = *svb->rootnode;

                //We are finished if the list has become empty.
                if(n1 == NULL)
                    continue;

                //Handle the first node in the list.
                if(dndnode_conflict(op, j, n1->op, n1->op_ary_idx))
                    if(--n1->op->refcount == 0)
                    {
                        if(nops < mnops && n1->op->optype == DNPY_NONCOMM)
                            op2apply[nops++] = n1->op;
                        else
                            ready_queue[ready_queue_size++] = n1->op;
                    }

                //Handle the rest of the nodes in the list.
                dndnode *n2 = n1->next;
                while(n2 != NULL)
                {
                    assert(n1->next == n2);
                    if(n2->op == op)//Remove the node.
                        n1->next = n2->next;
                    else
                    {
                        if(dndnode_conflict(op, j, n2->op, n2->op_ary_idx))
                            if(--n2->op->refcount == 0)
                            {
                                if(nops < mnops && n2->op->optype == DNPY_NONCOMM)
                                    op2apply[nops++] = n2->op;
                                else
                                    ready_queue[ready_queue_size++] = n2->op;
                            }
                        n1 = n2;
                    }
                    n2 = n2->next;//Go to next node.
                }
            }
        }
    }

    #ifdef DNDY_TIME
        DNDTIME_SUM(tdelta, dndt.dag_svb_rm)
    #endif

    return nops;
}/* dag_svg_remove */


/*===================================================================
 *
 * Flush the sub-view block DAG.
 * Frees the work buffer when 'free_workbuf' is true.
 * Private.
 */
static void dag_svb_flush(int free_workbuf)
{
    npy_intp i, j, f, commsize, ncommsize;
    int fcomm[DNPY_RDY_QUEUE_MAXSIZE];
    int fcommsize;
    MPI_Request reqs[DNPY_RDY_QUEUE_MAXSIZE];
    MPI_Status reqstatus[DNPY_RDY_QUEUE_MAXSIZE];
    dndop *comm[DNPY_RDY_QUEUE_MAXSIZE];
    dndop *ncomm[DNPY_RDY_QUEUE_MAXSIZE];
    MPI_Datatype dtype[DNPY_RDY_QUEUE_MAXSIZE];
    npy_intp dtypesize=0;

    #ifdef DNDY_TIME
        ++dndt.nflush;
        unsigned long long tdelta;
        DNDTIME(tdelta);
    #endif
    #ifdef DNPY_STATISTICS
        dag_svb_dump();
    #endif

    commsize=0; ncommsize=0;
    while(ready_queue_size + commsize + ncommsize > 0)
    {
        #ifdef DNDY_TIME
            unsigned long long tdelta2;
            DNDTIME(tdelta2);
        #endif

        assert(ready_queue_size <= DNPY_RDY_QUEUE_MAXSIZE);
        //Sort the queue into two queues - one for communication and
        //one for non-communication nodes.
        //Furthermore, initiate the communication nodes.
        for(i=0; i<ready_queue_size; i++)
        {
            assert(ready_queue[i]->refcount == 0);
            //Init. all communication nodes in the ready queue.
            if(ready_queue[i]->optype == DNPY_COMM)
            {
                dndop_comm *C = (dndop_comm*) ready_queue[i];
                MPI_Datatype comm_dtype;
                assert(C->refcount == 0);
                switch(C->op)
                {
                    case DNPY_RECV:
                        assert(C->narys == 1);
                        comm_dtype = calc_svb_MPIdatatype(C->views[0],
                                                          C->svbs[0]);
                        delayed_array_allocation(C->views[0]->base);
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
                        WORKBUF_INC(C->svbs[0]->nelem *
                                    C->views[0]->base->elsize);
                        assert(C->svbs[0]->data != NULL);
                        MPI_Irecv(C->svbs[0]->data, C->svbs[0]->nelem,
                                  C->views[0]->base->mpi_dtype,
                                  C->remote_rank, C->mpi_tag,
                                  MPI_COMM_WORLD, &reqs[commsize]);
                        break;
                    case DNPY_SEND:
                        assert(C->narys == 1);
                        comm_dtype = calc_svb_MPIdatatype(C->views[0],
                                                          C->svbs[0]);
                        delayed_array_allocation(C->views[0]->base);
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
                    case DNPY_REDUCE_SEND:
                        assert(C->narys == 1);
                        comm_dtype = calc_svb_MPIdatatype(C->views[0],
                                                          C->svbs[0]);
                        assert(C->svbs[0]->data != NULL);
                        MPI_Isend(C->svbs[0]->data +
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
                    default:
                        fprintf(stderr, "Unknown DAG operation: %s.\n",
                                optype2str(C->op));
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

        #ifdef DNDY_TIME
            DNDTIME_SUM(tdelta2, dndt.comm_init)
        #endif

        //Apply one non-communication node and move new non-depend
        //nodes to the ready queue.
        //Instead of moving new non-depended non-communication nodes
        //to the ready queue they are directly applied.
        if(ncommsize > 0)
        {
            dndop *op = ncomm[--ncommsize];//Using a FILO order.
            dndop **op2apply = workbuf_nextfree;
            npy_intp nops = dag_svg_remove(op, op2apply);
            npy_intp cur_op;
            WORKBUF_INC(nops);//Reserve memory.

            for(cur_op=0; cur_op < nops; cur_op++)
            {
                op = op2apply[cur_op];
                switch(op->op)
                {
                    case DNPY_UFUNC:
                        apply_ufunc((dndop_ufunc*)op);
                        Py_DECREF(((dndop_ufunc*)op)->PyOp);
                        break;
                    case DNPY_DESTROY_ARRAY:
                        assert(op->narys == 1);
                        assert(op->views[0] != NULL);
                        rm_dndview(op->views[0]->uid);
                        break;
                    default:
                        fprintf(stderr, "Unknown DAG operation: %s.\n",
                                optype2str(op->op));
                        MPI_Abort(MPI_COMM_WORLD, -1);
                }
            }
            workbuf_nextfree -= nops;//Unreserve memory.

            ++dndt.napply;
            dndt.nconnect += commsize;
            dndt.nconnect_max = MAX(dndt.nconnect_max, commsize);
        }
        //Test for ready communication and possibly move new non-depend
        //nodes to the ready queue. Furthermore, if there is nothing
        //else to do (no operations that are ready) we wait.
        if(commsize > 0)
        {
            #ifdef DNDY_TIME
                unsigned long long tdelta2;
                DNDTIME(tdelta2);
            #endif

            if(ncommsize > 0)
            {
                MPI_Testsome(commsize, reqs, &fcommsize, fcomm,
                             reqstatus);
                assert(fcommsize != MPI_UNDEFINED);
            }
            else if(ready_queue_size == 0)
            {
                MPI_Waitsome(commsize, reqs, &fcommsize, fcomm,
                             reqstatus);
                assert(fcommsize != MPI_UNDEFINED);
            }
            else
                fcommsize = 0;

            #ifdef DNDY_TIME
                DNDTIME_SUM(tdelta2, dndt.ufunc_comm)
            #endif

            for(f=0; f<fcommsize; f++)
            {
                dndop *op = comm[fcomm[f]];
                dag_svg_remove(op, NULL);
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
    }
    //Do the delayed MPI type freeing.
    for(i=0; i<dtypesize; i++)
        MPI_Type_free(&dtype[i]);

    assert(ready_queue_size == 0);
    #ifdef DNDY_TIME
        DNDTIME_SUM(tdelta, dndt.dag_svb_flush)
    #endif
    if(free_workbuf)
    {
        workbuf_nextfree = workbuf;
    }
}/* dag_svb_flush */


/*===================================================================
 *
 * Add the list of node to the sub-view block DAG.
 * Node that all nodes in the list must relate to the same operation.
 * Private.
 */
static void dag_svb_add(dndnode *nodes, int nnodes, int force_laziness)
{
    npy_intp i;
    int n, idx;

    #ifdef DNDY_TIME
        unsigned long long tdelta;
        DNDTIME(tdelta);
    #endif

    //Init values for statustics.
    #ifdef DNPY_STATISTICS
        for(n=0; n<nnodes; n++)
            nodes[n].uid = ++node_uid_count;
        nodes[0].op->uid = ++op_uid_count;
    #endif
    //Handle one node at a time.
    for(n=0; n<nnodes; n++)
    {
        dndnode *node = &nodes[n];
        assert(node->op != NULL);
        assert(node->op_ary_idx >= 0);
        assert(node->op_ary_idx < node->op->narys);
        assert(nodes[0].op == node->op);
        node->next = NULL;
        idx = node->op_ary_idx;
        //We append the new node to the linked list and for each
        //conflict we increase the refcount of the new node.
        if(node->op->svbs[idx] == NULL)//Whole array.
        {
            dndarray *ary = node->op->views[idx]->base;
            for(i=0; i<ary->nblocks; i++)
            {
                if(ary->rootnodes[i] == NULL)
                    ary->rootnodes[i] = node;
                else
                {
                    dndnode *tnode = ary->rootnodes[i];
                    while(1)
                    {
                        if(tnode->op != node->op)
                            node->op->refcount++;
                        if(tnode->next == NULL)//We are finished.
                            break;
                        tnode = tnode->next;//Go to next node.
                    }
                    tnode->next = node;
                    assert(tnode->next->next == NULL);
                }
                //Need to clone the new node to get it "spanning" over
                //the whole array.
                memcpy(workbuf_nextfree, node, sizeof(dndnode));
                node = workbuf_nextfree;
                WORKBUF_INC(sizeof(dndnode));

                #ifdef DNPY_STATISTICS
                    node->uid = ++node_uid_count;
                #endif
            }
        }
        else
        {
            assert(node->op->svbs[idx]->rootnode != NULL);
            dndnode *tnode = *node->op->svbs[idx]->rootnode;
            if(tnode == NULL)
                *node->op->svbs[idx]->rootnode = node;
            else
            {
                while(1)
                {
                    if(tnode->op != node->op &&
                       dndnode_conflict(node->op, idx, tnode->op,
                                        tnode->op_ary_idx))
                        node->op->refcount++;

                    if(tnode->next == NULL)//We are finished.
                        break;
                    tnode = tnode->next;//Go to next node.
                }
                tnode->next = node;
            }
        }
    }

    //Place the operation in the ready queue when no dependency was
    //found.
    if(nodes[0].op->refcount == 0)
    {
        assert(ready_queue_size+1 <= DNPY_RDY_QUEUE_MAXSIZE);
        ready_queue[ready_queue_size++] = nodes[0].op;
    }

    assert(ready_queue_size > 0);

    #ifdef DNDY_TIME
        DNDTIME_SUM(tdelta, dndt.dag_svb_add)
    #endif

    #ifdef DNPY_NO_LAZY_EVAL
    if(!force_laziness)
        dag_svb_flush(0);//Note that we do not free the work buffer
    #endif
} /* dag_svb_add */


/*NUMPY_API
 *===================================================================
 * Create dndarray.
 * Public
 */
static intp
dnumpy_create_dndarray(PyArrayObject *pyary, intp onerank)
{
    int i;
    dndview *retview;

    //Make sure that the init message has been handled.
    if(initmsg_not_handled)
        dnumpy_init_data_layout(NULL,NULL);

    //Create dndarray.
    dndarray *newarray = malloc(sizeof(dndarray));

    newarray->dtype = PyArray_TYPE(pyary);
    newarray->elsize = PyArray_ITEMSIZE(pyary);
    newarray->ndims = PyArray_NDIM(pyary);
    newarray->refcount = 1;
    newarray->onerank = onerank;
    for(i=0; i<PyArray_NDIM(pyary); i++)
        newarray->dims[i] = PyArray_DIM(pyary, i);

    //Create dndview. NB: the base will have to be set when 'newarray'
    //has found its final resting place. (Done by put_dndarray).
    dndview *newview = malloc(sizeof(dndview));
    newview->uid = ++uid_count;
    newview->nslice = PyArray_NDIM(pyary);
    newview->ndims = PyArray_NDIM(pyary);
    newview->alterations = 0;
    for(i=0; i<PyArray_NDIM(pyary); i++)
    {
        //Default the view will span over the whole array.
        newview->slice[i].start = 0;
        newview->slice[i].step = 1;
        newview->slice[i].nsteps = PyArray_DIM(pyary, i);
    }

#ifndef DNPY_SPMD
    //Tell slaves about the new array
    msg[0] = DNPY_CREATE_ARRAY;
    memcpy(&msg[1], newarray, sizeof(dndarray));
    memcpy(((char *) &msg[1]) + sizeof(dndarray), newview,
           sizeof(dndview));

    *(((char *) &msg[1])+sizeof(dndarray)+sizeof(dndview)) = DNPY_MSG_END;

    msg2slaves(msg,2*sizeof(npy_intp)+sizeof(dndarray)+sizeof(dndview));
#endif

    //The master should also do the operation
    #ifdef DISTNUMPY_DEBUG
        printf("Rank 0 received msg: ");
    #endif
    retview = do_CREATE_ARRAY(newarray, newview);

#ifdef DNPY_SPMD
    //In SPMD the python array points directly to the local data
    PyArray_BYTES(pyary) = retview->base->data;
#endif

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

#ifndef DNPY_SPMD
    //Tell slaves about the destruction
    msg[0] = DNPY_DESTROY_ARRAY;
    msg[1] = ary->uid;
    msg[2] = DNPY_MSG_END;
    msg2slaves(msg,3 * sizeof(npy_intp));
#endif

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
    newview.nblocks = 1;

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
            memcpy(&newview.slice[ni], &slice[si], sizeof(dndslice));
            ni++; si++;
            newview.ndims++;
            newview.alterations |= DNPY_NDIMS;
        }
        else if(orgview->slice[oi].nsteps == PseudoIndex)
        {
            memcpy(&newview.slice[ni], &orgview->slice[oi],
                   sizeof(dndslice));

            if(slice[si].start > 0)
            {
                //This is a special case where the user indexes the
                //PseudoIndex, which is legal and will return [].
                newview.nblocks = 0;
            }

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

    //Check if the view is not block alligned
    for(si=0; si<newview.nslice; ++si)
        if(newview.slice[si].start % blocksize != 0 ||
           newview.slice[si].step != 1)
        {
            newview.alterations |= DNPY_NONALIGNED;
            break;
        }

#ifndef DNPY_SPMD
    //Tell slaves about the new view.
    //NB: It is up to the slaves to add the newview.base adresse.
    msg[0] = DNPY_CREATE_VIEW;
    msg[1] = orgview->uid;
    memcpy(&msg[2], &newview, sizeof(dndview));
    *(((char *) &msg[2])+sizeof(dndview)) = DNPY_MSG_END;

    msg2slaves(msg, 3 * sizeof(npy_intp) + sizeof(dndview));
#endif

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
 * If the optional option barrier is true, a MPI barrier is included
 * in SPMD mode.
 */
static PyObject *
dnumpy_evalflush(PyObject *NPY_UNUSED(ignored), PyObject *args,
                 PyObject *kws)
{
    Bool barrier = FALSE;
    static char *kwd[]= {"barrier", NULL};

    if(!PyArg_ParseTupleAndKeywords(args, kws, "|O&", kwd,
                                    PyArray_BoolConverter, &barrier))
        return NULL;

#ifndef DNPY_SPMD
    //Tell slaves
    msg[0] = DNPY_EVALFLUSH;
    msg[1] = DNPY_MSG_END;
    msg2slaves(msg, 2 * sizeof(npy_intp));
#endif
    //The master should also do the operation.
    #ifdef DISTNUMPY_DEBUG
        printf("Rank 0 received msg: EVALFLUSH\n");
    #endif
    dag_svb_flush(1);

#ifdef DNPY_SPMD
    if(barrier)
        MPI_Barrier(MPI_COMM_WORLD);
#endif

    Py_RETURN_NONE;
} /* dnumpy_evalflush */

/*NUMPY_API
 *===================================================================
 * Public
 * Reset the timing.
 */
static PyObject *
dnumpy_time_reset(void)
{
#ifndef DNPY_SPMD
    //Tell slaves
    msg[0] = DNPY_TIME_RESET;
    msg[1] = DNPY_MSG_END;
    msg2slaves(msg, 2 * sizeof(npy_intp));
#endif
    //The master should also do the operation.
    do_TIME_RESET();
    Py_RETURN_NONE;
} /* dnumpy_reset_time */


/*NUMPY_API
 *===================================================================
 * Public
 * Returns timing in a Python Dictionary
 */
static PyObject *
dnumpy_time_getdict(void)
{
#ifndef DNPY_SPMD
    //Tell slaves
    msg[0] = DNPY_TIME_GETDICT;
    msg[1] = DNPY_MSG_END;
    msg2slaves(msg, 2 * sizeof(npy_intp));
#endif
    //The master should also do the operation.
    return do_TIME_GETDICT();
} /* dnumpy_time_getdict */



/*NUMPY_API
 *===================================================================
 * Public
 * Returning the rank of the calling MPI-process
 */
static PyObject *
dnumpy_myrank(void)
{
    return PyLong_FromLong((long) myrank);
} /* dnumpy_myrank */

/*NUMPY_API
 *===================================================================
 * Public
 * Returning the number of MPI-processes in MPI_COMM_WORLD.
 */
static PyObject *
dnumpy_worldsize(void)
{
    return PyLong_FromLong((long) worldsize);
} /* dnumpy_worldsize */

/*NUMPY_API
 *===================================================================
 * Public
 * Returning the rank of the calling MPI-process
 */
static PyObject *
dnumpy_blocksize(void)
{
    return PyInt_FromLong((long) blocksize);
} /* dnumpy_blocksize */


/*NUMPY_API
 *===================================================================
 * Public
 * Returns True if DistNumPy is compiled in SPMD mode.
 */
static PyObject *
dnumpy_SPMD_mode(void)
{
    #ifdef DNPY_SPMD
        Py_RETURN_TRUE;
    #else
        Py_RETURN_FALSE;
    #endif
} /* dnumpy_SPMD_mode */


/*NUMPY_API
 *===================================================================
 * Public
 * Returns True when the two array-view overlap.
 */
static char
dnumpy_data_overlap(intp Auid, intp Buid)
{
    dndview *A = get_dndview(Auid);
    dndview *B = get_dndview(Buid);
    //At the moment we only check at the array-base level.
    if(Auid == Buid)
        return 0;
    if(A->base->uid == B->base->uid)
        return 1;
    else
        return 0;
} /* dnumpy_data_overlap */

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

#ifndef DNPY_SPMD
    int ndims = view->ndims;
    int elsize = view->base->elsize;
    //Tell slaves about the new item.
    msg[0] = DNPY_PUT_ITEM;
    msg[1] = view->uid;
    memcpy(&msg[2], PyArray_DATA(citem2), elsize);
    memcpy(((char *) &msg[2]) + elsize, coordinate,
           sizeof(npy_intp) * ndims);
    *(((char *) &msg[2]) + elsize + sizeof(npy_intp) * ndims) = DNPY_MSG_END;

    msg2slaves(msg, 3 * sizeof(npy_intp) + elsize +
                    ndims * sizeof(npy_intp));
#endif

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
                        intp coordinate[NPY_MAXDIMS])
{
    //Get arrray structs.
    dndview *view = get_dndview(uid);

#ifndef DNPY_SPMD
    //Tell slaves to send item.
    msg[0] = DNPY_GET_ITEM;
    msg[1] = view->uid;
    memcpy(&msg[2], coordinate, sizeof(npy_intp)*view->ndims);
    *(((char *) &msg[2]) + sizeof(npy_intp)*view->ndims) = DNPY_MSG_END;

    msg2slaves(msg, 3*sizeof(npy_intp) + view->ndims*sizeof(npy_intp));
#endif

    //The master will retrive the whole array.
    #ifdef DISTNUMPY_DEBUG
        printf("Rank 0 received msg: ");
    #endif

    do_PUTGET_ITEM(0, view, retdata, coordinate);
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
    npy_intp type_num = 0; //Scalar datatype.
    npy_intp elsize = 0; //Scalar datasize, zero if no scalar is in arylist.
    char *scalar = NULL;
    npy_intp aryuids[NPY_MAXARGS];

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
            type_num = arylist[i]->descr->type_num;
            elsize = arylist[i]->descr->elsize;
            scalar = arylist[i]->data;
            break;
        }

    for(i=0; i<narys;i++)
        aryuids[i] = arylist[i]->dnduid;

#ifndef DNPY_SPMD
    msg[0] = DNPY_UFUNC;
    msg[1] = narys;
    msg[2] = nout_arys;
    msg[3] = appropriate_function;
    msg[4] = type_num; //Scalar datatype.
    msg[5] = elsize; //Scalar datasize, zero if no scalar is in arylist.
    if(elsize > 0)
        memcpy(msg+6, scalar, msg[5]);

    //Copy arylist's uid to msg.
    for(i=0; i<narys;i++)
        memcpy(((char*)(msg+6))+msg[5]+i*sizeof(npy_intp),
                &arylist[i]->dnduid, sizeof(npy_intp));

    //Copy the string op.
    strcpy(((char*)(msg+6+msg[1]))+msg[5], op);

    *(((char*)(msg+6+msg[1]))+msg[5]+strlen(op)+1) = DNPY_MSG_END;

    msg2slaves(msg, (7+msg[1])*sizeof(npy_intp)+msg[5]+strlen(op)+1);
#endif

    //The master should also do the operation.
    #ifdef DISTNUMPY_DEBUG
        printf("Rank 0 received msg: ");
    #endif

    do_UFUNC(aryuids, narys, nout_arys, appropriate_function, type_num,
             elsize, scalar, op);
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

    if(in->base->onerank >= 0)
    {
        PyErr_SetString(PyExc_RuntimeError,
                        "One rank array is not allowed.");
        return -1;
    }

    if(out_ary->dnduid > 0)
    {
        dndview *out = get_dndview(out_ary->dnduid);
        //Check if the views covers the whole array.
        if(in->alterations != 0 || out->alterations != 0)
        {
            PyErr_SetString(PyExc_RuntimeError,
                            "reduce on distributed arrays must "
                            "be whole arrays and not a partial "
                            "view of a underlying array.");
            return -1;
        }
    }

#ifndef DNPY_SPMD
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
#endif

    //The master should also do the operation
    #ifdef DISTNUMPY_DEBUG
        printf("Rank 0 received msg: ");
    #endif

    do_UFUNC_REDUCE(in_ary->dnduid, out_ary->dnduid,
                    PyArray_TYPE(out_ary), PyArray_ITEMSIZE(out_ary),
                    PyArray_BYTES(out_ary), axis, op);
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

#ifndef DNPY_SPMD
    //Tell slaves
    msg[0] = DNPY_ZEROFILL;
    msg[1] = view->uid;
    msg[2] = DNPY_MSG_END;
    msg2slaves(msg, 3 * sizeof(npy_intp));
#endif

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

#ifndef DNPY_SPMD
    //Get filename length
    int filename_length = strlen(filename);

    //Tell slaves
    msg[0] = DNPY_DATAFILL;
    msg[1] = view->uid;
    memcpy(&msg[2], &datapos, sizeof(long));
    strcpy((char*)&msg[2]+sizeof(long), filename);
    *(&msg[2]+sizeof(long)+(filename_length+1)*sizeof(char)) = DNPY_MSG_END;
    msg2slaves(msg, 3*sizeof(npy_intp) + sizeof(long) + (filename_length+1)*sizeof(char));
#endif

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

#ifndef DNPY_SPMD
    //Get filename length
    int filename_length = strlen(filename);

    //Tell slaves
    msg[0] = DNPY_DATADUMP;
    msg[1] = view->uid;
    memcpy(&msg[2], &datapos, sizeof(long));
    strcpy((char*)&msg[2]+sizeof(long), filename);
    *(&msg[2]+sizeof(long)+(filename_length+1)*sizeof(char)) = DNPY_MSG_END;
    msg2slaves(msg, 3*sizeof(npy_intp) + sizeof(long) + (filename_length+1)*sizeof(char));
#endif

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
    count = ((stop-start) / step) + (((stop-start) % step) != 0);

    //Create the returning distributed vector.
    ret = PyArray_New(&PyArray_Type, 1, &count, view->base->dtype,
                      NULL, NULL, 0,
                      NPY_BEHAVED | NPY_CARRAY | DNPY_DISTRIBUTED,
                      NULL);

#ifndef DNPY_SPMD
    //Tell slaves
    msg[0] = DNPY_DIAGONAL;
    msg[1] = view->uid;
    msg[2] = PyArray_DNDUID(ret);
    msg[3] = offset;
    msg[4] = axis1;
    msg[5] = axis2;
    msg[6] = DNPY_MSG_END;
    msg2slaves(msg, 7 * sizeof(npy_intp));
#endif

    //The master should also do the operation.
    #ifdef DISTNUMPY_DEBUG
        printf("Rank 0 received msg: ");
    #endif
    do_DIAGONAL(view, get_dndview(PyArray_DNDUID(ret)), offset, axis1,
                axis2);

    return ret;
} /* dnumpy_diagonal */


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

#ifndef DNPY_SPMD
    //Tell slaves
    msg[0] = DNPY_MATMUL;
    msg[1] = Auid;
    msg[2] = Buid;
    msg[3] = Cuid;
    msg[4] = DNPY_MSG_END;
    msg2slaves(msg, 5 * sizeof(npy_intp));
#endif

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
dnumpy_exit(void)
{
    int i;
#ifndef DNPY_SPMD
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
#endif
    //Make sure that the sub-view-block DAG is flushed.
    dag_svb_flush(1);

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

    //Free the memory pool.
    free_mem_pool(mem_pool);

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
    //Initiate timers to zero.
    memset(&dndt, 0, sizeof(dndtime));
    DNDTIME(totaldelta)

#ifdef DNPY_SPMD
    return Py_True;
#else

    if(myrank == 0)
        return Py_True;

    int shutdown = 0;
    while(shutdown == 0)//Work loop
    {
        char *t1, *t2, *t3;
        npy_intp d1, d2, d3, d4, d5;
        long l1;
        dndview *ary, *ary2, *ary3;
        //Receive message from master.
        MPI_Bcast(msg, DNPY_MAX_MSG_SIZE, MPI_BYTE, 0, MPI_COMM_WORLD);
        char *msg_data = (char *) &msg[1];
        #ifdef DISTNUMPY_DEBUG
            printf("Rank %d received msg: ", myrank);
        #endif
        switch(msg[0])
        {
            case DNPY_INIT_PGRID:
                do_INIT_PGRID((int*)msg_data);
                break;
            case DNPY_INIT_BLOCKSIZE:
                blocksize = *((npy_intp*)msg_data);
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
                dag_svb_flush(1);
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
            case DNPY_TIME_RESET:
                do_TIME_RESET();
                break;
            case DNPY_TIME_GETDICT:
                do_TIME_GETDICT();
                break;
            default:
                fprintf(stderr, "Unknown msg: %ld\n", (long)msg[0]);
                MPI_Abort(MPI_COMM_WORLD, -1);
        }
    }
    return Py_False;
#endif
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
dnumpy_get_reged_ufunc_module(void)
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


/*NUMPY_API
 *===================================================================
 * Public
 * Returns the process grid the array is distributed on.
 */
static PyObject *
dnumpy_proc_grid(PyArrayObject *self, PyObject *args)
{
    int i;
    if (!PyArg_ParseTuple(args, ""))
        return NULL;

    if(!PyArray_ISDISTRIBUTED(self))
    {

        PyErr_SetString(PyExc_RuntimeError,
                        "The array is not distributed.");
        return NULL;
    }

    dndview *ary = get_dndview(PyArray_DNDUID(self));
    int *cdims = cart_dim_sizes[ary->base->ndims-1];
    PyObject *tuple = PyTuple_New(ary->base->ndims);
    for(i=0; i<ary->base->ndims; ++i)
        PyTuple_SetItem(tuple, i, PyInt_FromLong((long) cdims[i]));

    return tuple;

} /* dnumpy_proc_grid */

/*NUMPY_API
 *===================================================================
 * Public
 * Returns the process coordinate in the process grid the array is
 * distributed on.
 */
static PyObject *
dnumpy_proc_grid_mycoord(PyArrayObject *self, PyObject *args)
{
    int i;
    if (!PyArg_ParseTuple(args, ""))
        return NULL;

    if(!PyArray_ISDISTRIBUTED(self))
    {

        PyErr_SetString(PyExc_RuntimeError,
                        "The array is not distributed.");
        return NULL;
    }
    dndview *ary = get_dndview(PyArray_DNDUID(self));
    int coords[NPY_MAXDIMS];
    rank2cart(ary->base->ndims, myrank, coords);
    PyObject *tuple = PyTuple_New(ary->base->ndims);
    for(i=0; i<ary->base->ndims; ++i)
        PyTuple_SetItem(tuple, i, PyInt_FromLong((long) coords[i]));

    return tuple;

} /* dnumpy_proc_grid_mycoord */

/*NUMPY_API
 * ===================================================================
 * Public
 * Returns a Python Array Object that points to the local part of the
 * distributed array.
 */
static PyObject *
dnumpy_local_data(PyArrayObject *self, PyObject *args)
{
    if(!PyArg_ParseTuple(args, ""))
        return NULL;

    if(!PyArray_ISDISTRIBUTED(self))
    {

        PyErr_SetString(PyExc_RuntimeError,
                        "The array is already a local numpy array.");
        return NULL;
    }

    dndview *view = get_dndview(PyArray_DNDUID(self));

    dag_svb_flush(1);
    delayed_array_allocation(view->base);

    if(view->alterations == 0)//No views.
        return PyArray_SimpleNewFromData(view->ndims,
                                         view->base->localdims,
                                         view->base->dtype,
                                         view->base->data);
    if(view->base->onerank >= 0)
    {
        PyErr_SetString(PyExc_RuntimeError,
                  "One rank array is not allowed with local of views.");
        return NULL;
    }

    npy_intp s,i,b, offset[NPY_MAXDIMS], start, strides[NPY_MAXDIMS];
    npy_intp dims[NPY_MAXDIMS];
    int *cdims = cart_dim_sizes[view->base->ndims-1];

    int pcoord[NPY_MAXDIMS];
    rank2cart(view->base->ndims, myrank, pcoord);

    i=0;b=0;
    for(s=0; s < view->nslice; ++s)
    {
        if(view->slice[s].nsteps == PseudoIndex)
        {
            dims[i] = 1;
            offset[b] = 0;
        }
        if(view->slice[s].nsteps == SingleIndex)
        {
                PyErr_SetString(PyExc_RuntimeError, "local of array-"
                            "views must have all dimension visible");
                return NULL;
        }
        else
        {
            if(view->slice[s].step != 1)
            {
                PyErr_SetString(PyExc_RuntimeError, "local of array-"
                                "views must have a step size of one");
                return NULL;
            }
            if(view->slice[s].start % blocksize != 0)
            {
                PyErr_SetString(PyExc_RuntimeError, "local of array-"
                                "views must have a start offset that "
                                "are divisible with np.BLOCKSIZE");
                return NULL;
            }
            if(view->slice[s].nsteps % blocksize != 0)
            {
                PyErr_SetString(PyExc_RuntimeError, "local of array-"
                                "views must have a length that are "
                                "divisible with np.BLOCKSIZE");
                return NULL;
            }
            //Block offset
            npy_intp boffset = view->slice[s].start / blocksize;
            //Process offset
            npy_intp poffset = boffset % cdims[b];

            offset[b] = boffset / cdims[b];
            if(poffset > pcoord[b])
            {
                offset[b] += 1;
            }
            dims[i] = dnumroc(view->slice[s].nsteps, blocksize,
                              pcoord[b], cdims[b], poffset);
        }
        ++i;
        ++b;
    }

    //Compute the stride and the data start.
    s = view->base->elsize;
    start=0;
    for(i=view->base->ndims-1; i >= 0; --i)
    {
        start += offset[i] * s;
        strides[i] = s;
        s *= view->base->localdims[i];
    }
    start *= blocksize;

    return PyArray_New(&PyArray_Type,
                       view->ndims,
                       dims,
                       view->base->dtype,
                       strides,
                       view->base->data + start,
                       view->base->elsize,
                       NPY_BEHAVED, NULL);

}/* local_data*/


/*NUMPY_API
 *===================================================================
 * Public
 * Returns True if the Process is in the process coordinate pattern.
 * incoord is a list of coordinate. None is a wildcard e.g. [1,None]
 * means all coordinates in the second row.
 * If 'blocks' is true the coordinate is in
 */
static PyObject *
dnumpy_proc_grid_incoord(PyArrayObject *self, PyObject *args)
{
    int i;
    PyObject *incoord = NULL;
    if (!PyArg_ParseTuple(args, "O", &incoord))
        return NULL;

    /*
    int blocks = 0;
    static char *kwd[] = {"blocks", NULL};
    if(!PyArg_ParseTupleAndKeywords(args, kwds, "O|O&", kwd, &incoord,
                                    PyArray_BoolConverter, blocks))
        return NULL;
    */

    #ifndef DNPY_SPMD
        PyErr_Format(PyExc_ValueError,
                     "this makes only sense in SPMD mode.");
        return NULL;
    #endif

    if(!PyArray_ISDISTRIBUTED(self))
    {
        PyErr_Format(PyExc_ValueError, "The array is not distributed.");
        return NULL;
    }

    dndview *view = get_dndview(PyArray_DNDUID(self));
    int pcoords[NPY_MAXDIMS];
    rank2cart(view->base->ndims, myrank, pcoords);

    if(PySequence_Size(incoord) != view->base->ndims)
    {
        PyErr_Format(PyExc_ValueError, "The incoord argument must have "
                     "an element for each dimension in the base array");
        return NULL;
    }

    for(i=0; i<view->base->ndims; ++i)
    {
        PyObject *item = PySequence_ITEM(incoord, i);
        if(PyInt_Check(item))
            if(PyInt_AsLong(item) != pcoords[i])
                Py_RETURN_FALSE;
    }

    Py_RETURN_TRUE;
} /* dnumpy_proc_grid_incoord */


/*===================================================================
 * Private
 * This is the __iter__() method for the block iterator object that
 * returns it self.
 */
static PyObject *blocks_iter(PyObject *self)
{
    Py_INCREF(self);
    return self;
}/* dnumpy_block_iter */


/*===================================================================
 * Private
 * This is the next() method for the block iterator object that
 * returns the next local block.
 */
static PyObject *blocks_iternext(PyObject *self)
{
    int i,j;
    dndblock_iter *iter = (dndblock_iter *)self;
    npy_intp offset;
    PyObject *ret;

    if(iter->curblock[0] == -1)//Finish flag.
    {
        PyErr_SetNone(PyExc_StopIteration);
        return NULL;
    }

    //Compute the array offset
    offset = 0;
    for(i=iter->view->base->ndims-1; i >= 0; --i)
        offset += iter->curblock[i] * iter->strides[i];
    offset *= blocksize;

    ret = PyArray_New(&PyArray_Type,
                      iter->view->ndims,
                      iter->dims,
                      iter->view->base->dtype,
                      iter->strides,
                      iter->view->base->data + offset,
                      iter->view->base->elsize,
                      NPY_BEHAVED, NULL);

    //Iterate current coord one block.
    for(j=iter->view->ndims-1; j >= 0; j--)
    {
        iter->curblock[j] += 1;
        if(iter->curblock[j] >= iter->slice[j].nsteps)
        {
            //We are finished, if wrapping around.
            if(j == 0)
            {
                iter->curblock[0] = -1;//Flag saying we are finished.
                break;
            }
            iter->curblock[j] = iter->slice[j].start;//Start coord.
        }
        else
            break;
    }
    return ret;
}/* dnumpy_block_iternext */


//Python type definition of the the block iterator
static PyTypeObject dndblock_iterType = {
    PyObject_HEAD_INIT(NULL)
    0,                         /*ob_size*/
    "numpy.blockIter",         /*tp_name*/
    sizeof(dndblock_iter),     /*tp_basicsize*/
    0,                         /*tp_itemsize*/
    0,                         /*tp_dealloc*/
    0,                         /*tp_print*/
    0,                         /*tp_getattr*/
    0,                         /*tp_setattr*/
    0,                         /*tp_compare*/
    0,                         /*tp_repr*/
    0,                         /*tp_as_number*/
    0,                         /*tp_as_sequence*/
    0,                         /*tp_as_mapping*/
    0,                         /*tp_hash */
    0,                         /*tp_call*/
    0,                         /*tp_str*/
    0,                         /*tp_getattro*/
    0,                         /*tp_setattro*/
    0,                         /*tp_as_buffer*/
    Py_TPFLAGS_DEFAULT | Py_TPFLAGS_HAVE_ITER,
      /* tp_flags: Py_TPFLAGS_HAVE_ITER tells python to
         use tp_iter and tp_iternext fields. */
    "Local block iterator object.",           /* tp_doc */
    0,  /* tp_traverse */
    0,  /* tp_clear */
    0,  /* tp_richcompare */
    0,  /* tp_weaklistoffset */
    blocks_iter,     /* tp_iter: __iter__() method */
    blocks_iternext  /* tp_iternext: next() method */
};


/*NUMPY_API
 *===================================================================
 * Public
 * Returns the local block iterator.
 */
static PyObject *
dnumpy_blocks(PyArrayObject *self, PyObject *args)
{
    int i,s;
    int pcoord[NPY_MAXDIMS];
    PyObject *slices = NULL;
    if (!PyArg_ParseTuple(args, "O", &slices))
        return NULL;

    if(!PyArray_ISDISTRIBUTED(self))
    {
        PyErr_Format(PyExc_ValueError, "The array is not distributed.");
        return NULL;
    }
    dndview *view = get_dndview(PyArray_DNDUID(self));
    if(view->alterations != 0)
    {
        PyErr_Format(PyExc_ValueError, "Views are not suported.");
        return NULL;
    }
    if(PySequence_Size(slices) != view->ndims)
    {
        PyErr_Format(PyExc_ValueError, "The argument must have "
                     "an block slice for each dimension in the array");
        return NULL;
    }
    int *cdims = cart_dim_sizes[view->ndims-1];

    dag_svb_flush(1);
    delayed_array_allocation(view->base);

    //We don't need python callable __init__() method for this iterator,
    //so lets simply allocate it as PyObject and initialize it by hand.
    dndblock_iter *iter = PyObject_New(dndblock_iter, &dndblock_iterType);
    if (!iter) return NULL;

    //Set the view
    iter->view = view;

    //Set the slice based on the global input
    for(i=0; i<view->ndims; ++i)
    {
        PyObject *s = PySequence_GetItem(slices, i);
        int len = PySequence_Size(s);
        if(len == 2)
        {
            PyObject *t = PySequence_GetItem(s,0);
            iter->slice[i].start  = PyInt_AsLong(t); Py_DECREF(t);
            t = PySequence_GetItem(s,1);
            iter->slice[i].nsteps = PyInt_AsLong(t); Py_DECREF(t);

            if(iter->slice[i].nsteps > view->blockdims[i])
            {
                PyErr_Format(PyExc_ValueError, "The end value is "
                      "greater than the number of blocks in the array");
                return NULL;
            }
        }
        else
        {
            PyErr_Format(PyExc_ValueError, "The block slice format is: "
                                           "(start, end)");
            return NULL;
        }
        Py_DECREF(s);
        if(view->base->dims[i] % blocksize != 0)
        {
            PyErr_SetString(PyExc_RuntimeError, "array length must be "
                            "divisible with np.BLOCKSIZE");
                return NULL;
        }
    }

    if(view->base->onerank < 0)
    {
        //Convert global block slice to local block slice.
        rank2cart(view->base->ndims, myrank, pcoord);
        for(i=0; i<view->ndims; ++i)
        {
            npy_intp tmp = iter->slice[i].start / cdims[i];
            if(iter->slice[i].start % cdims[i] > pcoord[i])
                ++tmp;
            iter->slice[i].start = tmp;

            tmp = iter->slice[i].nsteps / cdims[i];
            if(iter->slice[i].nsteps % cdims[i] > pcoord[i])
                ++tmp;

            iter->slice[i].nsteps = tmp;
        }
    }
    else if(view->base->onerank != myrank)//We have nothing
        for(i=0; i<view->ndims; ++i)
            iter->slice[i].nsteps = 0;

    //Set current block coordinate.
    for(i=0; i < view->ndims; ++i)
    {
        iter->curblock[i] = iter->slice[i].start;
        if(iter->slice[i].nsteps <= iter->slice[i].start)
            iter->curblock[0] = -1;//Set finish flag
    }

    //Compute the stride and dimensions
    s = iter->view->base->elsize;
    for(i=iter->view->base->ndims-1; i >= 0; --i)
    {
        iter->dims[i] = blocksize;
        iter->strides[i] = s;
        s *= iter->view->base->localdims[i];
    }

    return (PyObject *)iter;

} /* dnumpy_blocks */


/*NUMPY_API
 * ===================================================================
 * Public
 * Initialization of the data layout.
 */
static PyObject *
dnumpy_init_data_layout(PyArrayObject *self, PyObject *args)
{
    PyObject *pgrid = Py_None;
    int i,j;
    int tmpsizes[NPY_MAXDIMS*NPY_MAXDIMS];

    if(args != NULL)//If not called from dnumpy_create_dndarray()
        if (!PyArg_ParseTuple(args, "O", &pgrid))
            return NULL;

    if(!initmsg_not_handled && ndndarrays > 0)
    {
        PyErr_Format(PyExc_RuntimeError, "numpy.datalayout must be "
                "called when no distributed array are allocated "
                "(%ld arrays are currently allocated).", ndndarrays);
        return NULL;
    }

    //Check for user-defined process grid.
    //The syntax used is: ndims:dim:size;
    //E.g. DNPY_PROC_SIZE="2:2:4;3:3:2" which means that array
    //with two dimensions should, at its second dimension, have
    //a size of four etc.
    memset(tmpsizes, 0, NPY_MAXDIMS*NPY_MAXDIMS*sizeof(int));
    char *env = getenv("DNPY_PROC_GRID");
    if(env != NULL)
    {
        char *res = strtok(env, ";");
        while(res != NULL)
        {
            char *size_ptr;
            int dsize = 0;
            int dim = 0;
            int ndims = strtol(res, &size_ptr, 10);
            if(size_ptr != '\0')
                dim = strtol(size_ptr+1, &size_ptr, 10);
            if(size_ptr != '\0')
                dsize = strtol(size_ptr+1, NULL, 10);
            //Make sure the input is valid.
            if(dsize <= 0 || dim <= 0 || dim > ndims ||
               ndims <= 0 || ndims > NPY_MAXDIMS)
            {
                PyErr_Format(PyExc_ValueError, "DNPY_PROC_GRID, invalid"
                             " syntax or value at \"%s\"\n", res);
                return NULL;
            }
            tmpsizes[(ndims-1)*NPY_MAXDIMS+(dim-1)] = dsize;
            //Go to next token.
            res = strtok(NULL, ";");
        }
    }
    else if(pgrid != Py_None)
    {//The environment variable supersedes the function call.
        for(i=0; i<PySequence_Size(pgrid); i++)
        {
            PyObject *tuple = PySequence_ITEM(pgrid, i);
            if(!PySequence_Check(tuple) || PySequence_Size(tuple) != 3)
            {
                PyErr_Format(PyExc_ValueError, "The datalayout "
                    "must use the following layout: [(#dimensions, "
                    "dimension, size),...] where each tuple impose a "
                    "dimension size.");
                return NULL;
            }
            PyObject *item = PySequence_ITEM(tuple, 0);
            int ndims = PyInt_AsLong(item);
            Py_DECREF(item);
            item = PySequence_ITEM(tuple, 1);
            int dim = PyInt_AsLong(item);
            Py_DECREF(item);
            item = PySequence_ITEM(tuple, 2);
            int dsize = PyInt_AsLong(item);
            Py_DECREF(item);
            Py_DECREF(tuple);

            //Make sure the input is valid.
            if(dsize <= 0 || dim <= 0 || dim > ndims ||
               ndims <= 0 || ndims > NPY_MAXDIMS)
            {
                PyErr_Format(PyExc_ValueError, "invalid values");
                return NULL;
            }
            tmpsizes[(ndims-1)*NPY_MAXDIMS+(dim-1)] = dsize;
        }
    }

    //Find a balanced distributioin of processes per direction
    //based on the restrictions specified by the user.
    for(i=0; i<NPY_MAXDIMS; i++)
    {
        int ndims = i+1;
        int t[NPY_MAXDIMS];
        int d = 0;
        //Need to reverse the order to match MPI_Dims_create
        for(j=i; j>=0; j--)
            t[d++] = tmpsizes[i*NPY_MAXDIMS+j];

        MPI_Dims_create(worldsize, ndims, t);
        d = ndims;
        for(j=0; j<ndims; j++)
            tmpsizes[i*NPY_MAXDIMS+j] = t[--d];
    }

    #ifndef DNPY_SPMD
        //Tell slaves
        msg[0] = DNPY_INIT_PGRID;
        memcpy(&msg[1], tmpsizes, NPY_MAXDIMS*NPY_MAXDIMS*sizeof(int));
        *(&msg[1]+NPY_MAXDIMS*NPY_MAXDIMS*sizeof(int)) = DNPY_MSG_END;
        msg2slaves(msg, NPY_MAXDIMS*NPY_MAXDIMS*sizeof(int));
    #endif

    do_INIT_PGRID(tmpsizes);

    Py_RETURN_NONE;
}/* dnumpy_init_data_layout */


/*NUMPY_API
 * ===================================================================
 * Public
 * Initialization of distnumpy.
 */
static void
dnumpy_init(void)
{
    int provided;
    int flag;
    int i;

    //Make sure we only initialize once.
    MPI_Initialized(&flag);
    if (flag)
    {
        fprintf(stderr, "DistNumPy Warning - multiple "
                         "initialization attempts.\n");
        return;
    }

    //We make use of MPI_Init_thread even though we only ask for
    //a MPI_THREAD_SINGLE level thread-safety because MPICH2 only
    //supports MPICH_ASYNC_PROGRESS when MPI_Init_thread is used.
    //Note that when MPICH_ASYNC_PROGRESS is defined the thread-safety
    //level will automatically be set to MPI_THREAD_MULTIPLE.
    MPI_Init_thread(NULL, NULL, MPI_THREAD_SINGLE, &provided);

    MPI_Comm_rank(MPI_COMM_WORLD, &myrank);
    MPI_Comm_size(MPI_COMM_WORLD, &worldsize);

    //Allocate buffers.
    workbuf = malloc(DNPY_WORK_BUFFER_MAXSIZE);
    workbuf_nextfree = workbuf;
    assert(workbuf != NULL);

    //We subtract one MB to avoid segmentation faults when the workbuf
    //is used before the call to WORKBUF_INC()
    workbuf_max = ((char*)workbuf) + DNPY_WORK_BUFFER_MAXSIZE - 1048576;

    //Init the block iterator
    dndblock_iterType.tp_new = PyType_GenericNew;
    if(PyType_Ready(&dndblock_iterType) < 0)  return;
    Py_INCREF(&dndblock_iterType);

    //Allocate cart_dim_sizes and cart_dim_strides.
    for(i=0; i<NPY_MAXDIMS; i++)
    {
        cart_dim_sizes[i] = malloc((i+1)*sizeof(int));
        cart_dim_strides[i] = malloc((i+1)*sizeof(int));
    }

    //Set blocksize
    if(myrank == 0)
    {
        char *env;
        //Check for user-defined block size.
        env = getenv("DNPY_BLOCKSIZE");
        if(env == NULL)
            blocksize = DNPY_BLOCKSIZE;
        else
            blocksize = atoi(env);

        if(blocksize <= 0)
        {
            fprintf(stderr, "User-defined blocksize must be greater "
                            "than zero\n");
            MPI_Abort(MPI_COMM_WORLD, -1);
        }
    }
    MPI_Bcast(&blocksize, sizeof(npy_intp), MPI_BYTE, 0, MPI_COMM_WORLD);

} /* dnumpy_init */

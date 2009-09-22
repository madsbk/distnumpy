#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <sys/time.h>
#include <math.h>
#include "mpi.h"

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

/*===================================================================
 *
 * Returns a string describing dtype.
 * Private
 */
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

/*===================================================================
 *
 * Return size of dtype in bytes.
 * Private
 */
static int dtypesize(int dtype)
{
    switch(dtype)
    {
		case NPY_BOOL:
            return sizeof(npy_bool);		
        case NPY_FLOAT:
            return sizeof(npy_float);
        case NPY_DOUBLE:
            return sizeof(npy_double);
        case NPY_CFLOAT:
            return sizeof(npy_cfloat);
        case NPY_CDOUBLE:
            return sizeof(npy_cdouble);
        case NPY_LONG:
            return sizeof(npy_long);            
        default:
            fprintf(stderr, "typesize of %s is not implemented\n",
                    types_numpy2str(dtype));
            MPI_Abort(MPI_COMM_WORLD, -1);
            return 0;
    }
} /* typesize */

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
    int first_process = 0; //Assume array begins at pracess 0.
	
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
	memset(coords, 0, NPY_MAXDIMS*sizeof(int));
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
    fprintf(stderr, "get_dndview, uid %ld does not exist\n", uid);
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
 * Putting the element 'src' into the distributed array 'dest_ary' at
 * coordinates 'lcoords' (local to the 'src_ary' array).
 * No broadcasting, but if 'reduce_axis' >= 0, the 'reduce_axis'
 * dimension will be removed.
 * The coordinates are in Row-major.
 * Private.
 */
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
	MPI_Win_lock(MPI_LOCK_SHARED, rank, 0, *win);
	MPI_Put(src, itemsize, MPI_BYTE, rank, offset*dest_ary->elsize, 
			dest_ary->elsize, MPI_BYTE, *win);
	MPI_Win_unlock(rank, *win);

	return 0;
}/* remote_put */

/*===================================================================
 * 
 * Access a distributed array element located at 'src_ary' using local 
 * coordinates (local to the 'dest_ary' array and NOT the view).
 * Broadcasting if required.
 * The coordinates are in Row-major.
 * Private.
 */
static int remote_get(char *dest, dndview *src_ary, dndview *dest_ary,
					  MPI_Win *win, const npy_intp lcoords[NPY_MAXDIMS])
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
	MPI_Win_lock(MPI_LOCK_SHARED, rank, 0, *win);
	MPI_Get(dest, src_ary->base->elsize, MPI_BYTE, rank, 
			offset*src_ary->base->elsize, 
			src_ary->base->elsize, MPI_BYTE, *win);
	MPI_Win_unlock(rank, *win);

//    printf("5 - reading data at rank %d, value: %lf\n", rank, *((double*)dest));

	return 0;
}/* remote_get */


/*===================================================================
 * 
 * Access a distributed array block located at 'src_ary' using local 
 * block coordinates (local to the 'dest_ary' array and NOT the view).
 * Broadcasting if required.
 * The coordinates are in Row-major.
 * Private.
 */
static PyObject *remote_blockget(char *dest, dndview *src_ary,
                                 dndview *dest_ary, MPI_Win *win,
                                 const npy_intp lcoords[NPY_MAXDIMS])
{
    int i, ai, bi, vi, rbi;
    //A scalar dummy can be returned directly.
    if(src_ary->base->ndims == 0)
    {
        memcpy(dest, src_ary->base->data, src_ary->base->elsize);
        return NULL;
    }


    printf("src_ary-view id: %d\n",src_ary->uid);
    printf("1 - lcoords:");
    for(i=0; i<dest_ary->base->ndims; i++)
        printf(" %ld", lcoords[i]);
    printf("\n");

    
	//Convert local coordinates to global coordinates.
    npy_intp coordsA[NPY_MAXDIMS];
    npy_intp coordsB[NPY_MAXDIMS];
    int pcoords[NPY_MAXDIMS];
	for(i=0; i<dest_ary->base->ndims; i++)
	{
		rank2cart(dest_ary->base->ndims, myrank, pcoords);
		coordsA[i] = pcoords[i] + lcoords[i] *
                     cart_dim_sizes[dest_ary->base->ndims-1][i];
	}

    printf("2 - coordsA (global coords  based on dest_ary):");
    for(i=0; i<dest_ary->base->ndims; i++)
        printf(" %ld", coordsA[i]);
    printf("\n");


    //Convert coords to be based on 'src_ary' instead of 'dest_ary'.
    //Including broadcasting.
    //NB: the coordinates in coordsB is in reverse order after this
    //routine.
    bi=0;
    ai=dest_ary->base->ndims-1;
    rbi=src_ary->base->ndims-1;
    int *cart_dim_size = cart_dim_sizes[src_ary->base->ndims-1];
    int pcoordsR[NPY_MAXDIMS];   
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

    //Reverse again to get the correct order of the coordinates.
    for(i=0; i<src_ary->base->ndims; i++)
    {
        pcoords[i] = pcoordsR[--bi];
        coordsA[i] = coordsB[bi];
    }


    printf("3 - coordsA (local non-block coords based on src_ary):");
    for(i=0; i<src_ary->base->ndims; i++)
        printf(" %ld", coordsA[i]);
    printf("\n");
    printf("3 - pcoords:");
    for(i=0; i<src_ary->base->ndims; i++)
        printf(" %d", pcoords[i]);
    printf("\n");


    //Compute block-/local-/visible-dims.
    npy_intp msgsize = src_ary->base->elsize;
    npy_intp vdims[NPY_MAXDIMS];
	int bdims[NPY_MAXDIMS];	       //Using int instead of npy_intp here 
    int ldims[NPY_MAXDIMS];	       //because MPI_Type_create_subarray
	int subary_coords[NPY_MAXDIMS];//only support int.
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
			subary_coords[bi] = 0;
        }
        else
        {    
			subary_coords[bi] = coordsA[bi];
            //Compute local dims (local to the target MPI-process).
            ldims[bi] = dnumroc(src_ary->base->dims[bi],
                                blocksize,
                                pcoords[bi],
                                cart_dim_size[bi]);

            //Compute a block's dims.
            bdims[bi] = ldims[bi] - subary_coords[bi];
            if(bdims[bi] > blocksize)
                bdims[bi] = blocksize;

            vdims[vi++] = bdims[bi];
        }
        //Computes total message size.
		printf("%d - bdims: %d, ldims: %d, subary_coords: %d\n", bi, bdims[bi], ldims[bi], subary_coords[bi]);
        msgsize *= bdims[bi];
        bi++;
    }

    //TODO: detect earlier.
    if(msgsize < 1)
	{printf("msgsize: %ld\n",msgsize);
        return NULL;
	}
    
	//Convert to a MPI rank number.
	int rank = cart2rank(src_ary->base->ndims, pcoords);

//    printf("myrank %d - MPI_Type_create_subarray - src_ary->base->ndims: %d, ldims: [%d, %d],  bdims: [%d, %d], coordsA: [%d, %d], msgsize: %d\n", myrank, src_ary->base->ndims, ldims[0], ldims[1], bdims[0], bdims[1], coordsA[0], coordsA[1], msgsize);

    //Setup a matching MPI datatype.
    MPI_Datatype arraytype, elementtype;
    MPI_Type_contiguous(src_ary->base->elsize, MPI_BYTE, &elementtype);    
    MPI_Type_create_subarray(src_ary->base->ndims,
                             ldims,
                             bdims,
                             subary_coords,
                             MPI_ORDER_C, elementtype, &arraytype);
    MPI_Type_commit(&arraytype);

	//Transfer element.
	MPI_Win_lock(MPI_LOCK_SHARED, rank, 0, *win);
	MPI_Get(dest, msgsize, MPI_BYTE, rank, 0, 1, arraytype, *win);
	MPI_Win_unlock(rank, *win);

    MPI_Type_free(&arraytype);
    MPI_Type_free(&elementtype);

    printf("4 - reading data at rank %d, value: %lf, %lf, %lf, %lf\n", rank, ((double*)dest)[0],((double*)dest)[1],((double*)dest)[2],((double*)dest)[3]);
    printf("5 - Create array - ndims: %d, vdims: [%d, %d]\n", src_ary->ndims, vdims[0], vdims[1]);

    PyObject *tmp = PyArray_SimpleNewFromData(src_ary->ndims,
                        vdims, src_ary->base->dtype, dest);

    PyObject_Print(tmp, stdout, 0);
    printf("\n");

    
	return tmp;
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

    //Acummulate total number of local elements and save it.
    npy_intp nelements = 1;
    for(i=0; i < ary->ndims; i++)
    {
        ary->localdims[i] = dnumroc(ary->dims[i],
                            blocksize, coords[i], dims[i]);
        nelements *= ary->localdims[i];
    }
    ary->nelements = nelements;

    //Acummulate max total local size.
    npy_intp localsize = 1;
    for(i=0; i<ndims; i++)
        localsize *= ceil(ceil((double) ary->dims[i]/
                     blocksize)/dims[i]) * blocksize;
 
    //Allocate local data.
    MPI_Alloc_mem(localsize * ary->elsize, 
				  MPI_INFO_NULL, &ary->data);

    //Allocate and copy the array to the views's base.
	view->base = malloc(sizeof(dndarray));
    memcpy(view->base, ary, sizeof(dndarray));

    //Save the new view.
    put_dndview(view);

    //Print debug info.
    #ifdef DISTNUMPY_DEBUG
        printf("CREATE_ARRAY - uid: %ld, dtype: %s(%d)," 
		       "localsize: %ld, nelem: %ld, ndims: %d, dims: {", 
			   (long) view->uid,
               types_numpy2str(ary->dtype), dtypesize(ary->dtype),
               (long)localsize, (long)nelements, ndims);
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
        printf("DESTROY_ARRAY - uid: %ld\n", uid);
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
    int elsize = dtypesize(base->dtype);
       
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

	MPI_Win win;
    MPI_Win_create(base->data, base->nelements * 
				   elsize, 1, MPI_INFO_NULL, 
                   MPI_COMM_WORLD, &win);

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
        MPI_Win_lock(MPI_LOCK_SHARED, rank, 0, win);
        MPI_Put(value, elsize, MPI_BYTE, rank, offset*elsize, elsize, 
                MPI_BYTE, win);
        MPI_Win_unlock(rank, win);
    }
    MPI_Win_free(&win);
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

	MPI_Win win;
    MPI_Win_create(base->data, base->nelements * 
				   elsize, 1, MPI_INFO_NULL, 
                   MPI_COMM_WORLD, &win);

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
        MPI_Win_lock(MPI_LOCK_SHARED, rank, 0, win);
        MPI_Get(retdata, elsize, MPI_BYTE, rank, offset*elsize, elsize, 
                MPI_BYTE, win);
        MPI_Win_unlock(rank, win);
    }
   
    MPI_Win_free(&win);
    
} /* do_GET_ITEM */



/*===================================================================
 *
 * Message handler for UFUNC.
 * Private.
 */
static void do_UFUNC(npy_intp arylist[NPY_MAXARGS], int narys, 
					 int nout_arys, int scalar_dtype, int scalar_size,
					 char* scalar, char* op)
{
    int i, j;
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
		{	//Create a dummy view for the scalar.
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
        printf("UFUNC - nocomm_method\n");
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
                             views[i]->base->elsize, NPY_WRITEABLE,
                             NULL));
        }       

		//Call the python operator.
		PyObject_CallObject(PyOp, arytuple);

        //Clean up
        for(i=0; i<narys; i++)
            if(views[i]->base->ndims == 0)//Scalar dummy.
            {
                free(views[i]->base);
                free(views[i]);
            }            
		return;
	}//End of the simple method.

	MPI_Win win[NPY_MAXARGS];
	for(i=0; i<nin; i++)
		MPI_Win_create(views[i]->base->data, views[i]->base->nelements * 
					   views[i]->base->elsize, 1, 
					   MPI_INFO_NULL, MPI_COMM_WORLD, &win[i]);

	npy_intp coords[NPY_MAXDIMS];
	memset(coords, 0, NPY_MAXDIMS * sizeof(npy_intp));
    char *buffer[NPY_MAXARGS];
	if(block_method && 0)
	{
        printf("UFUNC - block_method\n");

        //Allocate a buffer for every array.
        for(i=0; i<nin; i++)
        {
            npy_intp s = views[i]->base->elsize;
            for(j=0; j<views[i]->base->ndims; j++)
                s *= blocksize;
            buffer[i] = malloc(s);
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
        
        int notfinished = 1;
        npy_intp offset = 0;
        while(notfinished)
        {

            printf("\nIteration\n");
            PyObject *arytuple = PyTuple_New(narys);
            for(j=0; j<nin; j++)
            {
                PyObject *tmp = remote_blockget(buffer[j], views[j], 
									            lastary, &win[j], 
												coords);
                if(tmp == NULL)
                {
					printf("Finished since block not on this proc.\n");
                    notfinished = 0;
                    break;
                }  
                PyTuple_SET_ITEM(arytuple, j, tmp);
            }
            if(notfinished == 0)
                break;
                
            //Compute a block's dims for the dest array.
            npy_intp bdims[NPY_MAXDIMS];
            npy_intp strides[NPY_MAXDIMS];
            offset = 0;
            for(j=0; j < lastary->base->ndims; j++)
            {
                npy_intp stride = lastary->base->elsize;
                int s;
                for(s=j+1; s<lastary->base->ndims; s++)
                    stride *= lastary->base->localdims[s];
                strides[j] = stride;

                bdims[j] = lastary->base->localdims[j] - 
						   coords[j] * blocksize;
                if(bdims[j] > blocksize)
                    bdims[j] = blocksize;

                offset += stride * (coords[j] * blocksize);
            }
            
            j=2;
            PyObject *tmp =  PyArray_New(&PyArray_Type,
                                         views[j]->base->ndims,
                                         bdims,
                                         views[j]->base->dtype,
                                         strides,
                                         views[j]->base->data + offset,
                                         views[j]->base->elsize,
                                         NPY_WRITEABLE, NULL);


    printf("stride: [%ld, %ld], offset: %ld\n", PyArray_STRIDES(tmp)[0], PyArray_STRIDES(tmp)[1], offset);

            for(j=nin; j<narys; j++)
                PyTuple_SET_ITEM(arytuple, j, tmp);
 
            //Call the python operator.
            PyObject_CallObject(PyOp, arytuple);          


            PyObject_Print(tmp, stdout, 0);
            printf("\n");

            printf("Whole ret array :\n");
            tmp = PyArray_SimpleNewFromData(views[2]->base->ndims, views[2]->base->dims, views[2]->base->dtype, views[2]->base->data);
            PyObject_Print(tmp, stdout, 0);
            printf("\n");

            //Iterate coords one block.
            for(j=lastary->base->ndims-1; j >= 0; j--)
            {
                if(++coords[j] >= ceil(lastary->base->localdims[j] / (double) blocksize))
                {
                    //We a finished, if wrapping around.
                    if(j == 0)
                    {
                        notfinished = 0;
                        break;
                    }
                    coords[j] = 0;
                }
                else
                    break;
            }
        }
    }//End of the block method.
    else
    {
        printf("UFUNC - NO simple method\n");
        
        //Allocate a buffer for every array.
        for(i=0; i<nin; i++)
            buffer[i] = malloc(views[i]->base->elsize);
            
        for(i=0; i<lastary->base->nelements; i++)
        {       
            int not_inside_view = 0;
            for(j=0; j<nin; j++)
                not_inside_view += remote_get(buffer[j], views[j],
                                              lastary, &win[j],
                                              coords);
            //If just one is outside we jump to next element.
            if(!not_inside_view)
            {
                npy_intp one = 1;
                PyObject *arytuple = PyTuple_New(narys);
                for(j=0; j<nin; j++)
                    PyTuple_SET_ITEM(arytuple, j, PyArray_SimpleNewFromData(1, 
                                     &one, views[j]->base->dtype, buffer[j]));
                    
                for(j=nin; j<narys; j++)
                    PyTuple_SET_ITEM(arytuple, j, PyArray_SimpleNewFromData(1, 
                                     &one, views[j]->base->dtype,
                                     views[j]->base->data +
                                     i*views[j]->base->elsize));
                
                //Call the python operator.
                PyObject_CallObject(PyOp, arytuple);
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
    }

	//Clean up
	for(i=0; i<narys; i++)
	{
        if(views[i]->base->ndims == 0)//Scalar dummy.
        {
            free(views[i]->base);
            free(views[i]);
        }            
	}
    for(i=0; i<nin; i++)
    {
        MPI_Win_free(&win[i]);
        free(buffer[i]);
    }

} /* do_UFUNC */

/*===================================================================
 *
 * Message handler for UFUNC_REDUCE.
 * out_ary_uid is the output array uid or scalar datatype.
 * Private.
 */
static void do_UFUNC_REDUCE(npy_intp in_ary_uid, npy_intp out_ary_uid,
							char* out_scalar_address, int axis, char* op)
{
    int i,j;
	dndview *in_view = get_dndview(in_ary_uid);
	dndarray *in_ary = in_view->base;
    PyObject *local_scalar, *tmpobj, *tmp_scalar;

    //Get Python function.
    PyObject *PyOp = PyObject_GetAttrString(ufunc_module, op);
    if(PyOp == NULL)
    {
        fprintf(stderr, "UFUNC : unknown operator: %s\n", op);
        MPI_Abort(MPI_COMM_WORLD, -1);
    }

	//If output is a scalar we just use this simple method.
	if(in_ary->ndims == 1)
	{
		int out_dtype = out_ary_uid;
		
		//Print debug info.
		#ifdef DISTNUMPY_DEBUG
			printf("UFUNC_REDUCE (%s) on - in: %ld, axis: %d, "
				   "returning a scalar at address: %p\n", op,
				   (long)in_ary_uid, axis, out_scalar_address);
		#endif

		//The master should use the final output address.
		if(myrank == 0)
			local_scalar = PyArray_SimpleNewFromData(0, NULL,
						   out_dtype, out_scalar_address);
		else//And the rest just use a new location.
			local_scalar = PyArray_SimpleNew(0, NULL, out_dtype);
	
		if(in_ary->localdims[0] > 0)
		{
            tmpobj = PyArray_SimpleNewFromData(1, &in_ary->nelements,
                                               in_ary->dtype,
                                               in_ary->data);
			//Compute local reduce.
			PyObject_CallMethod(PyOp, "reduce", "(O,i,O,O)",
                                tmpobj, axis, Py_None, local_scalar);

			//Send local output scalar to the master process.
			if(myrank > 0)
				MPI_Send(PyArray_DATA(local_scalar),
						 dtypesize(out_dtype), MPI_BYTE, 0, 0,
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
						 dtypesize(out_dtype), MPI_BYTE,
						 cart2rank(1, &zero), 0, MPI_COMM_WORLD,
						 MPI_STATUS_IGNORE);
			}
			
			//We need a tmp scalar for receiving scalars. 
            tmp_scalar = PyArray_SimpleNew(0, NULL, out_dtype);

			//For all ranks in dimension 'axis', we reduces the scalars
			//to local_scalar at the master.
			for(i=1; i < cart_dim_sizes[0][0]; i++)
			{
				if(0 == dnumroc(in_ary->dims[0], blocksize,
								i, cart_dim_sizes[0][0]))
					continue;

				MPI_Recv(PyArray_DATA(tmp_scalar), dtypesize(out_dtype),
						 MPI_BYTE, cart2rank(1, &i), 0, MPI_COMM_WORLD,
						 MPI_STATUS_IGNORE);
				PyObject_CallFunction(PyOp, "(O,O,O)", local_scalar,
									  tmp_scalar, local_scalar);
			}
            Py_DECREF(tmp_scalar);
		}
        Py_DECREF(local_scalar);
		return;
	}

	dndview *out_view = get_dndview(out_ary_uid);
	dndarray *out_ary = out_view->base;
	
	//Print debug info.
	#ifdef DISTNUMPY_DEBUG
		printf("UFUNC_REDUCE (%s) on - in: %ld, out: %ld, axis: %d\n", op,
			   (long)in_ary_uid, (long)out_ary_uid, axis);
	#endif

	//There is two kind of roles for MPI-processes. They can be
	//receivers or senders. The receivers are receiving the sub-results
	//computed by the senders.
	//All MPI-processes with zero-coordinate in the axis-dimension are
	//receivers and all other MPI-processes are senders.
	//The senders should send its sub-results to the receiver with
	//identical coordinates beside zero in the axis-dimension.

	//First we creates are MPI window for the output array.
	MPI_Win win;
	MPI_Win_create(out_ary->data, out_ary->nelements * 
					   dtypesize(out_ary->dtype), 1, MPI_INFO_NULL, 
					   MPI_COMM_WORLD, &win);	

    PyObject *tmp = PyArray_SimpleNewFromData(in_ary->ndims,
                                              in_ary->localdims,
                                              in_ary->dtype,
                                              in_ary->data);

	//All MPI-processes computes its own sub-result.
	PyObject *subres = PyObject_CallMethod(PyOp, "reduce", "(O,i)",
                                           tmp, axis);
    Py_DECREF(tmp);

	int subres_dtypesize = dtypesize(PyArray_TYPE(subres));

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
					 dtypesize(out_ary->dtype), MPI_BYTE, rank, 0,
					 MPI_COMM_WORLD, MPI_STATUS_IGNORE);

			//We use ufunc to "reduce" the two arrays over the
			//non-existing 'axis' dimension.
			PyObject_CallFunction(PyOp, "(O,O,O)", subres, tmpres,
								  subres);
		}

		//Reusing cart_coords.
		npy_intp lcoords[NPY_MAXDIMS];
		memset(lcoords, 0, in_ary->ndims * sizeof(npy_intp));
		char *element = PyArray_DATA(subres);
		for(i=0; i<PyArray_SIZE(subres); i++)
		{
			remote_put(element, in_ary, out_ary, axis, &win, lcoords);

			//Iterate coords one element.
			for(j=in_ary->ndims-1; j >= 0; j--) 
			{
				//We ignore the 'axis' dimension - it stays on zero.
				if(j != axis && ++lcoords[j] >= in_ary->localdims[j])
						lcoords[j] = 0;
				else if(j != axis)
					break;
			}
			element += subres_dtypesize;
		}
	}
	else//MPI-process is a sender.
	{
		//Send subresult to receiver.
		cart_coords[axis] = 0;
	
		MPI_Send(PyArray_DATA(subres), PyArray_SIZE(subres) *
				 subres_dtypesize, MPI_BYTE, cart2rank(in_ary->ndims,
				 cart_coords), 0, MPI_COMM_WORLD);
	}

	//Cleanup
	MPI_Win_free(&win);
    Py_DECREF(subres);

} /* do_UFUNC_REDUCE */

/*===================================================================
 *
 * Message handler for ZERO_FILL.
 * Private.
 */
static void do_ZEROFILL(dndview *view)
{
    int i;
    //Print debug info.
    #ifdef DISTNUMPY_DEBUG    
        printf("ZEROFILL - uid: %ld\n", (long)view->uid);
    #endif
    
    //Chech if the view covers the whole array.
    int cover_whole = 1;
    for(i=0; i<view->nslice;i++)
    {
        if(view->slice[i].nsteps == SingleIndex ||
           view->slice[i].nsteps == PseudoIndex ||
           view->slice[i].start != 0 || view->slice[i].step != 1 ||
           view->slice[i].nsteps != view->base->dims[i])
        {
            cover_whole = 0;
        }
    }

    if(cover_whole)
    {
        memset(view->base->data, 0, view->base->nelements *
                                    dtypesize(view->base->dtype));
    }
    else
    {
        fprintf(stderr, "ZEROFILL with views not implemented\n");
		MPI_Abort(MPI_COMM_WORLD, -1);
    }
} /* do_ZEROFILL */

/*NUMPY_API
 *===================================================================
 * Create dndarray.
 * Public
 */
static intp
dnumpy_create_dndarray(int ndims, intp dims[NPY_MAXDIMS], int dtype)
{
    int i;
    #ifdef DISTNUMPY_DEBUG
        if(ndims > NPY_MAXDIMS)
        {
            fprintf(stderr,
                "dnumpy_create_dndarray failed: ndims > NPY_MAXDIMS\n");
            MPI_Abort(MPI_COMM_WORLD, -1);
        }
        if(ndims <= 0)
        {
            fprintf(stderr,
                "dnumpy_create_dndarray failed: ndims <= 0\n");
            MPI_Abort(MPI_COMM_WORLD, -1);
        }            
        if(myrank != 0)
        {
            fprintf(stderr,
                "dnumpy_create_dndarray failed: myrank != 0\n");
            MPI_Abort(MPI_COMM_WORLD, -1);
        }
    #endif

	//Create dndarray.
    dndarray *newarray = malloc(sizeof(dndarray));

    newarray->dtype = dtype;
    newarray->elsize = dtypesize(dtype);
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

    MPI_Barrier(MPI_COMM_WORLD);
    MPI_Bcast(msg, 2*sizeof(npy_intp) + sizeof(dndarray) + sizeof(dndview),
              MPI_BYTE, 0, MPI_COMM_WORLD);

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

    MPI_Barrier(MPI_COMM_WORLD);
    MPI_Bcast(msg, 3 * sizeof(npy_intp), MPI_BYTE, 0, MPI_COMM_WORLD);

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

    MPI_Barrier(MPI_COMM_WORLD);
    MPI_Bcast(msg, 3 * sizeof(npy_intp) + sizeof(dndview), MPI_BYTE, 0,
			  MPI_COMM_WORLD);

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
    int elsize = dtypesize(view->base->dtype);

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

    MPI_Barrier(MPI_COMM_WORLD);
    MPI_Bcast(msg, 3 * sizeof(npy_intp) + elsize + ndims * sizeof(npy_intp),
              MPI_BYTE, 0, MPI_COMM_WORLD);

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

    MPI_Barrier(MPI_COMM_WORLD);
    MPI_Bcast(msg, 3 * sizeof(npy_intp) + ndims * sizeof(npy_intp),
              MPI_BYTE, 0, MPI_COMM_WORLD);

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
			 int nout_arys, char *op)
{
    int i;
    //Tell slaves about the destruction.
    msg[0] = DNPY_UFUNC;
    msg[1] = narys;
	msg[2] = nout_arys;
	msg[3] = 0; //Scalar datatype.
	msg[4] = 0; //Scalar datasize, zero if no scalar is in arylist.

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
			msg[3] = arylist[i]->descr->type_num;
			msg[4] = arylist[i]->descr->elsize;
			memcpy(msg+5, arylist[i]->data, msg[4]);
			break;
		}
	
	//Copy arylist's uid to msg.
	for(i=0; i<narys;i++)
		memcpy(((char*)(msg+5))+msg[4]+i*sizeof(npy_intp),
				&arylist[i]->dnduid, sizeof(npy_intp));
	
	//Copy the string op.
    strcpy(((char*)(msg+5+msg[1]))+msg[4], op);

    *(((char*)(msg+5+msg[1]))+msg[4]+strlen(op)+1) = DNPY_MSG_END;

    MPI_Barrier(MPI_COMM_WORLD);
    MPI_Bcast(msg, (6+msg[1])*sizeof(npy_intp) + msg[4] + strlen(op)+1,
			  MPI_BYTE, 0, MPI_COMM_WORLD);

    //The master should also do the operation.
    #ifdef DISTNUMPY_DEBUG
        printf("Rank 0 received msg: ");
    #endif

    do_UFUNC((npy_intp*)(((char*)(msg+5))+msg[4]), msg[1], msg[2], msg[3],
			msg[4], (char*) (msg+5), op);
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
        for(i=0; i<out->nslice;i++)
        {
            if(out->slice[i].nsteps == SingleIndex ||
               out->slice[i].nsteps == PseudoIndex ||
               out->slice[i].start != 0 || out->slice[i].step != 1 ||
               out->slice[i].nsteps != out->base->dims[i])
            {
                PyErr_SetString(PyExc_RuntimeError,
                                "reduce on distributed arrays must"
                                "be whole arrays and not a partial"
                                " view of a underlying array.");
                return -1;
            }
        }
    }
        
    //Tell slaves about the destruction
    msg[0] = DNPY_UFUNC_REDUCE;
    msg[1] = in_ary->dnduid;
	msg[2] = out_ary->dnduid;
	msg[3] = axis;

	//If out_ary is a scalar we will send the datatype instead of uid.
	if(msg[2] == 0)
		msg[2] = out_ary->descr->type_num;

    strcpy((char*)(msg+4), op);

    *(((char*)(msg+4))+strlen(op)+1) = DNPY_MSG_END;

    MPI_Barrier(MPI_COMM_WORLD);
    MPI_Bcast(msg, 5*sizeof(npy_intp)+strlen(op)+1, MPI_BYTE, 0,
              MPI_COMM_WORLD);

    //The master should also do the operation
    #ifdef DISTNUMPY_DEBUG
        printf("Rank 0 received msg: ");
    #endif

    do_UFUNC_REDUCE(msg[1], msg[2], out_ary->data, axis, op);
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

    MPI_Barrier(MPI_COMM_WORLD);
    MPI_Bcast(msg, 3 * sizeof(npy_intp), MPI_BYTE, 0, MPI_COMM_WORLD);

    //The master should also do the operation.
    #ifdef DISTNUMPY_DEBUG
        printf("Rank 0 received msg: ");
    #endif
    do_ZEROFILL(view);
} /* dnumpy_zerofill */


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
    int argc = 0; char** argv;
    MPI_Init(&argc, &argv);
     
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
		//	coord (0,0): rank 0.
		//	coord (0,1): rank 1.
		//	coord (1,0): rank 2.
		//	coord (1,1): rank 3.
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
        MPI_Barrier(MPI_COMM_WORLD);
        MPI_Bcast(msg, 2 * sizeof(npy_intp), MPI_BYTE, 0, MPI_COMM_WORLD);
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
 */
static void
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

        MPI_Barrier(MPI_COMM_WORLD);
        MPI_Bcast(msg, 3 * sizeof(npy_intp), MPI_BYTE, 0, MPI_COMM_WORLD);

		#ifdef DISTNUMPY_DEBUG
        printf("Rank 0 received msg: ");
		#endif
        do_INIT_BLOCKSIZE(msg[1]);
        return;
    }
        
    int shutdown = 0;
    while(shutdown == 0)//Work loop
    {
        char *t1, *t2, *t3;
        npy_intp d1, d2, d3, d4;
        dndview *ary;
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
                t2 = t1+dtypesize(ary->base->dtype);
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
				t1 = msg_data+sizeof(npy_intp)*4;
                t2 = t1+d4;				
				t3 = t2+d1*sizeof(npy_intp);
                do_UFUNC((npy_intp *)t2,d1,d2,d3,d4,t1,t3);
                break;
            case DNPY_UFUNC_REDUCE:			
				d1 = *((npy_intp*)msg_data);
				d2 = *(((npy_intp*)msg_data)+1);
				d3 = *(((npy_intp*)msg_data)+2);
				t1 = msg_data+sizeof(npy_intp)*3;
				//3. arg: msg_data is just used as a non-NULL pointer.
				do_UFUNC_REDUCE(d1, d2, NULL, d3, t1);
                break;
            case DNPY_ZEROFILL:
                do_ZEROFILL(get_dndview(*((npy_intp*)msg_data)));
                break;
            default:
                fprintf(stderr, "Unknown msg: %ld\n", (long)msg[0]);
                MPI_Abort(MPI_COMM_WORLD, -1);
        }
    }
    dnumpy_exit();
    exit(0);
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


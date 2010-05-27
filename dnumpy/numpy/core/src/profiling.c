#include "numpy/profiling.h"

#ifdef HAS_PROFILING
inline int __MPI_Barrier(PROFINFO, MPI_Comm comm) {
    CREATE_WRAPPER_BODY(MPI_Barrier, PROF_WAIT_TIME, comm)
}

inline int __MPI_Bcast(PROFINFO, void* buffer, int count, MPI_Datatype datatype, int root, MPI_Comm comm) {
    CREATE_WRAPPER_BODY(MPI_Bcast, PROF_COMM_TIME, buffer, count, datatype, root, comm)
}

inline int __MPI_File_read_all(PROFINFO, MPI_File fh, void *buf, int count, MPI_Datatype datatype, MPI_Status *status) {
    CREATE_WRAPPER_BODY(MPI_File_read_all, PROF_IO_TIME, fh, buf, count, datatype, status)
}

inline int __MPI_File_write_all(PROFINFO, MPI_File fh, void *buf, int count, MPI_Datatype datatype, MPI_Status *status) {
    CREATE_WRAPPER_BODY(MPI_File_write_all, PROF_IO_TIME, fh, buf, count, datatype, status)
}

inline int __MPI_Get (PROFINFO, void *origin_addr, int origin_count, MPI_Datatype origin_datatype, int target_rank, MPI_Aint target_disp, int target_count, MPI_Datatype target_datatype, MPI_Win win) {
    CREATE_WRAPPER_BODY(MPI_Get, PROF_TRANSFER_TIME, origin_addr, origin_count, origin_datatype, target_rank, target_disp, target_count, target_datatype, win)
}

inline int __MPI_Put(PROFINFO, void *origin_addr, int origin_count, MPI_Datatype origin_datatype, int target_rank, MPI_Aint target_disp, int target_count, MPI_Datatype target_datatype, MPI_Win win) {
    CREATE_WRAPPER_BODY(MPI_Put, PROF_TRANSFER_TIME, origin_addr, origin_count, origin_datatype, target_rank, target_disp, target_count, target_datatype, win)
}

inline int __MPI_Recv(PROFINFO, void* buf, int count, MPI_Datatype datatype, int source, int tag, MPI_Comm comm, MPI_Status *status) {
    CREATE_WRAPPER_BODY(MPI_Recv, PROF_TRANSFER_TIME, buf, count, datatype, source, tag, comm, status)
}
inline int __MPI_Send(PROFINFO, void* buf, int count, MPI_Datatype datatype, int dest, int tag, MPI_Comm comm) {
    CREATE_WRAPPER_BODY(MPI_Send, PROF_TRANSFER_TIME, buf, count, datatype, dest, tag, comm)
}

inline int __MPI_Win_unlock(PROFINFO, int rank, MPI_Win win) {
    CREATE_WRAPPER_BODY(MPI_Win_unlock, PROF_WAIT_TIME, rank, win)
}


inline int _MPI_Finalize() {
    prof_total_time = MPI_Wtime() - prof_init_time;
    
    return MPI_Finalize();
}

inline int _MPI_Init(int *argc,char ***argv) {
    if (0 != (prof_mpierrno = MPI_Init(argc, argv)))
        return prof_mpierrno;
    prof_init_time = MPI_Wtime();
    return 0;
}

#endif

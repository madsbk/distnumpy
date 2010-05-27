#ifndef PROFILING_H
#define PROFILING_H

#include <mpi.h>

#ifdef HAS_PROFILING

int prof_enabled;
int prof_mpierrno;

#define PROFINFO const char *__func, int __linenum

inline int __MPI_Barrier(PROFINFO, MPI_Comm comm);
inline int __MPI_Bcast(PROFINFO, void* buffer, int count, MPI_Datatype datatype, int root, MPI_Comm comm);
inline int __MPI_File_read_all(PROFINFO, MPI_File fh, void *buf, int count, MPI_Datatype datatype, MPI_Status *status);
inline int __MPI_File_write_all(PROFINFO, MPI_File fh, void *buf, int count, MPI_Datatype datatype, MPI_Status *status);
inline int __MPI_Get(PROFINFO, void *origin_addr, int origin_count, MPI_Datatype origin_datatype, int target_rank, MPI_Aint target_disp, int target_count, MPI_Datatype target_datatype, MPI_Win win);
inline int __MPI_Put(PROFINFO, void *origin_addr, int origin_count, MPI_Datatype origin_datatype, int target_rank, MPI_Aint target_disp, int target_count, MPI_Datatype target_datatype, MPI_Win win);
inline int __MPI_Recv(PROFINFO, void* buf, int count, MPI_Datatype datatype, int source, int tag, MPI_Comm comm, MPI_Status *status);
inline int __MPI_Send(PROFINFO, void* buf, int count, MPI_Datatype datatype, int dest, int tag, MPI_Comm comm);
inline int __MPI_Win_unlock(PROFINFO, int rank, MPI_Win win);

inline int _MPI_Finalize(void);
inline int _MPI_Init(int *argc,char ***argv);


#define _MPI_Barrier(...) do { \
    __MPI_Barrier(__func__, __LINE__, __VA_ARGS__); \
} while (0)

#define _MPI_Bcast(...) do { \
    __MPI_Bcast(__func__, __LINE__, __VA_ARGS__); \
} while (0)

#define _MPI_File_read_all(...) do { \
    __MPI_File_read_all(__func__, __LINE__, __VA_ARGS__); \
} while (0)

#define _MPI_File_write_all(...) do { \
    __MPI_File_write_all(__func__, __LINE__, __VA_ARGS__); \
} while (0)

#define _MPI_Get(...) do { \
    __MPI_Get(__func__, __LINE__, __VA_ARGS__); \
} while (0)

#define _MPI_Put(...) do { \
    __MPI_Put(__func__, __LINE__, __VA_ARGS__); \
} while (0)

#define _MPI_Recv(...) do { \
    __MPI_Recv(__func__, __LINE__, __VA_ARGS__); \
} while (0)

#define _MPI_Send(...) do { \
    __MPI_Send(__func__, __LINE__, __VA_ARGS__); \
} while (0)

#define _MPI_Win_unlock(...) do { \
    __MPI_Win_unlock(__func__, __LINE__, __VA_ARGS__); \
} while (0)

#define PROF_COMM_TIME      (1 << 0)
#define PROF_TRANSFER_TIME  (1 << 1)
#define PROF_WAIT_TIME      (1 << 2)
#define PROF_IO_TIME        (1 << 3)

double prof_init_time = 0.0;
double prof_total_time = 0.0;
double prof_tmp_begin, prof_tmp_end;
double prof_wait_time = 0.0;
double prof_comm_time = 0.0;
double prof_transfer_time = 0.0;
double prof_io_time = 0.0;

#define CREATE_WRAPPER_BODY(what, type, ...) \
    if (prof_enabled) { \
        prof_tmp_begin = MPI_Wtime(); \
        prof_mpierrno = what(__VA_ARGS__); \
        prof_tmp_end = MPI_Wtime(); \
        switch (type) { \
        case PROF_COMM_TIME: \
            prof_comm_time += prof_tmp_end - prof_tmp_begin; \
            break; \
        case PROF_TRANSFER_TIME: \
            prof_transfer_time += prof_tmp_end - prof_tmp_begin; \
            break; \
        case PROF_WAIT_TIME: \
            prof_wait_time += prof_tmp_end - prof_tmp_begin; \
            break; \
        case PROF_IO_TIME: \
            prof_io_time += prof_tmp_end - prof_tmp_begin; \
            break; \
        } \
        return prof_mpierrno; \
    } else { \
        return what(__VA_ARGS__); \
    }

#else

#define _MPI_Barrier(...) MPI_Barrier(__VA_ARGS__)
#define _MPI_Bcast(...) MPI_Bcast(__VA_ARGS__)
#define _MPI_File_read_all(...) MPI_File_read_all(__VA_ARGS__)
#define _MPI_File_write_all(...) MPI_File_write_all(__VA_ARGS__)
#define _MPI_Finalize() MPI_Finalize()
#define _MPI_Get(...) MPI_Get(__VA_ARGS__)
#define _MPI_Init(...) MPI_Init(__VA_ARGS__)
#define _MPI_Put(...) MPI_Put(__VA_ARGS__)
#define _MPI_Recv(...) MPI_Recv(__VA_ARGS__)
#define _MPI_Send(...) MPI_Send(__VA_ARGS__)
#define _MPI_Win_unlock(...) MPI_Win_unlock(__VA_ARGS__)

#endif /* HAS_PROFILING */

#endif /* PROFILING_H */

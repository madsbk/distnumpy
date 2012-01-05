#include <math.h>
#include <string.h>
#include <stdio.h>
#include <stdlib.h>

#include <sys/time.h>
struct timeval tv;
struct timezone tz;
unsigned long long start, end, delta;

int main(int argc, char *argv[])
{
    int size = atoi(argv[1]);
    int count = atoi(argv[2]);
    double *A = malloc(size * size * sizeof(double));
    double *B = malloc(size * sizeof(double));
    double *h = malloc(size * sizeof(double));
    memset(h, 0, size * sizeof(double));
    double *hnew = malloc(size * sizeof(double));
    double *tmp0 = malloc(size * size * sizeof(double));
    double *tmp1 = malloc(size * sizeof(double));
    double *tmp2 = malloc(size * sizeof(double));
    double *AD = malloc(size * sizeof(double));

    int n, i, j;
    for(i=0; i<size; i++)
    {
        B[i] = rand()/(double)RAND_MAX;
        h[i] = rand()/(double)RAND_MAX;
        AD[i] = rand()/(double)RAND_MAX;
    }
    for(i=0; i<size*size; i++)
        A[i] = rand()/(double)RAND_MAX;


    double dmax = 1.0;
    //double tol = 0.005;

    gettimeofday(&tv, &tz);
    start = (long long)tv.tv_usec + ((long long)tv.tv_sec)*1000000;

    //while(dmax > tol)
    for(n=0; n<count; n++)
    {
        //multiply(A,h,tmp0)
        //#pragma omp parallel for
        for(i=0; i<size; i++)
            for(j=0; j<size; j++)
                tmp0[i*size+j] = A[i*size+j] * h[j];

        //add.reduce(tmp0,1,out=tmp1)
        //#pragma omp parallel for
        for(i=0; i<size; i++)
        {
            double sum = 0.0;
            for(j=0; j<size; j++)
                sum += tmp0[i*size+j];
            tmp1[i] = sum;
        }

        //tmp2 = AD
        memcpy(tmp2, AD, size * sizeof(double));
        dmax = 0.0;
        #pragma omp parallel for
        for(i=0; i<size; i++)
        {
            //subtract(B, tmp1, tmp1)
            tmp1[i] = B[i] - tmp1[i];
            //divide(tmp1, tmp2, tmp1)
            tmp1[i] = tmp1[i] / tmp2[i];
            //hnew = h + tmp1
            hnew[i] = h[i] + tmp1[i];
            //subtract(hnew,h,tmp2)
            tmp2[i] = hnew[i] - h[i];
            //divide(tmp2,h,tmp1)
            tmp1[i] = tmp2[i] / h[i];
            //absolute(tmp1,tmp1)
            tmp1[i] = fabs(tmp1[i]);
            //dmax = maximum.reduce(tmp1)
            if(tmp1[i] > dmax)
                dmax = tmp1[i];
        }
        //h = hnew
        memcpy(h, hnew, size * sizeof(double));
    }

    gettimeofday(&tv, &tz);
    end = (long long)tv.tv_usec + ((long long)tv.tv_sec)*1000000;
    delta = end - start;

    printf("Iter: %d size: %d time: %lld (ANSI C)\n", count, size, delta);

    return 0;
}

#include <math.h>
#include <string.h>
#include <stdio.h>
#include <stdlib.h>

#include <sys/time.h>
struct timeval tv;
struct timezone tz;
unsigned long long start, end, delta;

void copy2workarray(double* inout, double* w, int size, int fullsize)
{
    int i,j;
    inout += fullsize;//skip one row.
    for(j=0; j<size; ++j)
    {
        ++inout;
        for(i=0; i<size; ++i)
        {
            *w = *inout;
            ++w;
            ++inout;
        }
        ++inout;
    }


}

int main(int argc, char *argv[])
{
    int size = atoi(argv[1]);
    int count = atoi(argv[2]);
    int n,i,j;
    int fullsize = size + 2;
    double *t;

    double *inout = malloc(fullsize * fullsize * sizeof(double));//Input and output array
    for(n=0; n<fullsize * fullsize; ++n)
        inout[n] = n;

    double *work = malloc(size * size * sizeof(double));//Tmp array
    for(n=0; n<size * size; ++n)
        work[n] = 0;


    gettimeofday(&tv, &tz);
    start = (long long)tv.tv_usec + ((long long)tv.tv_sec)*1000000;

    for(n=0; n<count; n++)
    {


        copy2workarray(inout, work, size, fullsize);
/*
        printf("a:\n");
        t = inout;
        for(i=0; i<fullsize; ++i)
        {
            for(j=0; j<fullsize; ++j)
            {
                printf(" %3.0lf", *t);
                ++t;
            }
            printf("\n");
        }
        printf("\n");
*/
        double *a = inout;
        double *w = work;
        for(i=0; i<size; ++i)
        {
            double *up    = a+1;
            double *left  = a+fullsize;
            double *right = a+fullsize+2;
            double *down  = a+1+fullsize*2;

            for(j=0; j<size; ++j)
            {
                *w += *up + *left + *right + *down;
                ++w;
                ++up;
                ++left;
                ++right;
                ++down;
            }
            a += fullsize;
        }

/*
        printf("w:\n");
        t = work;
        for(i=0; i<size; ++i)
        {
            for(j=0; j<size; ++j)
            {
                printf(" %3.0lf", *t);
                ++t;
            }
            printf("\n");
        }
        printf("\n");
*/
    }


    gettimeofday(&tv, &tz);
    end = (long long)tv.tv_usec + ((long long)tv.tv_sec)*1000000;
    delta = end - start;

    printf("Iter: %d size: %d time: %lld (ANSI C)\n", count, size, delta);

    free(inout);
    free(work);

    return 0;
}

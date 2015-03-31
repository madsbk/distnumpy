# Optimization hierarchy #
The performance impact when optimizing a NumPy computation in the general case is very limited. It is far easier to optimize and make computation shortcuts if we have some predefined assumptions.

We therefore introduce a hierarchy of implementations all optimized for specific computation scenarios. When applying a NumPy computation, a search through the hierarchy will determine the must optimized implementation for that particular computation.

## Universal function ##

<ol>
<li>The ufunc apply an operation on a number of arrays element-wise and it is therefore possible to use a very simple computation method if all arrays have identical shape and are not sub-array views. In this case, it is a simple matter of applying the operation individually by all MPI-processes and communication is therefore not needed.<br>
<br>
</li>
<li>Communication between MPI-processes is required if we allow arrays to have different shapes. Instead of working (communicating and computing) on one array-element at a time, the overhead can be reduced by blocking array-elements together. It becomes very complicated and will require a significant administration overhead if blocking is performed on arbitrary views. The problem is the data distribution scheme (<a href='DataLayout.md'>N-Dimensional Block Cyclic Distribution</a>) combined with views that contains multiple steps. It becomes hard for a MPI-process to determine which part of its local data the view defines. By only allow views with 1-length steps, we have implemented a very efficient blocked universal function.<br>
<br>
</li>
<li>Finally, if anything else fails, a non-optimized implementation that supports all NumPy operations with arbitrary views is used.<br>
<br>
</li>
</ol>


# Data structure #

## Reusing PyArray objects ##
The idea is to reuse PyArray objects instead of creating completely new objects.
This has been implemented in [r53](https://code.google.com/p/distnumpy/source/detail?r=53) for blocked universal functions but the performance result was not that great. It is my believe that the object-creation-overhead is too little to really matter.

Some performance benchmark when running on a single CPU-core:
`mpiexec -l -n 1 -env DNPY_BLOCKSIZE 1000 python distnumpy/test/matrix_solvers.py 1 20000 1`
|          |With reuse|Without reuse|Diff.|
|:---------|:---------|:------------|:----|
|remote get|0002835 us|   0003722 us|0887 us |
|apply     |2691679 us|   2695546 us|3867 us|
|admin     |0188107 us|   0187202 us|0905 us|
|total time|2882621 us|   2886470 us|3849 us|



# Communication #
## Latency hiding ##




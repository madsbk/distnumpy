<h1> Introduction </h1>

<h2> Contiguous Distribution </h2>
<p>
A straightforward distribution approach is simply to assign a continuous sub-array to every process, starting from the upper-left corner. Then it will be very simple to convert between local and global coordinates and it will be very simple to determine, which part of a process’ local data an array-view exposes.<br>
The main drawback is load balance in problems in which the computational load is non-uniformly spread across a domain. This is not a problem in operations like the universal- or reduce-functions in NumPy but in numerical problems like Gaussian elimination, it becomes a huge problem.<br>
</p>

<h2> N-Dimensional Block Cyclic Distribution </h2>
<p>
N-Dimensional Block Cyclic Distribution is a very popular distribution scheme and it is used in libraries like <a href='http://www.netlib.org/scalapack/'>ScaLAPACK</a> and <a href='http://www.netlib.org/linpack/'>LINPACK</a>.<br>
It has a good load balance in numerical problems like Gaussian elimination, which have a diagonal workflow. Since the matrix elements are divided in a round-robin fashion, it has a good load balance even though the computation starts in the upper left corner and moves towards the bottom right corner.<br>
<br>
The distribution is illustrated in figure 1 and a more formal description can be found at <a href='http://www.netlib.org/scalapack/slug/node75.html'>her</a>.<br>
</p>

<table width='1' border='0'>
<tr>
<blockquote><td><img src='http://www.netlib.org/benchmark/hpl/mat2.jpg' /></td>
</tr>
<tr>
<td><b>Figure 1:</b> The data is distributed onto a two-dimensional 3-by-2 grid of processes according to the block-cyclic distribution. A square in the figure illustrate an n-by-n block, which consists of n matrix-elements in all dimensions. The left matrix is viewed in a global perspective and the right matrix is viewed in a local perspective.</td>
</tr>
</table></blockquote>

<h3> Memory layout inside a block </h3>
<p>
In the block-cyclic distribution, a block consists of a number of dimensions, which my or my not lie continuously in memory. That is, if A<sub>11</sub>  A<sub>14</sub> and A<sub>17</sub> in figure 1 are 4-by-4 blocks, should we then <i>stride</i> 4 elements to go from the first row in a block to the second row or do we have to <i>stride</i> 12 elements.<br>
</p>
<p>
For and against continuous memory blocks:<br>
<br>
<ul>
<li>If a block lies non-continuously in memory, it is up to the communication layer to avoid extra copying when sending the block from one process to another. This is not an issue if the block already lies continuously in memory.</li>

<li>In NumPy’s universal function (<a href='http://docs.scipy.org/doc/numpy/reference/ufuncs.html'>ufunc</a>) one block is computed at a time. First, the process that owns the output block fetches the input blocks and then computes and writes the result to the output block. This write to memory will therefore be non-continuously, if the block non-continuously in memory.  </li>
</ul>
</p>



<a href='Hidden comment: 
<h1> A more formal description 

Unknown end tag for &lt;/h1&gt;


<p>
Given a set of _M_ elements, _P_ processes, and a block size _r_, the block-cyclic distribution first divides the elements into contiguous blocks of _r_ elements each (though the last block may not be full). Then the blocks are assigned to processes in a round-robin fashion so that the _B_ th block is assigned to process number *mod*(_B_,_P_). Thus, the block-cyclic distribution maps the global index _m_ to a process, _p_, a block index, _b_, local to the process, and an item index, _i_, local to the block, with all indexes starting at 0. The mapping _m_ -> (_p_,_b_,_i_) may be written as


Unknown end tag for &lt;/p&gt;


'></a>
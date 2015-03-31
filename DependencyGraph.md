A dependency graph is a directed acyclic graph ([DAG](http://en.wikipedia.org/wiki/Directed_acyclic_graph)), in which each node represents an operation and directed edges represents dependencies. An operation can be anything between the creation and destruction of whole arrays to the calculation and communication of memory blocks. Each MPI process maintains its own dependency graph and the synchronization between processes are handled explicitly by peer-to-peer communication in the graph.

## Building the dependency graph ##
The dependency graph is build by walking through the computation and instead of doing the operation immediately the operations is added to the dependency graph. A new node is checked for conflicts and directed edges are drawn between any conflicts.

## Evaluate the dependency graph ##
The intuitive approach to evaluate the dependency graph is simply to evaluate iterative in two steps: first evaluate all nodes that have no dependencies and then remove the evaluated nodes from the graph and start over; similar to the traditional [BSP](http://en.wikipedia.org/wiki/Bulk_synchronous_parallel) model.  However, this approach may result in a deadlock as illustrated in figure 1. The correct approach is therefore to evaluate as much as possible before waiting for any communication. Since not all initiated communication may be able to finish, a mechanism such as MPI\_Waitany() or MPI\_Waitsome() must be used. Figure 2. illustrates the correct evaluation of the dependency graph from figure 1.


<table width='1' border='0'>
<tr>
<blockquote><td><img src='http://sites.google.com/site/distnumpy/wikipics/DependencyGraph_Deadlock.png' width='500' /></td>
</tr>
<tr>
<td><b>Figure 1:</b> Illustration of the naive evaluation approach, which is a traditional <a href='http://en.wikipedia.org/wiki/Bulk_synchronous_parallel'>BSP</a> model. The result is a deadlock in the first iteration since both process one and two are waiting for the receive-node to finish, but that will never happen because the matching send-node is in second iteration.</td>
</tr></blockquote>

<table width='1' border='0'>
<tr>
<blockquote><td><img src='http://sites.google.com/site/distnumpy/wikipics/DependencyGraph.png' width='500' /></td>
</tr>
<tr>
<td><b>Figure 2:</b> Illustration of the correct evaluation of the dependency graph in figure 1. The approach evaluate as much as possible before waiting for any communication.</td>
</tr></blockquote>


</table>
# Introduction #
The overall goal in DistNumPy is transparency and it is therefore essential that dispatching occur without any involvement from the user. The user is allowed to apply any valid python command on an array and if the array happens to be distributed between multiple MPI-processes, the python command will be parallelized. This means that the MPI interface should not be exposed to the user and any program-branch based on MPI-rank must be held inside our library.

# Hierarchy #
We have a flat MPI-process hierarchy with one MPI-process (master) placed above the others (slaves). All MPI-processes runs the Python interpreter but only the master runs the user-program, the others will block at the “import numpy” statement.

# Dispatch flow #
  1. The master is the dispatcher and will, when the user applies a python command on a distributed array, compose a message with meta-data describing the command.
  1. The message is then broadcasted from the master to the slaves with a blocking MPI-broadcast. It is important to note that the message only consist of meta-data and not any actual array data.
  1. After the broadcast, all MPI-processes will apply the command on the sub-array they own and exchange array elements as required (Point-to-Point communication).
  1. When the MPI-processes are done, the slaves will the next command from the master and the master will return to the user’s python program. The master will return even though some slaves may still be working on the command, synchronization is therefore required before the next command broadcast.
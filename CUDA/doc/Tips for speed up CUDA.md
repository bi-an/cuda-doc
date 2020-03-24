Tips for speed up your algorithm in the CUDA programming
Share
There are a couple of things you can do to speed up your algorithm in the CUDA programming :

1.)Try to attain .75(75%) to 1 (100%) occupancy of every kernel execution.

This can be ensured by optimizing the number of resisters used by the Kernal and number of threads per block. We need to figure out the optimum register count per thread for the target device.

2.) Avoid host to device and device to host memory transfers.Try to minimize Memory fetch operations so that the local cache need not be refreshed frequently.host to device data transfer bandwidth is 4 GB/s and divice to device data transfer bandwidth is 76.5 GB/s.So, do more computation on GPU rather then transfer data to and fro device to host.

3.) Store runtime variables in registers for fastest execution of instructions.

4.) Do not use local arrays in your code like int a[3]={1,2,3} – better use variables such as a0=1;a1=.. etc if possible. Variables and array of size 4 and less are stored in register (by deault).

5.) Write simple and small kernels. Kernel launch cost is negligible( 5 us).

If you have one large Kernel, try to split it up into multiple small ones – it might be faster due to less registers used. For small kerenels we get much resources(registers, shared memory,constant memory, etc.) because it is limited to each kernel.

6.) Texture reads are cached where as global memory reads are not. Use textures to store your data where possible.

7.)Prevent threads from diverging, Conditional jumps should branch equal for all threads.Try to give conditional branching depands on multiple of wrap size.

8.) Avoid loops which are run only by a minority of threads while the others are idle. All the threads in a block wait for every thread in that block to finish.

9.) Use fast math routines where possible.

– e.g. __mul24(), __mul32(), __sin(), etc.

10.) A complex calculation often is faster than a large lookup table so recalculation is better than caching. Remember Memory transfers can be slow.

11.) Writing your own cache manager that uses shared memory for caching might not be an advantage.

12.) For fastest global memory access, use coalescence of global memory.

13.) Try to avoid multiple threads accessing the same memory element,sothat no bank conflict is possible on shared and constant memory.

14.) Try to avoid bank conflicts for reading memories.To achive high memory bandwidth, shared and constant memories are devided into equally sized memory modules, called banks.

And each thread of half wrap access elements of different banks to avoid bank conflict.

15.) Small lookup tables can be stored in shared mem.If this small lookup table is used by all threads of the same block then small data should be cached so that it can be accessed faster.

16.) Experiment with the number of parallel threads to find the optimum. Exprementation always gives better performance improvement in CUDA.e.g. number of threads per blocks(192, 256 etc.)

17.) Using parallelism efficiently Partitioning our Computations so that our GPU multiprocessors should be equally busy.

18.) Keeping our resource usage low .Resource usages should be low enough to support multiple active thread blocks per multiprocessor.

19.) Thumb rule for Grid/Block size :

# of blocks > # of multiprocessors

-So all multiprocessors have at least one block to execute.

# of block > 2 * # of multiprocessors

-Multiple blocks can run concurrently in a multiprocessor.
 

20.) Compile with –ptxas-options=-v flag

This will give various information at the time of compilation e.g.

- Page size

- Total allocated memory

- Total available memory

- Nrof small block pages

- Nrof large block pages

- Longest free list size

- Average free list size
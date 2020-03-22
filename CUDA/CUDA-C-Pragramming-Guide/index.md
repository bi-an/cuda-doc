\[[上级目录](..)\]

参见 [CUDA C Programming Guide Reference](https://docs.nvidia.com/cuda/cuda-c-programming-guide)

### 3.1.4 应用程序兼容性（Application Compatibility）


### 3.2.4 页锁定主机内存（Page-Locked Host Memory）
<https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#page-locked-host-memory>

#### 3.2.4.1 Portable Memory

#### 3.2.4.2 Write-Combining Memory

#### 3.2.4.3 Mapped Memory


# 4 硬件实现（Hardware Implementation）

## 4.1 单指令多线程架构（SIMT Architecture）


## 4.2 硬件多线程（Hardware Multithreading）

一个多处理器的每个线程束（warp）的执行上下文（execution context，例如pragram counters, registers, etc.）保存在片上（on-chip），生命期为整个warp. 因此，执行上下文切换没有代价。每次指令发起（at every instruction issue time），线程束调度器（warp schedulers）选择所有线程都准备好下一条指令的线程束，并且把指令发给这些线程。

特别地，每个SM将32位寄存器（32-bit registers）在warps之间分区，将并行数据高速缓存（parallel data cache）和共享内存（shared memory）在线程块（thread blocks）之间分区。

SM上一个给定kernel可以驻留的线程块和线程束数量，取决于kernel使用的寄存器和共享内存的大小和SM拥有的寄存器和共享内存的容量。一个SM上能驻留的blocks和warps的最大数量也有限制，参见 [Compute Capabilities](https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#compute-capabilities) 。如果每个SM没有足够的寄存器和共享内存运行至少一个block，那么这个kernel将会启动失败。

一个block的总warps数为
```
ceil(T/warpSize, 1)
```
> * T是每个block的线程数  
> * ceil(x, y)等于x四舍五入到y的最接近倍数。


# 5 性能指导方针（Performance Guidelines）

## 5.1 整体性能优化策略（Overall Performance Optimization Strategies）

> * 最大化并行执行实现最大利用率
> * 优化内存使用实现最大内存吞吐量
> * 优化指令使用实现最大指令吞吐量



## 5.2 最大化利用率（Maximize Utilization）

为了最大程度地利用应用程序，应以尽可能多的并行性来构造应用程序，并将该并行性有效地映射到系统的各个组件，以使它们在大多数时间保持忙碌状态。

### 5.2.1 应用程序级别（Application Level）

1. 属于相同的block，所以应该使用__syncthreads()并且通过共享内存共享数据。
2. 属于不同的block，所以必须通过全局内存共享数据，并且使用两个不同的kernel，一个写，一个读。

显然第二种方案不好，它调用了两个kernel，并且引入了全局内存。

因此，应通过将算法映射到CUDA编程模型，以使需要线程间通信的计算尽可能在单个线程块内执行，从而最大程度地减少其发生。

### 5.2.2 设备级别（Device Level）

应该在一个device的SMs之间最大化并行执行。

多个kernels可以在一个device中并发执行，所以可以通过使用streams启用足够多的kernels并发执行。
见 [3.2.5. Asynchronous Concurrent Execution](https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#asynchronous-concurrent-execution)

### 5.2.3 多处理器级别（Multiprocessor Level）

应该在各种函数单元之间并行执行。

如 [4.2. 硬件多线程](#42-hardware-multithreading) 所述，GPU的SM主要依赖线程级并行来实现最大化函数单元利用率（maximize utilization of its functional units）的目的。因此，利用率直接与驻留线程束（resident warps）关联。每次指令发布，线程束调度器（warp scheduler）选择一条准备好执行的指令。这条指令可以是本warp的另一条独立指令——利用指令级并行（instruction-level parallelism），更可能的是另外一个warp的指令——利用线程级并行（thread-level parallelism）。如果一条指令被选中，那么它被发布到warp的*活动*线程。一个warp等待执行下一条指令的时钟周期数称为*延迟（latency）*。如果每个时钟周期，线程束调度器都有一些指令发布到一些warp，那么延迟就被隐藏（hidden）。隐藏`L`个时钟周期的延迟时间所需的指令数量取决于这些指令各自的吞吐量（有关各种算术指令的吞吐量，请参见 [5.4.1. 算术指令](#541-arithmetic-instruction)）。如果指令吞吐量达到最大，那么它将等于：

1. 4L：对于计算能力为5.x, 6.1, 6.2 and 7.x的设备，因为对于这些设备，一个SM每个时钟周期发送4条指令，每条指令提供给一个warp（因为有4个warp schedulers，所以每个时钟周期可以驻留4个warps）。参见 [Compute Capabilities](https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#compute-capabilities) 。
2. 2L：对于计算能力为6.0的设备，因为对于这些设备，每个时钟周期发出两条指令，每条指令提供给两个不同的warps.
3. 8L：对于计算能里为3.x的设备，因为对于这些设备，每个时钟周期发出8条指令，分成4对提供给4个不同的warps，每对属于相同的warp.

*一个warp没有准备好执行最常见的原因是，指令的输入操作数没有准备好。*

如果所有操作数都是寄存器，那么延迟是由寄存器的相关性引起的，即某些输入操作数是由一些尚未完成执行的先前指令写入的。在这种情况下，等待时间等于前一条指令的执行时间，并且warp调度器必须在这段时间内调度其他warp的指令。执行时间因指令而异。在具有7.x计算能力的设备，大多数算术指令通常为4个时钟周期。这意味着每个多处理器需要16个活动线程束（4个周期，4个warp调度器）来隐藏算术指令延迟（假设所有指令都以最大吞吐量执行，否则需要的活动线程束个数可以减少）。如果独立的warps利用指令级并行，例如，在它所在的指令流中有多个独立的指令，那么就不需要这么多warps，因为一个warp的多个独立的指令可以连续执行。

如果一些操作数在片外存储器中，那么延迟就更大，一般需要几百个时钟周期。保持warp调度器忙碌的warp数量依赖于kernel代码的指令级并行度。通常，如果不需要片外内存的指令（比如大部分时间在执行算术指令）数占比（这个比率成为程序的算术密集度（arithmetic intensity））越小，需要的warps数量就会越大。


*warp没有准备好执行下一条指令的另一个原因是，它在某个内存栅栏（Memory Fence Functions）或同步点（Memory Fence Functions）中等待。*

同步点可以迫使多处理器处于空闲状态，因为越来越多的线程束等待在同步点之前完成指令的执行。在这种情况下，每个多处理器使用多个驻留blocks可以帮助减少空闲，因为来自不同block的warp不需要在同步点彼此等待。

一次kernel调用中，每个多处理器驻留的blocks和warps数量取决于此次调用的执行配置（kernel函数调用的参数配置）、多处理器中的内存资源和这个kernel的资源需求（见 [4.2. 硬件多线程](#42-hardware-multithreading) ）。编译器可以通过设置编译选项`-ptxas-options=-v`来报告寄存器和共享内存的使用量。

一个block需要的共享内存大小等于静态和动态分配（kernel调用时的配置参数）的总量。

kernel使用的寄存器数量可能对驻留warps的数量产生显著影响。例如，对于计算能力6.x的设备，如果一个kernel使用了64个寄存器，每个block有512个线程，使用很少的共享内存，那么，2个blocks（比如32个warps，其中`32=512*2/warpSize`）可以驻留在多处理器中，因为他们需要`2*512*64`个寄存器，等于多处理器拥有的寄存器数量（`65536`）。但是，一旦kernel多使用1个寄存器，那么就只能有一个block可以驻留，因为2个blocks则需要`2*512*65`个寄存器，超出了SM的硬件能力。因此，编译器试图在保持寄存器溢出(参见 [5.3.2 设备内存访问](#532-device-memory-access) )和指令数量最小化的同时最小化寄存器的使用。寄存器使用可以使用`maxrregcount`编译选项和启动限制（ [B.23 启动限定](#b23-launch-bounds) ）来控制。

寄存器文件被组织为32-bit寄存器，所以，寄存器中存放的任何变量都需要至少一个32-bit寄存器，比如一个`double`类型变量使用2个32-bit寄存器。

应用程序也可以基于寄存器文件大小和共享内存的大小设置启动配置（kernel启动参数），取决于设备的计算能力，以及多处理器和内存带宽的设备,所有这些都可以运行时查询（见参考手册）。

应该将每个块的线程数选择为warpSize的倍数，以避免尽可能由于warps不足导致的计算资源浪费。

#### 5.2.3.1 占用率计算器

有几个API函数可以帮助程序员根据寄存器和共享内存需求选择线程块大小。

> * 占用率计算器API，`cudaOccupancyMaxActiveBlocksPerMultiprocessor()`，可以提供基于块大小和内核共享内存使用情况的占用率预测。该函数根据每个多处理器的并发线程块数量报告占用情况。
>> * 注意，该值可以转换为其他指标，乘以每个块的warps数量会得出每个多处理器的并发warps数量；进一步将并行warps数除以每个多处理器的最大warps数，得出占用百分比。
> * 基于占用率的启动配置器API，`cudaOccupancyMaxPotentialBlockSize()`和`cudaOccupancyMaxPotentialBlockSizeVariableSMem()`，启发式地（heuristically）计算可以实现最大多处理器级别（Multiprocessor-Level）占用率的执行配置。




## 5.3 最大化内存吞吐量（Maximize Memory Throughput）

### 5.3.2 设备内存访问（Device Memory Access）

## 5.4 最大化指令吞吐量

### 5.4.1 算术指令（Arithmetic Instruction）



## B.23 启动限定（Launch Bounds）

如 [5.2.3 多处理器级别](#523-multiprocessor-level) 中详细讨论的，内核使用的寄存器越少，多处理器上可能驻留的线程和线程块就越多，这可以提高性能。

因此，编译器使用启发式（hheuristics）来最大程度地减少寄存器使用量，同时保持寄存器溢出（请参见设备内存访问）和指令计数最小。应用程序可以通过使用`__global__`函数定义中的`__launch_bounds__()`限定符指定的启动范围的形式向编译器提供其他信息，从而有选择地帮助这些启发式方法：
```
__global__ void
__launch_bounds__(maxThreadsPerBlock, minBlocksPerMultiprocessor)
MyKernel(...)
{
    ...
}
```
> * `maxThreadsPerBlock`指定启动`MyKernel()`的每个block的最大线程数；它编译生成`.maxntid`PTX指令（directive）；
> * `minBlocksPerMultiprocessor`是可选参数，指定每个多处理器所需的驻留块的最小数目；它编译生成`minnctapersm`PTX指令（directive）。

如果启动限制被指定，编译器会首先限制kernel使用的寄存器数量为`L`去保证`minBlocksPerMultiprocessor`个blocks（或一个block，如果`minBlockPerMultiprocessor`没有被指定的话）能够驻留。编译器通过以下方法优化寄存器的使用:
> * 如果初始寄存器使用量超过L，那么编译器减少它直至小于等于L，通常以本地内存增加和/或更高的指令数为代价；
> * 如果初始寄存器使用量小于L，
>> * 如果`maxThreadPerBlock`被指定但是`minBlocksPerMultiprocessor`没有，那么编译器使用`maxThreadPerBlock`去决定在`n`和`n+1`个常驻块之间转换（例如在 [5.2.3 多处理器级别](#523-multiprocessor-level) 的例子中，当使用一个较少的寄存器能为一个额外的常驻块腾出空间）的寄存器使用的门限，然后对没有指定启动限制（launch bounds）的也使用相同的启动式（heuristics）。
>> * 如果`minBlocksPerMultiprocessor`和`maxThreadPerBlock`都被指定，编译器可能会尽可能增加寄存器使用量使接近L以减少指令数，这样可以更好地隐藏指令延迟。

如果kernel一个block中使用超过启动限制`maxThreadPerBlock`的线程数执行，那么kernel可能启动失败。

给定内核的最佳启动范围通常随主要架构修订版（architecture revisions）会有所不同。下面的示例代码显示了使用 [3.1.4 应用程序兼容性](#314-application-compatibility) 中引入的`__CUDA_ARCH__`宏通常如何在设备代码中处理此问题。

```
#define THREADS_PER_BLOCK          256
#if __CUDA_ARCH__ >= 200
    #define MY_KERNEL_MAX_THREADS  (2 * THREADS_PER_BLOCK)
    #define MY_KERNEL_MIN_BLOCKS   3
#else
    #define MY_KERNEL_MAX_THREADS  THREADS_PER_BLOCK
    #define MY_KERNEL_MIN_BLOCKS   2
#endif

// 设备代码
__global__ void
__launch_bounds__(MY_KERNEL_MAX_THREADS, MY_KERNEL_MIN_BLOCKS)
MyKernel(...)
{
    ...
}
```

通常，核函数`MyKernel`使用每个线程块的最大数量的线程（由`__launch_bounds()`的第一个参数指定）启动，它试图将`MY_KERNEL_MAX_THREADS`用作执行配置中每个块的线程数：

```
// 主机代码
MyKernel<<<blocksPerGrid, MY_KERNEL_MAX_THREADS>>>(...); // Does not work!
```

但是，这将不起作用，因为如 [3.1.4 应用程序兼容性](#314-application-compatibility) 中所述，`__CUDA_ARCH__`在主机代码中未定义，因此，即使__CUDA_ARCH__大于或等于200，MyKernel也会以每个块256个线程启动。相反，应该以如下方式确定每个块的线程数：
> * 方法一，在编译器使用不依赖于`__CUDA__ARCH__`的宏，比如
```
// Host code
MyKernel<<<blocksPerGrid, THREADS_PER_BLOCK>>>(...);
```

> * 方法二，在运行时基于计算能力
```
// Host code
cudaGetDeviceProperties(&deviceProp, device);
int threadsPerBlock =
          (deviceProp.major >= 2 ?
                    2 * THREADS_PER_BLOCK : THREADS_PER_BLOCK);
MyKernel<<<blocksPerGrid, threadsPerBlock>>>(...);
```

寄存器使用量可以通过编译选项`--ptxas option=-v`来报告。
驻留块的数量可以从CUDA分析器（profiler）报告的占用率中得出（有关占用率的定义，请参阅 [5.3.2 设备内存访问](#532-device-memory-access) ）。

一个文件中所有`__global__`函数的寄存器使用量可以通过`maxrregcount`编译选项来控制。对于有启动限制（launch bounds）的函数，`maxrregcount`的值被忽略。


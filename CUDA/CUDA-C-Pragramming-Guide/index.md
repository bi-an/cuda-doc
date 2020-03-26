\[[上级目录](..)\]

参见 [CUDA C Programming Guide Reference](https://docs.nvidia.com/cuda/cuda-c-programming-guide)

### 3.1.4 应用程序兼容性（Application Compatibility）


### 3.2.4 页锁定主机内存（Page-Locked Host Memory）
<https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#page-locked-host-memory>

#### 3.2.4.1 Portable Memory

#### 3.2.4.2 Write-Combining Memory

#### 3.2.4.3 映射内存（Mapped Memory）


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

1. 属于相同的block，所以应该使用`__syncthreads()`并且通过共享内存共享数据。
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

如果一些操作数在片外存储器中，那么延迟就更大，一般需要几百个时钟周期。保持warp调度器忙碌的warp数量依赖于kernel代码的指令级并行度。通常，如果不需要片外内存的指令（比如大部分时间在执行算术指令）数占比（这个比率称为程序的*算术密集度（arithmetic intensity）*）越小，需要的warps数量就会越大。


*warp没有准备好执行下一条指令的另一个原因是，它在某个内存栅栏（[Memory Fence Functions](https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#memory-fence-functions)）或同步点（[Memory Fence Functions](https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#memory-fence-functions)）中等待。*

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

```
// Device code
__global__ void MyKernel(int *d, int *a, int *b)
{
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    d[idx] = a[idx] * b[idx];
}

// Host code
int main()
{
    int numBlocks;        // Occupancy in terms of active blocks
    int blockSize = 32;

    // These variables are used to convert occupancy to warps
    int device;
    cudaDeviceProp prop;
    int activeWarps;
    int maxWarps;

    cudaGetDevice(&device);
    cudaGetDeviceProperties(&prop, device);
    
    cudaOccupancyMaxActiveBlocksPerMultiprocessor(
        &numBlocks,
        MyKernel,
        blockSize,
        0); // 得到numBlocks=32，因为最多允许驻留Blocks数量为32。
        // 这样一个SM中只有32*32=1024个线程驻留，但是SM最多允许2048个线程驻留，所以占用率为50%

    activeWarps = numBlocks * blockSize / prop.warpSize;
    maxWarps = prop.maxThreadsPerMultiProcessor / prop.warpSize;

    std::cout << "Occupancy: " << (double)activeWarps / maxWarps * 100 << "%" << std::endl;
    
    return 0;
}
```

以下代码提供了基于占用率的启动配置：
```
// Device code
__global__ void MyKernel(int *array, int arrayCount)
{
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx < arrayCount) {
        array[idx] *= array[idx];
    }
}

// Host code
int launchMyKernel(int *array, int arrayCount)
{
    int blockSize;      // The launch configurator returned block size
    int minGridSize;    // The minimum grid size needed to achieve the
                        // maximum occupancy for a full device
                        // launch
    int gridSize;       // The actual grid size needed, based on input
                        // size

    cudaOccupancyMaxPotentialBlockSize(
        &minGridSize,
        &blockSize,
        (void*)MyKernel,
        0,
        arrayCount);

    // Round up according to array size
    gridSize = (arrayCount + blockSize - 1) / blockSize;

    MyKernel<<<gridSize, blockSize>>>(array, arrayCount);
    cudaDeviceSynchronize();

    // If interested, the occupancy can be calculated with
    // cudaOccupancyMaxActiveBlocksPerMultiprocessor

    return 0;
}
```

CUDA工具包还为不依赖CUDA软件堆栈的任何用例提供了一个自文档（self-documenting）的独立占用率计算器，并在<CUDA_Toolkit_Path> /include/cuda_occupancy.h中启动了启动配置程序实现。还提供了电子表格版本的占用计算器。电子表格版本作为学习工具特别有用，它可以可视化更改影响占用率的参数（块大小，每个线程的寄存器和每个线程的共享内存）的影响。


## 5.3 最大化内存吞吐量（Maximize Memory Throughput）

第一步就是要减少低带宽的内存传输。

最小化主机-设备之间的内存传输，因为它比全局内存到设备的内存带宽还要低。

最小化全局内存于设备的内存传输，最大化设备片上内存的使用：共享内存、高速缓存（例如，L1、L2高速缓存、纹理缓存和常量缓存）。

共享内存等效于用户管理的缓存：最常见模式是将设备内存存储到共享内存，对于Block中的一个线程
> * 从设备内存加载数据到共享内存；
> * 与该块的所有其他线程同步，以便每个线程可以安全地读取由不同线程填充的共享内存位置；
> * 处理共享内存的数据；
> * 重新同步，如果必要的话，以确保共享内存用执行结果更新；
> * 将结果写回设备内存。

对于某些应用程序（例如，其全局内存访问模式与数据相关的应用程序），传统的硬件管理的缓存更适合于利用数据局部性。例如对于具有计算能力3.x和7.x的设备，L1和共享内存使用相同的片上内存，并且可以为每次内核调用（kernel call）配置L1与共享内存。

内核（kernel）对存储器的访问吞吐量可能会根据每种类型的存储器的访问模式而相差一个数量级。因此，最大化内存吞吐量的下一步是根据 [5.3.2 设备内存访问](#532-device-memory-access) 中描述的最佳内存访问模式，尽可能优化地组织内存访问。这种优化对于全局存储器访问尤为重要，因为与可用的片上带宽和算术指令吞吐量相比，全局存储器带宽较低，因此非最优的全局存储器访问通常会对性能产生重大影响。

### 5.3.1 主机与设备间的数据传输（Data Transfer between Host and Deive）
应用程序应努力减少主机与设备之间的数据传输。一种实现此目的的方法是将更多代码从主机移至设备，即使这意味着运行的内核暴露的并行性不足以在设备上高效地执行。中间数据结构可以在设备内存中创建，由设备操作并销毁，而无需主机映射或复制到主机内存。

同样，由于与每个传输相关的开销，将许多小传输批量成一个大传输总是比单独进行每个传输更好。

在具有前端总线（front-side bus）的系统上，如 [3.2.4 页锁定主机内存](#324-page-locked-host-memory) 中所述，通过使用页锁定主机内存可以实现主机与设备之间更高的数据传输性能。

此外，使用映射的页面锁定内存（[3.2.4.3 映射内存](#3243-mapped-memory)）时，无需分配任何设备内存，也无需在设备和主机内存之间显式复制数据。每次内核访问映射的内存时，都会隐式执行数据传输。为了获得最佳性能，必须将这些内存访问与对全局内存的访问合并在一起（请参阅 [设备内存访问](#532-device-memory-access)）。假设它们都是，并且映射的内存只能读取或写入一次，则使用映射的页面锁定内存而不是在设备和主机内存之间进行显式复制可以提高性能。

在设备内存和主机内存在物理上相同的集成系统上，主机和设备内存之间的任何副本都是多余的，应改用映射的页锁内存。应用程序可以通过检查集成设备属性（请参阅 [设备枚举](https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#device-enumeration) ）等于1来查询设备是否是集成（`integrated`）的。

### 5.3.2 设备内存访问（Device Memory Access）

可能访问可寻址内存（即全局，本地，共享，常量或纹理内存）的指令需要重新发出多次，具体取决于线程束中线程之间内存地址的分布。分布如何以这种方式影响指令吞吐量特定于每种类型的存储器，并将在以下各节中进行介绍。例如，对于全局存储器，通常，地址越分散，吞吐量就越降低。

**全局内存**

全局内存驻留在设备内存中，并且可以通过32、64或128字节的内存事务访问设备内存。这些内存事务必须自然对齐：内存只能读取或写入与其大小对齐的32、64或128字节的设备内存段（即，其首地址是其大小的倍数）。

当warp执行访问全局内存的指令时，它会根据每个线程访问的字的大小以及线程访问间的的内存地址分布，将warp中线程的内存访问合并为一个或多个这些内存事务。通常，需要的事务越多，除了线程访问的字之外传输的未使用的字越多，从而相应地降低了指令吞吐量。例如，如果为每个线程的4字节访问生成了32字节的内存事务，那么吞吐量将除以8。

最大化内存合并方式如下：
> * 遵循不同计算能力的最优访问模式，参见 [H.3. Compute Capabilities 3.x](https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#compute-capability-3-0) 等。
> * 使用字节对齐的数据类型。
> * 某些情况下适当填充数据，比如下面的“二维数组”部分所述访问二维数组时。

**大小和对齐要求**

全局存储器指令支持读取或写入大小等于1、2、4、8或16个字节的字。当且仅当数据类型的大小为1、2、4、8或16个字节并且自然是自然数据时，对全局内存中数据的任何访问（通过变量或指针）都会编译为单个全局内存指令对齐（即，其地址是该大小的倍数）。

如果未满足此大小和对齐要求，则访问将编译为具有交错访问模式的多个指令，从而阻止这些指令完全合并。因此，建议对驻留在全局内存中的数据使用符合此要求的类型。

[内置向量类型](https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#built-in-vector-types) 会自动满足对齐要求。

对于结构体，编译器可以使用对齐说明符`__align __(8)`或`__align __(16)`来强制执行大小和对齐要求，例如
```
struct __align__(8) {
    float x;
    float y;
};
```
或
```
struct __align__(16) {
    float x;
    float y;
    float z;
};
```
驻留在全局内存中或由驱动程序或运行时API的内存分配惯例之一返回的变量的任何地址始终至少与256个字节对齐。

读取非自然对齐的8字节或16字节字会产生错误的结果（偏移几个字），因此必须特别注意保持这些类型的任何值或值数组的起始地址的对齐。一个可能容易被忽略的典型情况是，使用某些自定义全局内存分配方案时，多个数组的分配（对`cudaMalloc()`或`cuMemAlloc()`的多次调用）被单个大内存块的分配所取代划分为多个数组，在这种情况下，每个数组的起始地址都偏离块的起始地址。

**二维数组**

一种常见的全局内存访问模式是，当索引(tx，ty)的每个线程使用以下地址访问宽度为2D的2D数组的一个元素时，该元素位于类型为`type*`的地址BaseAddress上（其中类型满足 [5.2 最大化利用率](#52-maximize-utilization) 中描述的要求）利用率）：
```
BaseAddress + width * ty + tx
```
为了使这些访问完全合并，线程块的宽度和数组的宽度都必须是warp大小的倍数。

特别地，这意味着如果宽度实际不等于该大小倍数的数组被舍入到该大小的最接近倍数并相应地填充其行，则将更有效地访问宽度不是该大小倍数的数组。参考手册中介绍的`cudaMallocPitch()`和`cuMemAllocPitch()`函数以及相关的内存复制函数使程序员能够编写与硬件无关的代码来分配符合这些约束的数组。

**本地内存**

本地存储器访问仅针对某些自动变量发生，如 [变量存储器空间说明符](https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#variable-memory-space-specifiers) 中所述。编译器可能放置在本地内存中的自动变量是：
> * 无法确定为其定序索引的数组；
> * 会占用过多的寄存器空间的大型结构体或数组；
> * 内核使用的寄存器多于可用寄存器时的任何变量（这也称为寄存器溢出）。

检查PTX汇编代码（通过使用-ptx或-keep选项进行编译获得）可得知在第一个编译阶段是否已将变量放置在本地内存中，因为将使用`.local`助记符声明该变量，并使用`ld.local`和`st.local`助记符访问该变量。 即使第一编译阶段没有使用本地内存，如果随后的编译阶段发现它们消耗了目标体系结构过多的寄存器空间，后续的编译阶段仍可能会做出其他决定：使用`cuobjdump`检查`cubin`对象将确定是否是这种情况。另外，使用`--ptxas-options = -v`选项进行编译时，编译器会报告每个kernel（`lmem`）的本地内存总使用量。请注意，某些数学函数的实现路径可能会访问本地内存。

本地内存空间位于设备内存中，因此本地内存访问具有与全局内存访问相同的高延迟和低带宽，并且要遵循与 [5.3.2 设备内存访问](#532-device-memory-access) 中所述的内存合并相同的要求。但是，对本地内存进行了组织，以便连续的线程ID访问连续的32位字。因此，只要warp中的所有线程都访问相同的相对地址（例如，数组变量中的相同索引，结构变量中的相同成员），访问就会完全合并。

在某些具有计算能力3.x的设备上，本地内存访问总是以与全局内存访问相同的方式缓存在L1和L2中（请参阅 [Compute Capability 3.x](https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#compute-capability-3-0)）。

在具有5.x和6.x计算能力的设备上，本地内存访问总是以与全局内存访问相同的方式缓存在L2中（请参阅 [Compute Capability 5.x](https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#compute-capability-5-x) 和 [Compute Capability 6.x](https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#compute-capability-6-x) ）。

**共享内存**

因为共享内存在芯片上，所以它比本地或全局内存具有更高的带宽和更低的延迟。

为了获得高带宽，共享内存被分为大小相等的存储模块，称为存储体（`banks`），可以同时访问它们。因此，可以同时处理由落在n个不同的存储体中的n个地址组成的任何存储器读或写请求，从而产生的整体带宽是单个模块的带宽的n倍。

但是，如果一个内存请求的两个地址位于同一个存储体中，则存在存储体冲突（`bank conflicts`），访问将被序列化。硬件将具有存储体冲突的内存请求拆分为尽可能多的单独的无冲突请求，从而将吞吐量降低了等于单独的内存请求数量的倍数。如果单独的内存请求的数量为n，则称初始内存请求引起n路存储体冲突（n-way bank conflicts）。

因此，为了获得最佳性能，重要的是要了解内存地址如何映射到内存库，以便调度内存请求，从而最大程度地减少库冲突。这分别在  [Compute Capability 3.x](https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#compute-capability-3-0), [Compute Capability 5.x](https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#compute-capability-5-x), [Compute Capability 6.x](https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#compute-capability-6-x), 和 [Compute Capability 7.x](https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#compute-capability-7-x) 中描述。 

**猜测：**访问同一`bank`中不同的内存位置会发生`bank conflict`，本质可能是访存指令的不同。从数字电路上看，访存可能是逻辑门（例如异或门）电路实现的，比如读取8-bit的前半部分则将前4-bit的输入端设置为0，指令为`00001111`；读取后半部分，则指令为`11110000`。如果warp中两个线程同时访存，那么就必须串行。

**常量内存**

常量内存空间驻留在设备内存中，并缓存在常量缓存中。

如果将一次请求拆分成与初次请求时不同内存地址数量一样多的单独请求，吞吐量将降低单独请求数量的倍数。

**纹理与表面内存**

纹理和表面存储空间驻留在设备内存中，并缓存在纹理缓存中，因此纹理获取或表面读取仅在一次高速缓存未命中时才耗费一次设备内存读取，否则仅花费一次纹理缓存读取。纹理缓存针对2D空间局部性进行了优化，因此相同warp的线程读取2D中紧靠在一起的纹理或表面地址将获得最佳性能。此外，它还设计用于具有恒定延迟的流式获取。高速缓存命中减少了DRAM带宽需求，但没有获取延迟。

通过纹理或表面读取来读取设备内存具有一些好处，可以使其成为从全局或常量内存中读取设备内存的有利替代方案：
> * 如果存储器读取不遵循为获得良好性能而必须遵循的全局或恒定存储器读取的访问模式，则可以提供更高的带宽，前提是纹理读取或表面读取中存在局部性；
> * 寻址计算是在内核外部由专用单元执行的；
> * 打包的数据可以在单个操作中广播到单独的变量；
> * 可以选择将8位和16位整数输入数据转换为[0.0, 1.0]或[-1.0, 1.0]范围内的32位浮点值（请参见 [3.2.11.1. Texture Memory](https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#texture-memory) ）。

## 5.4 最大化指令吞吐量

为了最大化指令吞吐量，应用程序应该：
> * 最小化低吞吐量的算术指令的使用；这包括在不影响最终结果的情况下以速度为精度的交易，例如使用内部函数（[E.2. Intrinsic Functions](https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#intrinsic-functions)）而不是常规函数，单精度而不是双精度或将非规范化的数字刷新为零；
> * 最大限度地减少由控制流指令引起的warp分流（divergence），如 [5.4.2. Control Flow Instructions](https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#control-flow-instructions) 中所述
> * 减少指令的数量，例如通过尽可能地优化同步点（如 [5.4.3. Synchronization Instruction](https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#synchronization-instruction) 中所述）或使用受限指针（如`[__restrict__](B.2.5. __restrict__)`中所述）来减少指令数量。

在本节中，吞吐量以每个多处理器每个时钟周期的操作数给出。对于32的warp大小，一条指令对应32个操作，因此，如果N是每个时钟周期的操作数，则指令吞吐量为每个时钟周期N/32指令。

所有吞吐量都是针对一个多处理器的。必须将它们乘以设备中多处理器的数量，才能获得整个设备的吞吐量。

### 5.4.1 算术指令（Arithmetic Instruction）

[表3](https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#arithmetic-instructions) 给出了各种计算功能设备的硬件固有支持的算术指令的吞吐量。

| 计算能力                                                                                                                                     | 3.0, 3.2           | 3.5, 3.7           | 5.0, 5.2           | 5.3                | 6.0                | 6.1                | 6.2                | 7.x                |
| ------------------------------------------------------------------------------------------------------------------------------------------------ | ------------------ | ------------------ | ------------------ | ------------------ | ------------------ | ------------------ | ------------------ | ------------------ |
| 16-bit floating-point add, multiply, multiply-add                                                                                                | N/A                | N/A                | N/A                | 256                | 128                | 2                  | 256                | 128                |
| 32-bit floating-point add, multiply, multiply-add                                                                                                | 192                | 192                | 128                | 128                | 64                 | 128                | 128                | 64                 |
| 64-bit floating-point add, multiply, multiply-add                                                                                                | 8                  | 642                | 4                  | 4                  | 32                 | 4                  | 4                  | 323                |
| 32-bit floating-point reciprocal, reciprocal square root, base-2 logarithm (__log2f), base 2 exponential (exp2f), sine (__sinf), cosine (__cosf) | 32                 | 32                 | 32                 | 32                 | 16                 | 32                 | 32                 | 16                 |
| 32-bit integer add, extended-precision add, subtract, extended-precision subtract                                                                | 160                | 160                | 128                | 128                | 64                 | 128                | 128                | 64                 |
| 32-bit integer multiply, multiply-add, extended-precision multiply-add                                                                           | 32                 | 32                 | Multiple instruct. | Multiple instruct. | Multiple instruct. | Multiple instruct. | Multiple instruct. | 644                |
| 24-bit integer multiply (__[u]mul24)                                                                                                             | Multiple instruct. | Multiple instruct. | Multiple instruct. | Multiple instruct. | Multiple instruct. | Multiple instruct. | Multiple instruct. | Multiple instruct. |
| 32-bit integer shift                                                                                                                             | 32                 | 645                | 64                 | 64                 | 32                 | 64                 | 64                 | 64                 |
| compare, minimum, maximum                                                                                                                        | 160                | 160                | 64                 | 64                 | 32                 | 64                 | 64                 | 64                 |
| 32-bit integer bit reverse, bit field extract/insert                                                                                             | 32                 | 32                 | 64                 | 64                 | 32                 | 64                 | 64                 | Multiple Instruct. |
| 32-bit bitwise AND, OR, XOR                                                                                                                      | 160                | 160                | 128                | 128                | 64                 | 128                | 128                | 64                 |
| count of leading zeros, most significant non-sign bit                                                                                            | 32                 | 32                 | 32                 | 32                 | 16                 | 32                 | 32                 | 16                 |
| population count                                                                                                                                 | 32                 | 32                 | 32                 | 32                 | 16                 | 32                 | 32                 | 16                 |
| warp shuffle                                                                                                                                     | 32                 | 32                 | 32                 | 32                 | 32                 | 32                 | 32                 | 326                |
| sum of absolute difference                                                                                                                       | 32                 | 32                 | 64                 | 64                 | 32                 | 64                 | 64                 | 64                 |
| SIMD video instructions vabsdiff2                                                                                                                | 160                | 160                | Multiple instruct. | Multiple instruct. | Multiple instruct. | Multiple instruct. | Multiple instruct. | Multiple instruct. |
| SIMD video instructions vabsdiff4                                                                                                                | 160                | 160                | Multiple instruct. | Multiple instruct. | Multiple instruct. | Multiple instruct. | Multiple instruct. | 64                 |
| All other SIMD video instructions                                                                                                                | 32                 | 32                 | Multiple instruct. | Multiple instruct. | Multiple instruct. | Multiple instruct. | Multiple instruct. | Multiple instruct. |
| Type conversions from 8-bit and 16-bit integer to 32-bit types                                                                                   | 128                | 128                | 32                 | 32                 | 16                 | 32                 | 32                 | 16                 |
| Type conversions from and to 64-bit types                                                                                                        | 8                  | 327                | 4                  | 4                  | 16                 | 4                  | 4                  | 168                |
| All other type conversions                                                                                                                       | 32                 | 32                 | 32                 | 32                 | 16                 | 32                 | 32                 | 16                 |

其他指令和功能是在上面的本地指令（native instructions）之上实现的。对于具有不同计算能力的设备，其实现可能会有所不同，并且编译后的本地指令数量可能会随每个编译器版本而变化。对于复杂的功能，取决于输入，可以有多个代码路径。 `cuobjdump`可用于检查`cubin`对象中的特定实现。

某些函数的实现在CUDA头文件（`math_functions.h`，`device_functions.h`，...）上很容易获得。

通常，与`-ftz = false`编译的代码相比，使用`-ftz = true`编译的代码（将非规格化的数字刷新为零）具有更高的性能。同样，使用`-prec div = false`（精度较低的除法）编译的代码往往比使用`-prec div = true`编译的代码具有更高的性能代码，而使用`-prec-sqrt = false`（精度较低的平方根）编译的代码往往比使用`-prec-sqrt = true`编译的代码具有更高的性能。 nvcc用户手册更详细地描述了这些编译标志。

**单精度浮点除法（Single-Precision Floating-Point Division）**  
`__fdividef(x, y)`（见 [E.2. Intrinsic Functions](https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#intrinsic-functions) ）比除法操作符提供了更快的单精度浮点除法。

**单精度浮点倒数平方根（Single-Precision Floating-Point Reciprocal Square Root）**  
为了保留IEEE-754语义，仅当倒数和平方根都近似时（即`-prec-div = false`和`-prec-sqrt = false`），编译器才能将`1.0/sqrtf()`优化为`rsqrtf()`。因此，建议在需要的地方直接调用`rsqrtf()`。

**单精度浮点平方根（Single-Precision Floating-Point Square Root）**  
单精度浮点平方根的实现形式是倒数平方根，然后是倒数，而不是倒数平方根，然后是乘法，这样可以得出0和无穷大的正确结果。

**正弦和余弦（Sine and Cosine）**  
sinf(x), cosf(x), tanf(x), sincosf(x)，相应的双精度指令要昂贵得多，如果自变量x的大小较大，则甚至更高。

更确切地说，自变量约简代码（请参见 [B.7. Mathematical Functions](https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#mathematical-functions) ）包括两个代码路径，分别称为快速路径和慢速路径。

快速路径用于大小足够小的自变量，并且基本上由一些乘法加法运算组成。慢速路径用于大小较大的自变量，由冗长的计算组成，以在整个自变量范围内获得正确的结果。

目前，用于三角函数的自变量归约代码为单精度函数的量值小于`105615.0f`、双精度函数的量值小于`2147483648.0`的变量选择快速路径。

由于慢速路径需要比快速路径更多的寄存器，因此尝试通过在本地存储器（Local Memory）中存储一些中间变量来减少慢速路径中的寄存器压力，这可能会由于本地存储器高延迟和高带宽而影响性能（请参阅 [5.3.2 设备内存访问](#532-device-memory-access) ）。目前，单精度功能使用28字节的本地存储器，而双精度功能使用44字节。但是，确切的数量可能会发生变化。

由于冗长的计算和在慢速路径中使用本地内存，当需要慢速路径缩减而不是快速路径缩减时，这些三角函数的吞吐量降低了一个数量级。

**整数算术（Integer Arithmetic）**  
整数除法和模运算的成本很高，因为它们最多可编译20条指令。在某些情况下，可以将它们替换为按位运算：如果`n`为2的幂，则（`i/n`）等于（`i>>log2(n)`），而（`i%n`）等于（`i&(n-1)`）；如果`n`是立即数，则编译器将执行这些转换。

`__brev `和`__popc`映射到单个指令，而`__brevll`和`__popcll`映射到一些指令。

`__[u]mul24`是遗留的内部函数，不再有任何理由要使用。

**半精度算术（Half Precision Arithmetic）**  
为了获得良好的半精度浮点加，乘或乘加吞吐量，建议使用`half2`数据类型。然后可以使用向量内在函数（例如`__hadd2`，`__ hsub2`，`__ hmul2`，`__ hfma2`）在一条指令中执行两项操作。使用`half2`代替两次使用`half`的调用也可能有助于其他内在函数的性能，例如warp shuffles。

提供了内部的`__halves2half2`可以将两个半精度值转换为`half2`数据类型。

**类型转换（Type Conversion）**  
有时，编译器必须插入转换指令，从而引入额外的执行时钟周期。这种情况为：
> * 对`char`或`short`类型的变量进行操作的函数，其操作数通常需要转换为int；
> * 双精度浮点常量（即那些没有任何类型后缀的常量）用作单精度浮点计算的输入（由C/C++标准强制执行）。

可以通过使用单精度浮点常量（用`3.141592653589793f`, `1.0f`, `0.5f`等后缀f定义）来避免后一种情况。

### 5.4.2 控制流指令（Control Flow Instructions）

任何流控制指令（`if`, `switch`, `do`, `for`, `while`）都会由于使相同warp的线程分开（即遵循不同的执行路径）来显着影响有效指令吞吐量。如果发生这种情况，则必须序列化不同的执行路径，从而增加了为此warp执行的指令总数。

为了在控制流取决于线程ID的情况下获得最佳性能，应编写控制条件，以最大程度地减少发散线程束（divergent warps）的数量。这是可能的，因为如 [4.1. SIMT Architecture](https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#simt-architecture) 中所述，整个block上的warp分布是确定的。一个简单的例子是，控制条件仅取决于（`threadIdx / warpSize`），其中warpSize是warp大小。在这种情况下，由于控制条件与warps完全对准，因此没有warp发散。

有时，编译器可能会展开循环，或者可能会改用分支谓词（branch predication）来优化简短的 `if`或`switch`块，如下所述。在这些情况下，任何warps都不会发散。程序员还可以使用`#pragma unroll`指令控制循环的展开（请参阅 [B.24. #pragma unroll](https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#pragma-unroll) ）。

当使用分支谓词时，其执行取决于控制条件的指令均不会被跳过。相反，它们中的每一个都与基于控制条件设置为true或false的每个线程的条件代码或谓词相关联，并且尽管调度了这些指令中的每条指令以执行，但实际上仅执行具有真实谓词（true predication）的指令。带有错误谓词（false predication）的指令不会写入结果，也不会求地址或读取操作数。

### 5.4.3 同步指令（Synchronization Instruction）
`__syncthreads()`的吞吐量对于计算能力为3.x的设备而言为每个时钟周期128次操作，对于计算能力为6.0的设备而言为每个时钟周期32次操作，对于计算能力7.x的设备而言为每个时钟周期16次操作，对于计算能力为5.x，6.1和6.2的设备而言为每个时钟周期64次操作。

请注意，`__syncthreads()`可能会通过强制多处理器进入空闲状态来影响性能，如 [5.3.2 设备内存访问](#532-device-memory-access) 中所述。




## B.23 启动限定（Launch Bounds）

如 [5.2.3 多处理器级别](#523-multiprocessor-level) 中详细讨论的，内核使用的寄存器越少，多处理器上可能驻留的线程和线程块就越多，这可以提高性能。

因此，编译器使用启发式（hheuristics）来最大程度地减少寄存器使用量，同时保持寄存器溢出（请参见 [5.3.2 设备内存访问](#532-device-memory-access) ）和指令计数最小。应用程序可以通过使用`__global__`函数定义中的`__launch_bounds__()`限定符指定的启动范围的形式向编译器提供其他信息，从而有选择地帮助这些启发式方法：
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



# [线程束（Warps）](https://en.wikipedia.org/wiki/Thread_block_(CUDA_programming))

在硬件方面，线程块由“warp”组成。warp是一个线程块中32个线程的集合，这样warp中的所有线程都执行相同的指令。这些线程是由SM串行选择的。

一旦一个线程块发起在一个SM上，所有的线程束将一直驻留直到执行完毕。因此，一个新的线程块不会被启动，直到SM上有足够的空闲寄存器和共享内存。

假设一个有32个线程的线程束执行一条指令，如果一个或多个线程的操作数没有就绪（比如还没有从全局内存上取回），这时候会发生“上下文切换”，从一个warp切换到另外一个所有数据都在寄存器文件中（操作数就绪，所以可以立即执行）的warp。当一条指令没有显著的数据依赖时，也就是说，它的操作数都准备好了，相应的warp就可以执行了。如果有多个warp有资格执行，则父SM使用warp调度策略来决定哪个warp获得下一个获取的指令。

**线程束调度策略：**

1. 轮循（Round Robin）：指令管理器以轮循方式取得指令，SMs保持忙碌并且没有时钟周期浪费在访存延迟上。

2. 最近最少获取（LRF，Least Recently Fetch）：最长时间没有获取指令的线程束将拥有更高的优先级。

3. 公平（Fair）：优先将指令发给当前指令数较少的线程束，保证每个线程束执行的指令数公平。

4. 基于线程块的临界区唤醒线程束调度（CAWS，Criticality Aware Warp Scheduling）：旨在提高线程块的执行时间。给予需要更多执行时间的线程束以更多的时间资源。通过基于最关键（临界）线程束更高优先级，可以提高线程块更快执行完毕，从而资源可以更快获得。

为了利用warp架构，编程语言和开发人员需要了解如何合并内存访问以及如何管理控制流分歧。如果warp中的每个线程采用不同的执行路径，或者每个线程访问的内存明显不同，那么warp架构的好处就会丧失，性能也会显著下降。
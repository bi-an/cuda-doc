# 异步
由于CPU与GPU的异步执行性，可以实现异步函数。GPU执行计算密集型任务时，CPU将返回，例如去处理网络IO等。

# warp
warp是SM级别的，单个线程是SP级别的，SP每次运行一个线程。
每个SM的core数量决定其一个始终周期内可以运行的warp指令。
单个warp指令可以在任何给定的时钟周期内处理，但需要32个core才能完成（并且可能需要多个时钟周期才能完成，具体取决于指令）。
SM将block“解包”为warp，并在SM内部资源（例如core和SFU）可用时调度warp指令。

# 架构
每个SM都有L1 cache，所有的SM共享L2 cache；
L1 cache：用于缓存local memory，包括临时寄存器溢出。
L2 cache：用于缓存global memory。
缓存行为（例如，读操作同时使用L1和L2 cache，还是仅仅使用L2 cache）可以使用load或者store指令修饰符来部分配置，3.5以上计算能力设备允许通过编译选项配置。
>对于计算能力3.x的设备
>>on-chip memory（unified data cache）可以在shared memory和L1 cache之间分配（使用cudaFuncSetCacheConfig()/cuFuncCacheConfig()）：
>>>（1）48 KB作为shared memory，16 KB作为L1 cache；
>>>（2）32 KB作为shared memory，32 KB作为L1 cache。
>对于计算能力5.x的设备
>>on-chip memory（unified data cache）可以在shared memory和L1 cache之间分配（使用cudaFuncSetCacheConfig()/cuFuncCacheConfig()）：
>>>（1）48 KB作为shared memory，16 KB作为L1 cache；
>>>（2）32 KB作为shared memory，32 KB作为L1 cache。

# 设备内存访问
内存分布对指令吞吐量的影响：根据一个warp内的线程和内存地址的分布，可能会出现内存访问指令被发起（re-issue）多次的情况。例如，全局内存地址越分散，指令吞吐量就越小。
指令吞吐量：执行一条指令所需的时钟数。
## 全局内存（global memory / device memory）
内存事务（memory transactions）：内存事务指一次内存访问。可以以32字节、64字节、128字节访问。
合并访问（coalesces the memory accesses）：一个warp内的线程的内存访问会被合并。通常，内存事物越多，线程束的指令利用率越低，指令吞吐量越小。例如，32字节的内存，如果每个线程分配成4字节访问，那么指令吞吐量会降低为1/8倍。


*1个warp包含32线程，全局内存的缓存行（cache line）是128字节。*
如果内存事务为128字节，那么L1和L2缓存都会被用到；
如果内存事务为32字节，那么只有L2缓存会被用到；
L2缓存只能降低访存次数（over-fetch），例如，对地址分散的内存的访问。
如果每个线程对全局内存的访问超过4个字节，那么1个warp的内存访问请求会被分割成128字节的内存访问：
32 * 8 / 128 = 2次内存请求，每次请求为1/2个warp，如果每个线程请求8字节；
32 * 16 / 128 = 4次内存请求，每次请求为1/4个warp，如果每个线程请求16字节。

对于整个kernel生命周期内的只读数据，可以使用__ldg()函数（可能是load global的缩写）缓存在read-only data cache中。编译器也会使用__ldg()进行这方面的尝试，当使用const和__restrict__限定符标记这部分数据指针时，会增加编译器探测read-only条件（进而进行上述缓存优化）的概率。

*内存对齐会增加cache概率，减少访存（memory transaction）次数*

## 共享内存（shared memory）
bank conflict：存储体/存储池冲突

## 不同计算能力（Compute Capabilities）比较
||3.x|5.x|6.x|7.x|
||---|---|---|---|
|L1 cache|L1/shared memory|L1/texture|
|Shared Memory|(Max) 48 KB|64 KB|
|Texture Memory|-||
|L2 cache|
|Constant cache|
|CUDA cores / MP|192|128|64 (6.0), 128 (6.1, 6.2)|
|Special Function Units|32|32|
|warp Schedulers|4|4|


# D. 执行环境与内存模型
## D.2. 执行环境
### D.2.1. 父子线程格（Parent and child grids）
Parent grid中的线程可以创建grid，称为线程格嵌套（grids nesting）
嵌套grids会隐式同步。
### D.2.2. CUDA原语作用域
#### 

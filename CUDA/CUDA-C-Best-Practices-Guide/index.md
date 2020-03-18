参见[CUDA C Best Paratices](https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/index.html)


# 9. 内存优化

## 9.1 主机与设备间数据传输

### 9.1.1 固定内存（Pinned Memory）

设备内存和GPU之间的峰值理论带宽要远远高于主机内存和设备内存之间的峰值理论带宽(例如，NVIDIA Tesla V100的峰值理论带宽为898 GB/s)(例如，PCIe x16 Gen3的峰值理论带宽为16 GB/s)。因此，为了获得最佳的整体应用程序性能，最小化主机和设备之间的数据传输是很重要的，即使这意味着在GPU上运行内核与在主机CPU上运行内核相比没有任何加速。

中间数据结构应该在设备内存中创建，由设备操作，并且在没有被主机映射或复制到主机内存的情况下销毁。

同样，由于与每次传输相关的开销，将许多小的传输批量成一个较大的传输比单独进行每个传输要好得多，即使这样做需要将内存的非连续区域打包到连续的缓冲区中，然后在传输之后解包 。

最后，当使用页面锁定(或固定)内存时，主机和设备之间可以实现更高的带宽，如CUDA c++编程指南和本文档的固定内存部分所述。

页面锁定或固定的内存传输可以在主机和设备之间实现最高的带宽。例如，在PCIe x16 Gen3卡上，固定内存可以达到大约12 GB/s的传输速率。

使用运行时API中的`cudaHostAlloc()`函数分配固定内存。bandwidthTest CUDA示例展示了如何使用这些函数以及如何测量内存传输性能。

对于已经预先分配的系统内存区域，可以使用`cudaHostRegister()`动态地固定内存，而不需要分配单独的缓冲区并将数据复制到其中。

固定内存不应该被过度使用。过度使用会降低整个系统的性能，因为固定内存是一种稀缺的资源，但是多少内存是太多很难提前知道。此外，与大多数正常的系统内存分配相比，系统内存的固定是一个重量级操作，因此，对于所有优化，都要测试应用程序和它所运行的系统，以获得最佳性能参数。

### 9.1.2 计算与内存传输的异步和重叠
使用cudaMemcpy()在主机和设备之间进行的数据传输阻止了传输。 也就是说，仅在数据传输完成后，控制权才返回给主机线程。 cudaMemcpyAsync()函数是cudaMemcpy()的非阻塞变体，在该变体中，控制权立即返回给主机线程。 与cudaMemcpy()相比，异步传输版本需要固定的主机内存（请参见固定的内存），并且它包含一个附加参数，即流ID。 流只是在设备上按顺序执行的一系列操作。 不同流中的操作可以交错，在某些情况下可以重叠-该属性可用于隐藏主机与设备之间的数据传输。

异步传输以两种不同的方式使数据传输与计算重叠。在所有支持cuda的设备上，可以通过异步数据传输和设备计算来重叠主机计算。例如，以下代码展示了在将数据传输到设备和执行使用设备的内核时，如何执行例程cpuFunction()中的主机计算。

```
// 重叠计算和数据传输
cudaMemcpyAsync(a_d, a_h, size, cudaMemcpyHostToDevice, 0);
kernel<<<grid, block>>>(a_d);
cpuFunction();
```

cudaMemcpyAsync()函数的最后一个参数是流ID，在本例中它使用默认的流，流0。
kernel也使用默认的流，直到内存拷贝完成才会开始执行;因此，不需要显式同步。
因为内存副本和内核都立即将控制权返回给主机，所以主机函数cpuFunction()会与它们的执行重叠。

上面代码中，内存复制和内核执行是顺序进行的。在能够并发复制和计算的设备上，可以通过主机和设备之间的数据传输来重叠设备上的内核执行。设备是否具有此功能由cudaDeviceProp结构的asyncEngineCount字段表示(或在设备设备CUDA示例的输出中列出)。在具有此功能的设备上，重叠再次需要固定的主机内存，此外，数据传输和内核必须使用不同的非默认流(具有非零流id的流)。这种重叠需要非默认流，因为使用默认流的内存复制、内存集函数和内核调用只有在设备上的所有前面调用(在任何流中)完成之后才会开始，而且在设备上的任何操作(在任何流中)在完成之前都不会开始。

```
// 并发复制和执行
cudaStreamCreate(&stream1);
cudaStreamCreate(&stream2);
cudaMemcpyAsync(a_d, a_h, size, cudaMemcpyHostToDevice, stream1);
kernel<<<grid, block, 0, stream2>>>(otherData_d);
```

在这段代码中，根据cudaMemcpyAsync调用的最后一个参数和内核的执行配置，创建并在数据传输和内核执行中使用两个流。

```
// 分阶段复制和执行（Staged concurrent copy and execute）
size=N*sizeof(float)/nStreams;
for (i=0; i<nStreams; i++) {
    offset = i*N/nStreams;
    cudaMemcpyAsync(a_d+offset, a_h+offset, size, dir, stream[i]);
    kernel<<<N/(nThreads*nStreams), nThreads, 0,  
             stream[i]>>>(a_d+offset);
}
```

这段代码假设N可以被nThreads * nStreams整除。因为流中的执行是顺序进行的，所以在它们各自的流中的数据传输完成之前，没有一个内核会启动。当前的gpu可以同时处理异步数据传输和执行内核。

*图9-1 内存复制和kernel执行的时间线比较*
<div align=center>![Image text](images/9-1.png)</div>
>上方：顺序（Sequential）  
>下方：并发（Concurrent）


### 9.1.3 零拷贝内存（Zero Copy）
零拷贝是CUDA Toolkit v2.2及以上版本中增加的一个功能。它允许GPU线程直接访问主机内存。为此，它需要映射固定(非分页)内存。在集成的gpu上(即当CUDA设备属性结构的集成字段被设置为1)时，映射固定内存总是一个性能增益，因为它避免了多余的拷贝，因为集成的GPU和CPU内存在物理上是相同的。在独立的gpu上，映射固定内存只有在某些情况下才有优势。由于数据不是缓存在GPU上，映射的固定内存应该只被读写一次，读写内存的全局加载和存储应该被合并。零拷贝可以用来代替流，因为源于内核的数据传输会自动重叠内核执行，而无需设置和确定流的最佳数量。

```
// 零拷贝内存主机代码
float *a_h, *a_map;
...
cudaGetDeviceProperties(&prop, 0);
if (!prop.canMapHostMemory) 
    exit(0);
cudaSetDeviceFlags(cudaDeviceMapHost);
cudaHostAlloc(&a_h, nBytes, cudaHostAllocMapped);
cudaHostGetDevicePointer(&a_map, a_h, 0);
kernel<<<gridSize, blockSize>>>(a_map);
```

在这段代码中，`cudaGetDeviceProperties()`返回的结构的`canMapHostMemory`字段用于检查设备是否支持将主机内存映射到设备的地址空间。通过使用`cudaDeviceMapHost`调用`cudaSetDeviceFlags()`来启用页面锁定内存映射。注意，`cudaSetDeviceFlags()`必须在设置设备或进行需要状态的CUDA调用之前调用(本质上是在创建上下文之前)。使用`cudaHostAlloc()`分配页面锁定的映射主机内存，并通过函数`cudaHostGetDevicePointer()`获得指向映射设备地址空间的指针。在零拷贝主机代码中的代码中，kernel()可以使用指针`a_map`引用被映射的固定主机内存，这与a_map引用设备内存中的位置是完全一样的。

注意:映射固定主机内存允许CPU-GPU的内存传输与计算重叠，同时避免使用CUDA流。但是，由于对这些内存区域的任何重复访问都会导致CPU-GPU之间的重复传输，因此可以考虑在设备内存中创建第二个区域来手动缓存之前读取的主机内存数据。

## 9.2 设备内存空间

*图9-2 设备内存空间*
<div align=center>![Image text](images/9-2.png)</div>


*表9-1 设备内存的显著特征*

| Memory   | Location on/off chip | Cached | Access | Scope                | Lifetime        |
| -------- | -------------------- | ------ | ------ | -------------------- | --------------- |
| Register | On                   | n/a    | R/W    | 1 thread             | Thread          |
| Local    | Off                  | Yes†† | R/W    | 1 thread             | Thread          |
| Shared   | On                   | n/a    | R/W    | All threads in block | Block           |
| Global   | Off                  | †    | R/W    | All threads + host   | Host allocation |
| Constant | Off                  | Yes    | R      | All threads + host   | Host allocation |
| Texture  | Off                  | Yes    | R      | All threads + host   | Host allocation |
|          |                      |        |        |                      |                 |

>† 在计算能力6.0和7.x的设备上默认缓存在L1和L2中；默认情况下，在计算能力较低的设备上，只在L2中缓存，但有些设备允许通过可选的编译标志在L1中进行缓存。

>†† 默认缓存在L1和L2中，但计算能力为5.x的设备除外；5.x计算能力仅在L2中缓存局部变量。

1. SIMT
单指令多线程：分支分化（或说分流，branch-divergence）问题。

2. 多核心
核心个数：128（计算能力5.x）、64 (compute capablity 6.0) or 128 (6.1 and 6.2) 
SM个数：5个（Geforce 960m：计算能力5.0）、56个（Tesla P100：计算能力6.0），都是每个SM有64个cores.
SM个数与同时驻留block个数有关。
warps
warp调度器：4个（计算能力5.x、6.1、6.2），2个（计算能力6.0）
以Block为单位运行。

占用率：活动线程占用率（SM占用率）。这个与Block的size和内存资源有关。
占用率与访存延迟息息相关。
首先要知道一个SM能驻留多少线程（或说warps），一般是2048线程，也就是64 warps.
然后分配寄存器（64 K）。

3. 内存结构
共享内存与高速缓存差不多。
全局内存需要几百个时钟周期。
寄存器个数限制：与驻留线程束有关，

4. 访存延迟
占用率
全局内存的高速缓存：24 KB，

5. 驻留
grids：32（计算能力5.0）、128（计算能力6.0)。
猜测：驻留的grids数应该是指默认流中能够放到SM中的kernels数，应该不存在上下文切换。
blocks：32
驻留的blocks之间的存在上下文切换（warps切换伴随着上下文切换），上下文直接保存在寄存器中。
warps：64（threads：2048）

6. 浮点计算
精度：FMA（Floating Multiply-Add）
能力：吞吐量

7. 加速
访存：全局内存访存次数、cache、内存合并（例如对齐（cudaMallocPitch、cuMemAllocPitch()），一个warp访问连续地址）。
占用率
函数选择（指定精度的函数、减少隐式类型转换（例如带f后缀的浮点参数））
驻留grids数、驻留block数、驻留warps数
小内核


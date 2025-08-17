# cudaErrorLaunchFailure
> * 如果这是显示卡的话，显示驱动程序有一个看门狗定时器，它会杀死需要几秒钟以上才能完成的内核（kernel）。
> * 可以使用`cuda-memcheck`查看错误
> * [解决方案](https://stackoverflow.com/questions/6913206/cuda-limit-seems-to-be-reached-but-what-limit-is-that)：
The resource which is being exhausted is time. On all current CUDA platforms, the display driver includes a watchdog timer which will kill any kernel which takes more than a few seconds to execute. Running code on a card which is running a display is subject to this limit.

On the WDDM Windows platforms you are using, there are three possible solutions/work-arounds:

Get a Telsa card and use the TCC driver, which eliminates the problem completely
Try modifying registry settings to increase the timer limit (google for the TdrDelay registry key for more information, but I am not a Windows user and can't be more specific than that)
Modify your kernel code to be "re-entrant" and process the data parallel work load in several kernel launches rather than one. Kernel launch overhead isn't all that large and processing the workload over several kernel runs is often pretty easy to achieve, depending on the algorithm you are using.

耗尽的资源是时间。在所有的现有CUDA平台上，显示驱动器有个看门口定时器，它会杀死所有需要几秒钟以上的内核。在正在运行显示卡上运行的代码受此限制。

在你使用WDDM（Windows Display Driver Model）模式时，有几个解决办法如下：  
（1）尝试修改注册表设置，增加时间限制（google搜索“TdrDelay registry key”获取更多信息）；  
（2）将您的内核代码“重入”，并在多个内核启动(而不是一个)中处理数据并行工作负载。内核启动开销并不大，而且根据使用的算法，在多个内核运行期间处理工作负载通常非常容易实现。  
注释：TCC（Tesla Compute Cluster）.


[WDDM切换到TCC](https://superuser.com/questions/1173035/how-to-configure-gpu-to-work-in-tcc-mode-on-windows-10)：

建议使用NVidia NSight Monitor。以管理员身份打开后，您可以在属性部分中找到两个有趣的设置：a) WDDM，默认情况下设置为true；  b) WDDM超时（以毫秒为单位）。

将WDDM设置为false将禁用该Windows Watchdog，并使您能够随意使用自己的GPU。但请注意：您的屏幕可能会冻结。不用担心，一旦完成计算，它将解冻。

如果希望避免屏幕冻结，可以通过将该值设置为更高的值来增加WDDM超时。当然，这需要在时间范围内进行代码优化。

您可以做的另一件事是右键单击桌面，然后打开NVidia控制面板。在那里，在3d设置部分，您将具有“配置SLI，Surround，PhysX”设置。如果打开它，您将看到PhysX处理器被设置为（默认情况下）自动选择。当然，操作系统会选择GPU来渲染您的显示。将其设置为CPU。现在，您的显示将由CPU处理。

最后一件事，GeForce卡不支持TCC，仅支持WDDM。我相信Quadro两者都支持，因此，如果需要的话，可能会有一种将其设置为TCC的方法，但是它将毫无用处。.您拥有的Quadro版本无法在该模式下进行设置。您可以从以下[链接](https://devtalk.nvidia.com/default/topic/513659/can-tcc-mode-be-enabled-with-quadro-2000m-4000m/)中了解更多信息。

另外，使用TechPowerUp GPU-Z进行实时GPU资源监控。


[注意](https://forums.developer.nvidia.com/t/computation-mode-only/81421)：如果是笔记本电脑，则无法将dGPU（Quadro P4000）置于TCC模式。大多数带有dGPU的现代笔记本电脑都具有“擎天柱”设计的某些变体。这种优化设计意味着两个GPU（英特尔iGPU和NVIDIA dGPU）并不是真正完全独立的。它们旨在协同工作，因此dGPU不能单独切换到TCC模式。
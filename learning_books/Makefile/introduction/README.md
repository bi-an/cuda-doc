# 1. Makefiles简介

> 原文链接：[第2章 Makefiles简介](https://www.gnu.org/software/make/manual/html_node/Introduction.html)

通常，makefile文件告诉make怎么编译和链接一个程序。

在本章中，我们将讨论一个简单的makefile，它描述了怎么编译和链接一个包含8个C文件和3个头文件的文本编辑器的程序。

make重新编译程序（program）时，每一个改动的C源文件都必须被重新编译。
如果某个文件被改动，每个包含该头文件的C源文件都必须被重新编译。
编译以C源文件为单位，每个C源文件都会相应地产生一个目标文件（object file）。
如果任何一个C源文件被重新编译了，所有的目标文件，不论是新编译的还是上一次编译后保存好的，都会必须被重新链接，以产生新的可执行程序(executable）。

## 1.1. Makefile规则

一个简单的makefile包含以下形式的“规则（rules）”：

```makefile
target : prerequisites
	recipe
	...
	...
```

target通常是一个程序生成的文件名，比如可执行或目标文件（executable or object files）。

target也可以是要执行的动作（action）的名称，比如clean
（见[伪目标](https://www.gnu.org/software/make/manual/html_node/Phony-Targets.html)）。

prerequisites是用来作为输入以生成target的文件。一个target通常包含数个文件，也可以不包含任何prerequisites。

recipe是一个make将要执行的动作（action）。一个recipe可能包好不止一个命令（command），不论是写在同一行还是各自一行。
**请注意：** 你需要在每一个recipe行的开头放置一个tab符！如果你更喜欢其他符号而不是tab，你可以设置.RECIPEPREFIX变量为一个替代字符
（见[其他特殊字符](https://www.gnu.org/software/make/manual/html_node/Special-Variables.html)）。

**译者注：** 每个recipe行创建一个bash来执行命令，这意味着上一行recipe设置的环境变量在其他recipe行是无效的。测试代码如下：

```makefile
all:
	echo PID:$$$$
	echo PID:$$$$
```

语法解释：Makefile中`$$`表示转义`$`符号，所以最终转递给shell的命令是：

```bash
echo PID:$$
```

而bash语法中`$$`表示当前进程的PID。

测试结果：

```text
echo PID:$$
PID:85280
echo PID:$$
PID:85281
```

- Whitespace

Makefile对空格的处理似乎是：从第一个非空格字符开始，到明确的截止符（比如换行、'#'注释标记、逗号、括号等）为止。测试如下：

```makefile
a =   b   #注意末尾有3个空格
$(warning a=$(a)c)

find = $(findstring a  , a  b ) # 注意，2个'a'之后都有2个空格，
$(warning find=$(find)$$)

find1 = $(findstring a  , a b ) #注意：第一个'a'之后有2个空格，第二个'a'之后只有1个空格。
$(warning find1=$(find1)$$)
```

结果：

```text
Makefile:2: a=b   c
Makefile:5: find=a   $
Makefile:8: find1= $
```

- 多个makefile脚本同时存在

按照GNUmakefile、makefile、Makefile顺序查找，找到为止。比如GNUmakefile和Makefile同时存在，那么找到了GNUmakefile就停止，不会执行Makefile脚本。

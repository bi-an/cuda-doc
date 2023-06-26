# Whitespace

Makefile对空格的处理似乎是：从第一个非空格字符开始，到明确的截止符（比如换行、'#'注释标记、逗号、括号等）为止。测试如下：

```makefile
a =   b   #注意末尾有3个空格
$(warning a=$(a)c)
```

结果：

```text
a=    b   c
```

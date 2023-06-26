# 简介

计算机技术相关的读书笔记。使用Markdown语法。

# Markdown编辑器

推荐的[markdown编辑器](https://www.zhihu.com/tardis/zm/art/103348449?source_id=1003)：
- VSCode：免费。VSCode原生支持Markdown，安装一些插件可以帮助更快地编写markdown文件。
- Typora：现在已经开始收费。

VSCode markdown插件：
- Mardown All in One: 提供快捷键，帮助更快的编写markdown文件。
- Markdown+Math：提供数学公式支持。
- Markdown Preview Enhanced: 将原生markdown预览的黑色背景改成白色。
- Markdown Preview Github Styling：提供Github风格的预览。

[在线表格生成器](https://www.tablesgenerator.com/markdown_tables)：可以生成Markdown、Text、HTML、LaTex、MediaWiki格式的表格。

# 轻量级虚拟机WSL

WSL，[Windows Subsystem for Linux](https://learn.microsoft.com/en-us/windows/wsl/install)，是Windows提供的轻量级Linux虚拟机。

安装教程：见[链接](https://zhuanlan.zhihu.com/p/170210673)。

## WSL默认没有启用systemctl：

启用systemctl的方法：[链接](https://askubuntu.com/questions/1379425/system-has-not-been-booted-with-systemd-as-init-system-pid-1-cant-operate)。

替代方法：不需要启动systemctl，因为会比较占用资源，启动也会变慢。可以使用service命令替代。

## WSL默认没有安装openssl-server：

使用ssh连接到服务器时，需要服务器运行着sshd程序，否则连接不上，会出现"[Connection refused](https://www.makeuseof.com/fix-ssh-connection-refused-error-linux/)"错误。

参考[链接](https://askubuntu.com/questions/1339980/enable-ssh-in-wsl-system)。

查看openssh-server有没有安装：
```bash
dpkg --list | grep ssh
```

注：如果安装了openssh-server，执行which sshd可以看到路径。

WSL默认没有安装openssh-server，安装方法：
```bash
sudo apt-get install openssh-server
```

启动ssh：
```bash
sudo service ssh start
```

## 通过https登录到github

`git push`不再支持输入用户名和密码，当提示输入密码时，需要输入personal access token.

步骤1：在github上[创建personal access token](https://docs.github.com/en/authentication/keeping-your-account-and-data-secure/managing-your-personal-access-tokens#creating-a-personal-access-token-classic)；

步骤2：[在命令行上使用personal access token](https://docs.github.com/en/authentication/keeping-your-account-and-data-secure/managing-your-personal-access-tokens#using-a-personal-access-token-on-the-command-line)；

步骤3：为了避免每次都需要输入personal access token，可以将其[缓存在git client上](https://docs.github.com/en/get-started/getting-started-with-git/caching-your-github-credentials-in-git)：

```bash
gh auth login
```

注：使用`gh`命令需要先安装GitHub CLI：

```bash
sudo apt-get install gh
```

# 配置主页

## mkdocs

[mkdocs](https://www.mkdocs.org/)是一个快速的静态网页生成器。
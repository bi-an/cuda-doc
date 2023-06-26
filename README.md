# 1. Github Pages

这是使用[github pages](https://docs.github.com/en/pages)搭建的个人主页。

github pages简介：[官方链接][4]。

github pages使用了[CNAME record](https://en.wikipedia.org/wiki/CNAME_record)技术，参考：[链接1][1]、[链接2][2]、[Custom domains in Github Pages][3]。

## Github Pages 站点类型

有3种类型的 Github Pages 站点（sites）：project, user 和 organization 。

Project sites 连接到 github 上特定 project ，比如 Javascript library 或 recipe collection。user 或 organization sites 连接到 github.com 的特定账户。

发布 user site ，你必须创建一个你的个人账户下的一个名为 `<username>.github.io` 的 repository 。发布 organization site ，你必须创建一个组织所有的名为 `<organization>.github.io` 的 repository 。除非你使用 custom domain ，否则 user 和 organization sites 将位于 `http(s)://<username>.github.io` 或 `http(s)://<organization>.github.io` 。

project site 的源文件存储在作为 project 的相同的 repository 中。除非使用 custom domain ， 否则 project sites 将位于 `http(s)://<username>.github.io/<repository>` 或 `http(s)://<organization>.github.io/<repository>` 。

有关如何自定义影响您网站的域名的更多信息，参见"[About custom domains and GitHub Pages](https://docs.github.com/en/pages/configuring-a-custom-domain-for-your-github-pages-site/about-custom-domains-and-github-pages)"。

每个 github 账户允许创建 1 个 user 或 organization 站点。无论是被组织还是个人所有，project 站点的个数不限制。


# 2. 配置前准备

## 2.1. Markdown编辑器

推荐的[markdown编辑器](https://www.zhihu.com/tardis/zm/art/103348449?source_id=1003)：
- VSCode：免费。VSCode原生支持Markdown，安装一些插件可以帮助更快地编写markdown文件。
- Typora：现在已经开始收费。

VSCode markdown插件：
- Mardown All in One: 提供快捷键，帮助更快的编写markdown文件。
- Markdown+Math：提供数学公式支持。
- Markdown Preview Enhanced: 将原生markdown预览的黑色背景改成白色。
- Markdown Preview Github Styling：提供Github风格的预览。

[在线表格生成器](https://www.tablesgenerator.com/markdown_tables)：可以生成Markdown、Text、HTML、LaTex、MediaWiki格式的表格。


## 2.2. 轻量级虚拟机WSL

WSL，[Windows Subsystem for Linux](https://learn.microsoft.com/en-us/windows/wsl/install)，是Windows提供的轻量级Linux虚拟机。

安装教程：见[链接](https://zhuanlan.zhihu.com/p/170210673)。

### 2.2.1. WSL默认没有启用systemctl：

启用systemctl的方法：[链接](https://askubuntu.com/questions/1379425/system-has-not-been-booted-with-systemd-as-init-system-pid-1-cant-operate)。

替代方法：不需要启动systemctl，因为会比较占用资源，启动也会变慢。可以使用service命令替代。

### 2.2.2. WSL默认没有安装openssl-server：

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

### 2.2.3. 通过https登录到github

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


# 3. 静态站点生成器

以下几种[静态站点生成器][4]都可以用来搭建个人主页。如果使用除JekyII外的工具，则需要配置Github Actions以构建和发布你的站点。

## 3.1. mkdocs

[mkdocs](https://www.mkdocs.org/)是一个快速的静态网页生成器。

## 3.2. JekyII

Jekyll 是一个静态站点生成器，内置对 GitHub Pages 的支持和简化的构建进程。

参见[About GitHub Pages and Jekyll](https://docs.github.com/en/pages/setting-up-a-github-pages-site-with-jekyll/about-github-pages-and-jekyll)。


[1]: https://www.zhihu.com/question/39301250
[2]: https://www.zhihu.com/question/26609475
[3]: https://docs.github.com/en/pages/configuring-a-custom-domain-for-your-github-pages-site/about-custom-domains-and-github-pages#using-a-subdomain-for-your-github-pages-site
[4]: https://docs.github.com/en/pages/getting-started-with-github-pages/about-github-pages

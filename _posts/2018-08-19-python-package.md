---
layout: post
date: 2018-08-19 23:35:15.000000000 +09:00
title: 多级目录的python 包、模块的引用问题
categories: python
tags: axis
---

如果发现系统报错：找不到模块

那是因为自建模块没有加入到系统的环境变量路径中。

在导入包之前 加上包的上级目录的路径就可以解决。

**需要注意的是：**

在命令行执行时，若在文件中是使用的相对路径进行所有的路径操作

1.或者包的导入，

2.或者文件的读写

那执行Python时必须切换到相应的py文件目录中再执行。



![img](/assets/images/1.png)



![img](/assets/images/2.png)



![img](/assets/images/3.png)


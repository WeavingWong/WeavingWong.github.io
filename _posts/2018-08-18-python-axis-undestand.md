---
layout: post
date: 2018-08-18 23:35:15.000000000 +09:00
title: python的pandas库中axis的理解
categories: python
tags: axis
---

df.mean其实是在每一行上取所有列的均值，而不是保留每一列的均值。也许简单的来记就是axis=0代表往**跨行（down)**，而axis=1代表**跨列（across)**，作为方法动作的副词（译者注）

换句话说:

- 使用0值表示沿着每一列或行标签\索引值向下执行方法

- 使用1值表示沿着每一行或者列标签模向执行对应的方法

  

![img](/assets/images/clipboard.png)



<center>axis参数作用方向图示</center>



另外，记住，Pandas保持了Numpy对关键字axis的用法，用法在Numpy库的词汇表当中有过解释：

轴用来为超过一维的数组定义的属性，二维数据拥有两个轴：第0轴沿着行的垂直往下，第1轴沿着列的方向水平延伸。

所以问题当中第一个列子 df.mean(axis=1)代表沿着列水平方向计算均值，而第二个列子df.drop(name, axis=1) 代表将name对应的列标签（们）沿着水平的方向依次删掉。
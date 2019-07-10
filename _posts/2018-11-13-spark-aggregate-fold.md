---
layout: post
date: 2018-11-13 23:35:15.000000000 +09:00
title: spark（1）- aggregate方法和fold方法总结
categories: 编程实践
tags: spark
---
### aggregate方法和fold方法总结
最近在学习spark，理解这两个函数时候费了一些劲，现在记录一下。

1. rdd.fold(value)(func)
    说到fold()函数，就不得不提一下reduce()函数，他俩的区别就在于一个初始值。
    reduce()函数是这样写的：


```scala
rdd.reduce(func)
```

<!-- more -->
参数是一个函数，这个函数的对rdd中的所有数据进行某种操作，比如：


```scala
val l = List(1,2,3,4)
l.reduce((x, y) => x + y)
```

对于这个x，它代指的是返回值，而y是对rdd各元素的遍历。
意思是对 l中的数据进行累加。
flod()函数相比reduce()加了一个初始值参数：


```
rdd.fold(value)(func)
```

scala的语法确实是比较奇怪的，既然有两个参数，你就不能把两个参数放在一个括号里吗？也是醉了，这种写法确实容易让人迷惑。


```
val l = List(1,2,3,4)
l.fold(0)((x, y) => x + y)
```

这个计算其实  0 + 1 + 2 + 3 + 4，而reduce()的计算是：1 + 2 + 3 + 4，没有初始值，或者说rdd的第一个元素值是它的初始值。

2. rdd.aggregate(value)(seqOp, combOp)
    刚才说到reduce()和fold()，这两个函数有一个问题，那就是它们的返回值必须与rdd的数据类型相同，啥意思呢？比如刚才那个例子，l的数据是Int，那么reduce()和flod()返回的也必须是Int。
    aggregate()函数就打破了这个限制。比如我返回(Int, Int)。这很有用，比如我要计算平均值的时候。
    要算平均值，我就有两个值是要求的，一个是rdd的各元素的累加和，另一个是元素计数，我初始化为(0, 0)。
    那么就是：

```
val l = List(1,2,3,4)
```

l.aggregate(0, 0)(seqOp, combOp)
那么seqOp和combOp怎么写呢？而combOp又是啥意思呢？
我们将seqOp写为：

(x, y) => (x._1 + y, x._2 + 1)
这啥意思？
在讲到reduce()函数的时候我说：


```scala
val l = List(1,2,3,4)
l.reduce((x, y) => x + y)
```

对于这个x，它代指的是返回值，而y是对rdd各元素的遍历。
在aggregate()这也一样，x不是返回值吗，我返回值是(Int, Int)啊，它有两个元素啊，我可以用x._1和x._2来代指这两个元素的，y不是rdd的元素遍历吗，那我x._1 + y就是各个元素的累加和啊，x._2 + 1就是元素计数啊。遍历完成后返回的(Int, Int)就是累加和和元素计数啊。
按理说有这么一个函数就应该结束了，后边那个combOp是干嘛的？
因为我们的计算是分布式计算，这个函数是将累加器进行合并的。
例如第一个节点遍历1和2, 返回的是(3, 2)，第二个节点遍历3和4, 返回的是(7, 2)，那么将它们合并的话就是3 + 7, 2 + 2，用程序写就是


```
(x, y) => (x._1 + y._1, x._2 + y._2)
```

最后程序是这样的：


```
val l = List(1,2,3,4)
r = l.aggregate(0, 0)((x, y) => (x._1 + y, x._2 + 1), (x, y) => (x._1 + y._1, x._2 + y._2))
m = r._1 / r._2.toFload
```
m就是所要求的均值。

### 参考文献
[简书：贰拾贰画生](https://www.jianshu.com/p/15739e95a46e)

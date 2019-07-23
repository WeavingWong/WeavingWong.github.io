---

layout: post
date: 2018-10-02 23:35:15.000000000 +09:00
title: mysql 学习笔记--关于SQL运行的理解
categories: 数据库
tags: 数据库

---

研究SQL有一段时间，在公司实习的阶段每天需要写大量的查询、汇总相关特征统计与构造方面的工作，刚刚开始的时候还有些不适应，毕竟语法不是很熟练。但是写的多了，渐渐发现找到感觉。自我感觉还行，不过对于其中的原理确实没有怎么深入理解，为此查阅了一些大佬分享的博文，外加买了一本网易出品的经典入门红皮书[《深入浅出MySQL》](https://book.douban.com/subject/25817684/)，深感大有裨益。特此也是分享给路过的各位读者。

![](/assets/images/mysql/book.png)



明天给各位新开一篇总结一下书中关于SQL优化方面的一些学习笔记。

回到对于SQL的理解方面，在大量的博文中，我做了如下的总结：

在格物的《[简单十步全部理解SQL](https://www.cnblogs.com/shockerli/p/10-easy-steps-to-a-complete-understanding-of-sql.html)》介绍了十个理解SQL的心得。有些比较容易理解，有些则有些深奥，可能我还水平不够，没有接触那么深入。

### 以下几点相对容易理解

　**1、SQL是声明性语言**

　　首先你需要思考的是，声明性。你唯一需要做的只是声明你想获得结果的性质，而不需要考虑你的计算机怎么算出这些结果的。

```mysql
SELECT first_name, last_name FROM employees WHERE salary > 100000
```

　　这很容易理解，你无须关心员工的身份记录从哪来，你只需要知道谁有着不错的薪水。

　　**从中我们学到了什么呢？**

　　那么如果它是这样的简单，会出现什么问题吗？问题就是我们大多数人会直观地认为这是命令式编程。如：“机器，做这，再做那，但在这之前，如果这和那都发生错误，那么会运行一个检测”。这包括在变量中存储临时的编写循环、迭代、调用函数，等等结果。

　　把那些都忘了吧，想想怎么去声明，而不是怎么告诉机器去计算。

　**2、SQL语法不是“有序的”**

　　常见的混乱源于一个简单的事实，SQL语法元素并不会按照它们的执行方式排序。语法顺序如下：

- SELECT [DISTINCT]
- FROM
- WHERE
- GROUP BY
- HAVING
- UNION
- ORDER BY

　　为简单起见，并没有列出所有SQL语句。这个语法顺序与逻辑顺序基本上不同，执行顺序如下： 

- FROM
- WHERE
- GROUP BY
- HAVING
- SELECT
- DISTINCT
- UNION
- ORDER BY

　　这有三点需要注意：

　　1、第一句是FROM，而不是SELECT。首先是将数据从磁盘加载到内存中，以便对这些数据进行操作。

　　2、SELECT是在其他大多数语句后执行，最重要的是，在FROM和GROUP BY之后。重要的是要理解当你觉得你可以从WHERE语句中引用你定义在SELECT语句当中的时候，。以下是不可行的：

```
SELECT A.x + A.y AS z

FROM A

WHERE z = 10 -- z is not available here!
```

　　如果你想重用z,您有两种选择。要么重复表达式: 

```
SELECT A.x + A.y AS z

FROM A

WHERE (A.x + A.y) = 10
```

　　或者你使用派生表、公用表表达式或视图来避免代码重复。请参阅示例进一步的分析：

　　3、在语法和逻辑顺序里，UNION都是放在ORDER BY之前，很多人认为每个UNION子查询都可以进行排序，但根据SQL标准和大多数的SQL方言，并不是真的可行。虽然一些方言允许子查询或派生表排序，但却不能保证这种顺序能在UNION操作后保留。

　　需要注意的是，并不是所有的数据库都以相同的形式实现，例如规则2并不完全适用于MySQL,PostgreSQL,和SQLite上

　　**从中我们学到了什么呢？**

　　要时刻记住SQL语句的语法顺序和逻辑顺序来避免常见的错误。如果你能明白这其中的区别，就能明确知道为什么有些可以执行有些则不能。

**3、SQL是关于数据表引用的** 

**4、SQL数据表引用可以相当强大**

**5、应使用SQL JOIN的表，而不是以逗号分隔表** 

　　前面，我们已经看到这语句： 

```
FROM a, b
```

　　高级SQL开发人员可能会告诉你，最好不要使用逗号分隔的列表，并且一直完整的表达你的JOINs。这将有助于改进你的SQL语句的可读性从而防止错误出现。

　　一个非常常见的错误是忘记某处连接谓词。思考以下内容：



```mysql
FROM a, b, c, d, e, f, g, h

WHERE a.a1 = b.bx

AND a.a2 = c.c1

AND d.d1 = b.bc

-- etc...
```

　　使用join来查询表的语法

- 更安全，你可以把连接谓词与连接表放一起，从而防止错误。
- 更富于表现力，你可以区分外部连接，内部连接，等等。

　　从中我们学到了什么呢？

　　使用JOIN，并且永远不在FROM语句中使用逗号分隔表引用。 

**6、SQL的不同类型的连接操作**

　　连接操作基本上有五种

- EQUI JOIN
- SEMI JOIN
- ANTI JOIN
- CROSS JOIN
- DIVISION

　　这些术语通常用于关系代数。对上述概念，如果他们存在，SQL会使用不同的术语。让我们仔细看看:

**EQUI JOIN（同等连接）**

　　这是最常见的JOIN操作。它有两个子操作:

- INNER JOIN(或者只是JOIN)
- OUTER JOIN(可以再次拆分为LEFT, RIGHT,FULL OUTER JOIN)

**SEMI JOIN（半连接）**

虽然不能肯定你到底是更加喜欢IN还是EXISTS，而且也没有规则说明，但可以这样说：

- IN往往比EXISTS更具可读性
- EXISTS往往比IN更富表现力（如它更容易表达复杂的半连接）
- 一般情况下性能上没有太大的差异，但，在某些数据库可能会有巨大的性能差异。

　　因为INNER JOIN有可能只产生有书的作者，因为很多初学者可能认为他们可以使用DISTINCT删除重复项。他们认为他们可以表达一个像这样的半联接：

```mysql
-- Find only those authors who also have books

SELECT DISTINCT first_name, last_name

FROM author
```

　　这是非常不好的做法，原因有二：

- 它非常慢，因为该数据库有很多数据加载到内存中，只是要再删除重复项。
- 它不完全正确，即使在这个简单的示例中它产生了正确的结果。但是，一旦你JOIN更多的表引用，，你将很难从你的结果中正确删除重复项。

　　更多的关于DISTINCT滥用的问题，可以访问这里的[博客](http://blog.jooq.org/2013/07/30/10-common-mistakes-java-developers-make-when-writing-sql/)。

**ANTI JOIN（反连接）**

　　这个关系的概念跟半连接刚好相反。您可以简单地通过将 NOT 关键字添加到IN 或 EXISTS中生成它。在下例中，我们选择那些没有任何书籍的作者：

**CROSS JOIN（交叉连接）**

**DIVISION（除法）**

**7、SQL的派生表就像表的变量**

**8、SQL GROUP BY转换之前的表引用**

**9、SQL SELECT在关系代数中被称为投影**



**10.相对简单一点的SQL DISTINCT,UNION,ORDER BY,和OFFSET**

看完复杂的SELECT之后，我们看回一些简单的东西。

- 集合运算（DISTINCT和UNION）
- 排序操作（ORDER BY,OFFSET..FETCH）

　　**集合运算**

　　集合运算在除了表其实没有其他东西的“集”上操作。嗯，差不多是这样，从概念上讲，它们还是很容易理解的

- DISTINCT投影后删除重复项。
- UNION求并集，删除重复项。
- UNION ALL求并集，保留重复项。
- EXCEPT求差集（在第一个子查询结果中删除第二个子查询中也含有的记录删除），删除重复项。
- INTERSECT求交集（保留所有子查询都含有的记录），删除重复项。

　　所有这些删除重复项通常是没有意义的，很多时候，当你想要连接子查询时，你应该使用UNION ALL。

　　**排序操作**

　　排序不是一个关系特征，它是SQL仅有的特征。在你的SQL语句中，它被应用在语法排序和逻辑排序之后。保证可以通过索引访问记录的唯一可靠方法是使用ORDER BY a和OFFSET..FETCH。所有其他的排序总是任意的或随机的，即使它看起来像是可再现的。



### 总结

在学习过程中还是会有很多的不懂，此处对于比较生涩的内容进行了跳过，后期接触到相关内容再进行补充，所以在笔记的摘录和回顾中进行了相应标题的留白，如果对相关主题感兴趣请移步源地址，见参考项。


###  参考
链接：https://www.cnblogs.com/shockerli/p/10-easy-steps-to-a-complete-understanding-of-sql.html

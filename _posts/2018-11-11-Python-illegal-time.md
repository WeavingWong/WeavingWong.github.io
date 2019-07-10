---
layout: post
date: 2018-11-11 23:35:15.000000000 +09:00
title: Python(1)--MySQL数据库中非法时间字段的处理方法
categories: 编程实践
tags: python
---


### 1.MySQL中常用时间数据类型范围大小

由于不同的数据结构类型导致能够显示的范围不一样，如果在工作中使用范围较小的类型，当输入的时间范围不够规范时，容易导致程序抛出异常，中断进程的运行。下面给大家列出MySQL中常用的数据格式的范围：

```
datetime 8 bytes YYYY-MM-DD HH:MM:SS   1000-01-01 00:00:00 ~ 9999-12-31 23:59:59 
timestamp 4 bytes YYYY-MM-DD HH:MM:SS  1970-01-01 00:00:01 ~ 2038 
date 3 bytes YYYY-MM-DD                1000-01-01 ~ 9999-12-31 
year 1 bytes YYYY                      1901 ~ 2155
```
<!-- more -->
由上可知，常用的timestamp显示范围仅为1970-01-01 00:00:01 ~ 2038，如果一个输入错误很容易导致系统无法识别。当然解决办法是有的：

#### 1.如果输入的字符为0000-00-00 00:00:00

这是很常见的输入异常数据，往往在大型系统中对于时间的输入如果没有做限制之时，有可能用户会选择留空或者乱填一下，这个时候如果输入端没有做正则表达式匹配判断（假设），那留空之后在数据库中就可能会显示0000-00-00 00:00:00.如果是用JDBC连接Mysql数据库比较好解决:

```
datasource.url=jdbc:mysql://localhost:3306/testdb?useUnicode=true&characterEncoding=utf-8&zeroDateTimeBehavior=convertToNull&transformedBitIsBoolean=true
```
在连接数据库的URL中添加处理字段即可

#### 2,。若输出字段是超过范围的其他值，比如“2048-00-00 00:00:00”
这个时候只能放大招了，可以确认这个数据是无效数据，是时候进行匹配和替换处理。
问题来了，一般的数据库表中时间字段可能会存在Null值，正常值，还有非法值，我们常见的处理办法是讲整个一列数据读取临时缓存，将临时数据中的该列全部转成字符串，再做正则匹配或者其他特定处理。但是，存在Null值得时候系统会报一个类型无法转换的错，意思是Null值的类型是None（在python中），无法转换成str,系统再次报错。
---
转换思想，我们做一个判断，先判断这条记录是不是None：

```python
if data[k，2] is None:
    pass 
    #判断是空，不做处理，因为至少数据不是非法
else:
    #其他处理
```
---
本以为没有问题，结果发现数据转换成字符串的代码报错，我的转换方式是：

```python
timestr = datatime[k,2].strftime('%Y-%m-%d %H:%M:%S')
```
报错的信息显示：字符串类型数据不能使用strftime函数
当时的我一脸懵逼，我本来是timestampe格式啊！！！

.....一顿瞎忙活之后，累死个人。突然发现，有些数据是可以转换，有些说不能使用这个函数，心想是不是那些非法数据不能用，经过测试证实确实是。
那就好办了啊。直接进行字符串匹配就好了。

### 2.最佳方法

更好的办法也在后续的调试中被发现，分析发现既然非法数据会在读取时自动转为str类型，那就这样好了：

```
if isinstance(data.iloc[k, 3], str):
    #若是不合法时间格式，系统自动会转成str类型，所以不需要做格式转换，直接做字符串匹配或判断类型
    data.iloc[k, 3] = default_date
    print(data.iloc[k, 3])
# elif data.iloc[k, 3] is None:
#     # 若是字段为Null，系统自动会转成None类型，不用转换，直接做判断
#     data.iloc[k, 3] = default_date #可选设置时间
else:
    pass
```
至此，在python中处理非法时间字段就完成了。
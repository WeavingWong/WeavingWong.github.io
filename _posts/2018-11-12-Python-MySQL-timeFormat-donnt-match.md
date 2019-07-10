---
layout: post
date: 2018-11-12 23:35:15.000000000 +09:00
title: Python(2)--Python与MySQL的时间格式通配符不一致
categories: 编程实践
tags: python
---

### 1.又拔了一个萝卜
由于今天工作上遇到需要将python中获取的当前时间存入MySQL数据库中，数据库的表示提前建好的，时间类型选择DATETIME（假设此处格式类型不可改变），python中有一个库为


```
from datetime import timedelta, datetime
```
所以理所应当的选择用这个方法获取时间

```
（datetime.today() + timedelta(-1)).strftime('%Y-%m-%d %H:%M:%S')
```
但是当我将获取的时间值写入数据库时，他的报错信息为
<!-- more -->

```
Not all str can be converted 

# 意思就是并不是所有的str都能直接转类型（这里说的额类型是DAETIME）
```
下面科普一下DTAETIME类型格式：

```
2018-10-29 17:03:06 

#具体不做讲解，大概长这样

```
经过查询，了解到MySQL支持一个叫做 str_to_date的内置函数，所以照着样子用用试试


```
INSERT INTO t_stats_sync_log VALUES(str_to_date('2018-10-29 17:03:06','%Y-%m-%d %H:%M:%S'))
```

结果，他疯狂报错，总说输入的格式有误。近乎崩溃。

最后我发现了一个小问题，MySQL的分钟的通配符竟然用的是

```
%i  # 这个是MySQL唯一使用作为分钟的通配符，即

INSERT INTO t_stats_sync_log VALUES(str_to_date('2018-10-29 17:03:06','%Y-%m-%d %H:%i:%S'))
```
至此，问题解决，数据写入正常。

### 附录常见MySQL通配符


```
1.mysql日期和字符相互转换方法 
date_format(date,’%Y-%m-%d’) ————–>oracle中的to_char(); 
str_to_date(date,’%Y-%m-%d’) ————–>oracle中的to_date();

%Y：代表4位的年份 
%y：代表2为的年份

%m：代表月, 格式为(01……12) 
%c：代表月, 格式为(1……12)

%d：代表月份中的天数,格式为(00……31) 
%e：代表月份中的天数, 格式为(0……31)

%H：代表小时,格式为(00……23) 
%k：代表 小时,格式为(0……23) 
%h： 代表小时,格式为(01……12) 
%I： 代表小时,格式为(01……12) 
%l ：代表小时,格式为(1……12)

%i： 代表分钟, 格式为(00……59) 【只有这一个代表分钟，大写的I 不代表分钟代表小时】

%r：代表 时间,格式为12 小时(hh:mm:ss [AP]M) 
%T：代表 时间,格式为24 小时(hh:mm:ss)

%S：代表 秒,格式为(00……59) 
%s：代表 秒,格式为(00……59)

2.例子： 
select str_to_date(‘09/01/2009’,’%m/%d/%Y’)

select str_to_date(‘20140422154706’,’%Y%m%d%H%i%s’)

select str_to_date(‘2014-04-22 15:47:06’,’%Y-%m-%d %H:%i:%s’)
```

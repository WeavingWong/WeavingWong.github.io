---

layout: post
date: 2018-09-12 23:35:15.000000000 +09:00
title: 剑指Offer--(1)查找空格 
categories: 算法与数据结构
tags: 数据结构

---

### 题目 
> 请实现一个函数，将一个字符串中的每个空格替换成“%20”。例如，当字符串为We Are Happy.则经过替换之后的字符串为We%20Are%20Happy。



### 解读
此题需要清楚一个字符串类型的属性，也就是它具有哪些性质，比如是否可迭代，还是需要列表帮忙？索引怎么找？是否需要切片？字符串是否是可以直接拼接？

### 1.我的解答
```python
# -*- coding:utf-8 -*-
class Solution:
    # s 源字符串
    def replaceSpace(self, s):
        # write code here
        for i in s:
            if i is ' ':
                #s[s.index(i)] = "20%"
                s = s[:s.index(i)] + "%20"+ s[s.index(i)+1:]
        return s
```
#### 思路
> 字符串具有的性质归纳： 
> 1. 可迭代（可以使用for遍历）  
> 2. 不可变(与tuple类似),但可以由部分拼接成新字符串
> 3. 可寻址（可用index()）
> 4. 可转变成list\tuple
> 5. ...
> 


### 2，其他解法

```python
# -*- coding:utf-8 -*-
class Solution:
    # s 源字符串
    def replaceSpace(self, s):
        # write code here
        s = list(s)
        count=len(s)
        for i in range(0,count):
            if s[i]==' ':
                s[i]='%20'
        return ''.join(s) #将列表s重新连接起来，用”“连接(即中间无空隙)
```
### 3.更简洁的方法

```python

# -*- coding:utf-8 -*-
class Solution:
    # s 源字符串
    def replaceSpace(self, s):
        # write code here
        return s.replace(' ','%20')
```
> 利用了python的内置方法replace()
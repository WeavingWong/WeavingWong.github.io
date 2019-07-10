---
layout: post
date: 2018-09-18 23:35:15.000000000 +09:00
title: 剑指Offer--(1)查找重复的数字
categories: 算法与数据结构
tags: 数据结构
---
### 摘要

从这个时间段开始，做一些编程的练习。回想起来，入学已经一年有余了，一直以来学的东西杂而多，感觉有一些抓不到主要矛盾，不过，大的方向上存在的技术壁垒还是依然存在的，我想，无论以后具体从事的是哪一种岗位，基本功还是需要的，现在想想也是可笑。每天学的东西听起来都是高大上，可以真正自己有几斤几两，还是特别清楚的。所以还是把基础打好吧。另外，在实验室的学习暂时没有项目作为引导，所以实习是很有必要。过了这一段时间，是该要出去锻炼一下了。

<!-- more -->

##  Python语言实现

### 1.采用字典（类似hash表）方法
>  利用字典表结构解决，首先定义一个空字典，然后从头到尾扫描数组中的每个数字，如果发现该数字不在在字典中（最初字典为空，所以第一个数字绝对满足），则将该数字加入到字典作为键，值可任意（如为0），继续查询，若查询到某数字存在于字典中，则证明存在重复，输出到duplication[0]中，返回True,否则返回False.

```python
# -*- coding:utf-8 -*-
class Solution:
    # 这里要特别注意~找到任意重复的一个值并赋值到duplication[0]
    # 函数返回True/False
    def duplicate(self, numbers, duplication):
        # write code here
        dict = {}
        for num in numbers:
            if num not in dict:
                dict[num]=0
            else:
                duplication[0]=num
                return True
        return False
```
> 为何不用list代替dict作为缓冲对象，因为dict.keys()实际上是list(keys),是dict的所有key组成的list。查找一个元素是否在list中是以list的下标为索引遍历list.而查询是否在dict中，是将key以hash值的形式直接找到key对应的索引，根据索引可直接访问value。对量大的dict查询，自然是后者快很多。 

以上方法可以实现时间复杂度为O(n),空间复杂度为O(n)。

---
### 2.空间复杂度为O(1)的解法
> 注意到数组中的数字都在0~n-1之间，若没有重复，则数组重排序之后数字i将出现在第i个位置，否则，就会出现重复或者位置缺失。做如下重排：依次扫描这个数组中的数字，当扫描到下标为i的数字时，首先比较这个数字（m表示）是不是i,若是，则接着扫描下一个数字；若不是，则用它与第m个数字进行比较，若与第m个数字相同，则找到一个重复的数字。若不相等，就把它与第m个数字交换。把m放到属于它的地方。如此循环。直到发现重复的数字。(该方法会改变数组的值)

```python
# -*- coding:utf-8 -*-
class Solution:
    # 这里要特别注意~找到任意重复的一个值并赋值到duplication[0]
    # 函数返回True/False
    def duplicate(self, numbers, duplication):
        # write code here
        if numbers is None or len(numbers) == 0:
            return False
        for i in numbers:
            if i < 0 or i >= len(numbers):
                return False
        for i in range(len(numbers)):
            while  i != numbers[i]:
                if numbers[i] == numbers[numbers[i]]:
                    duplication[0]=numbers[i]
                    return True
                temp = numbers[numbers[i]]
                numbers[numbers[i]]= numbers[i] 
                numbers[i] = temp
        return False
    
```
### 3.利用现有数组为标志位

> 不需要额外的数组保存，利用题目中说“数组里数字的范围在0 ~ n-1 之间”，所以可以利用现有数组设置标志，当一个数字被访问过后，可以设置对应位上的数 + n，之后再遇到相同的数时，会发现对应位上的数已经大于等于n了，那么直接返回这个数即可。(该方法会改变数组的值)

```python
class Solution:
    # 这里要特别注意~找到任意重复的一个值并赋值到duplication[0]
    # 函数返回True/False
    def duplicate(self, numbers, duplication):
        long = len(numbers)
        for i in range(len(numbers)):
            index = numbers[i]%long if numbers[i] >= long else numbers[i]
            if numbers[index] > long:
                duplication[0] = index
                return True
            numbers[index] += long
        return False
```

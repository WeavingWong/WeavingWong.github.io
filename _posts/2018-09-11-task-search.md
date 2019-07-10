---
layout: post
date: 2018-09-11 23:35:15.000000000 +09:00
title: 剑指Offer--(2)二位数组中的查找
categories: 算法与数据结构
tags: 数据结构
---



### 题目描述

在一个二维数组中（每个一维数组的长度相同），每一行都按照从左到右递增的顺序排序，每一列都按照从上到下递增的顺序排序。请完成一个函数，输入这样的一个二维数组和一个整数，判断数组中是否含有该整数。

### 题目解读
> python 在二维数组中的array的格式是array[row][column],所以在遍历时需要统计多维数组的行数和列数的时候，如果是直接对数组名求len(array),那得到的是多维数组的行数，如果是对某一行求len(array[i])则是求第i维的长度，也即是列数。

```python

# -*- coding:utf-8 -*-
class Solution:
    # array 二维列表
    def Find(self, target, array):
        # write code here
        row = 0
        col =len(array[0])-1
        if array == None:
            return Flase
        while row < len(array) and col >= 0:
            if array[row][col] == target:
                return True
            elif array[row][col] < target:
                row += 1
            else:
                col -=1
        return False
        
```
> 总结错误：编程太少，低级错误比较多，今后还是需要加强训练，对于多维数组的概念不是很清楚。
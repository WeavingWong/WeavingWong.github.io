---

layout: post

date: 2019-08-10 23:36:15.000000000 +09:00

title: 贝壳_算法笔试_采木（砍树）

categories: 秋招

tags: python

---

采用动态规划的方法求出在每棵树处理时的用每一种工具的最短时间

原题：

n 棵树需要砍，现有工具锯子和斧头，砍第i颗树所需时间为ai，bi，要求一开始拿斧头，砍第i颗树前交换工具需要ci时间，问依次看完这些树需要的最短时间为多少？


```python
#!/bin/python
# -*- coding: utf8 -*-
import sys
import os
import re


# 作者：凡晨201903060932989
# 链接：https://www.nowcoder.com/discuss/220708?type=post&order=time&pos=&page=1
# 来源：牛客网

def GetResult(n, matrix):
    dp = [[0 for _ in range(2)] for _ in range(n)]
    dp[0][0], dp[0][1] = matrix[0][0] + matrix[0][2], matrix[0][1]
    # 初始化第1棵树分别用工具0和1的时间 
    for i in range(1, n):
        dp[i][0] = min(dp[i - 1][0] + matrix[i][0], dp[i - 1][1] + matrix[i][0] + matrix[i][2])
        # 在第i棵树用工具0的最短时间
        dp[i][1] = min(dp[i - 1][1] + matrix[i][1], dp[i - 1][0] + matrix[i][1] + matrix[i][2])
        # 在第i颗树用工具1的最短时间
    print(dp)
    return min(dp[n - 1][0], dp[n - 1][1])

# 输入四行，第一行一个数，在输入三行，每个数字以空格隔开
n = int(input())
input_list = []
for i in range(n):
    input_list.append(list(map(int, input().split())))
#     for j in range(len(input_list[i])):
#         print(input_list[i][j])

res = GetResult(n, input_list)

print(res, "\n")
```

    3
    20 40 20
    10 4 25
    90 100 5
    [[40, 40], [50, 44], [139, 144]]
    139 



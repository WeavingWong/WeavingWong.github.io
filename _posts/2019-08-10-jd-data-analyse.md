

```python
---

layout: post

date: 2019-08-10 23:36:15.000000000 +09:00

title: 京东 数据分析 求特定数列第n项+满足a^b=c^d的全部个数

categories: 秋招

tags: python

---
```


```python
# 输入数字n，求数列（1,2，2,3,3,3,4,4,4,4....n个n）的第n项 是多少
def get_result(n):
    n_list=[]
    for i in range(1,n+1):
        n_list+=[i]*i
    res=n_list[n-1]
    return res
if __name__=='__main__':
    n = int(input())
    res = get_result(n)
    print(res)
    
# 以上内存需求太大。不适合
```


```python
# 最优结果
def getvaluesofn(n):
    counts = 0
    if n==1:
        return 1
    if n==2:
        return 2
    if n>=3:
        for i in range(n):
            for j in range(i):
                counts += 1
                if counts == n:
                    return i
N = int(input())
print(getvaluesofn(N))
```

    100000
    第100000项
    447
    


```python
# 输入n，求满足a^b=c^d的全部式子有多少个（1<=a,b,c,d<=n），如输入2输出6
import math
def gcd(a,b):
    a,b=min(a,b),max(a,b)
    for i in range(1,a+1):
        if a%i==0 and b%i==0:
            cd=i
    return cd
n=int(input())
mode=int(1e9+7)
rec=[0]*(n+1)
res=n*(2*n-1)%mode
for i in range(2,n+1):
    if rec[i]:
        continue
    else:
        rec[i]=1
        lgn=round(math.log(n,i))
    if pow(i,lgn)>n:
        lgn = lgn-1
    for lga in range(1,lgn):
        rec[i**lga]=1
        for lgc in range(lga+1,lgn+1):
            res+=(n//(lgc//gcd(lga,lgc))*2)
            res=res%mode
print(res)
```

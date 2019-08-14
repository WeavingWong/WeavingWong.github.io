---
layout: post
date: 2018-08-17 23:35:15.000000000 +09:00
title: 剑指offer-反转链表
categories: 编程实践
tags: python
---


输入一个链表，输出反转后的链表。 
剑指offer中的Python解法：


```python
# class ListNode:
#     def __init__(self, x):
#         self.val = x
#         self.next = None

class Solution:
    # 返回ListNode
    def ReverseList(self, pHead):
        if not pHead or not pHead.next:
            return pHead
        last = None    #指向上一个节点
        while pHead:
            #先用tmp保存pHead的下一个节点的信息，
            #保证单链表不会因为失去pHead节点的next而就此断裂
            tmp = pHead.next   
            
            # 保存完next，就可以让pHead的next指向last了
            pHead.next = last
            
            # 让last，pHead依次向后移动一个节点，继续下一次的指针反转
            last = pHead
            
            pHead = tmp
        return last

```
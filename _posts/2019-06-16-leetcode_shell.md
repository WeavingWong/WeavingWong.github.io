---
layout: post
date: 2019-06-16 23:35:15.000000000 +09:00
title: LeetCode shell 编程实践
categories: 编程实践
tags: shell
---


### 统计词频

写一个 bash 脚本以统计一个文本文件 words.txt 中每个单词出现的频率。

为了简单起见，你可以假设：

words.txt只包括小写字母和 ' ' 。
每个单词只由小写字母组成。
单词间由一个或多个空格字符分隔。
示例:

假设 words.txt 内容如下：

the day is sunny the the
the sunny is is
你的脚本应当输出（以词频降序排列）：

the 4
is 3
sunny 2
day 1
说明:

不要担心词频相同的单词的排序问题，每个单词出现的频率都是唯一的。
你可以使用一行 Unix pipes 实现吗？

```bash
awk '{
    for(i = 1; i <= NF; i++){
        res[$i] += 1 #以字符串为索引，res[$i]相同的累计
    }
}
END{
    for(k in res){
        print k" "res[k]
    }
}' words.txt | sort -nr -k2  # n：按数值排序，r：倒序，k：按第2列排序
```
```bash
cat words.txt | tr -s ' ' '\n' | sort | uniq -c | sort -r | awk '{ print $2, $1 }'

tr-s：用目标字符串截断字符串，但只剩下一个实例(例如，多个空格)
sort：使相同的字符串连续，以便uniq能够完整和正确地计数相同的字符串。
Uniq-c：uniq用于过滤连续的重复行，
-c :表示计数排序
-r ：-r表示按降序排列
awk'{print$2，$1}'：以格式化输出
```

### 有效电话号码

给定一个包含电话号码列表（一行一个电话号码）的文本文件 file.txt，写一个 bash 脚本输出所有有效的电话号码。

你可以假设一个有效的电话号码必须满足以下两种格式： (xxx) xxx-xxxx 或 xxx-xxx-xxxx。（x 表示一个数字）

你也可以假设每行前后没有多余的空格字符。

假设 file.txt 内容如下：

987-123-4567
123 456 7890
(123) 456-7890
你的脚本应当输出下列有效的电话号码：

987-123-4567
(123) 456-7890


```bash
grep -P '^(([0-9]{3}) |[0-9]{3}-)[0-9]{3}-[0-9]{4}$' file.txt
grep -P '^(\d{3}-|\(\d{3}\) )\d{3}-\d{4}$' file.txt
# -P 是采用perl语法进行匹配

sed -n -r '/^([0-9]{3}-|\([0-9]{3}\) )[0-9]{3}-[0-9]{4}$/p' file.txt
awk '/^([0-9]{3}-|\([0-9]{3}\) )[0-9]{3}-([0-9]{4})$/' file.txt

```

### 转置文件

给定一个文件 file.txt，转置它的内容。

你可以假设每行列数相同，并且每个字段由 ' ' 分隔.

示例:

假设 file.txt 文件内容如下：

name age
alice 21
ryan 30
应当输出：

name alice ryan
age 21 30


```bash
# Read from the file file.txt and print its transposed content to stdout.
awk '{ #这个大括号里的代码是 对正文的处理
    # NF表示列数，NR表示已读的行数
    # 注意for中的i从1开始，i前没有类型
    for (i=1; i<=NF; i++){#对每一列
        if(NR==1){       #如果是第一行
            #将第i列的值存入res[i],$i表示第i列的值，i为数组的下标，以列序号为下标，
            #数组不用定义可以直接使用
            res[i]=$i;   
        }
        else{
            #不是第一行时，将该行对应i列的值拼接到res[i]
            res[i]=res[i] " " $i
        }
    }
}
# BEGIN{} 文件进行扫描前要执行的操作；END{} 文件扫描结束后要执行的操作。
END{
    #输出数组
	for (i=1; i<=NF; i++){
		print res[i]
	}
}' file.txt
```

```bash
transpose=`head -n1 file.txt | wc -w`   # wc是计数，head -n1 是前1行

for i in `seq 1 $transpose`             # sep 1 10 生成1到10的序列
do
    echo `cut -d' ' -f$i file.txt`      # cut -d " " -f1 以空格隔开，第1个区域
done
```

### 第十行

给定一个文本文件 file.txt，请只打印这个文件中的第十行。

示例:

假设 file.txt 有如下内容：

Line 1
Line 2
Line 3
Line 4
Line 5
Line 6
Line 7
Line 8
Line 9
Line 10
你的脚本应当显示第十行：

Line 10
说明:
1. 如果文件少于十行，你应当输出什么？
2. 至少有三种不同的解法，请尝试尽可能多的方法来解题。

```bash
awk 'NR == 10' file.txt     # NR在awk中指行号

sed -n 10p file.txt         # -n表示只输出匹配行，p表示Print

tail -n+10 file.txt|head -1  # tail -n +10表示从第10行开始输出
```
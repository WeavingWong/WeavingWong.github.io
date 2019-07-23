---
layout: post
date: 2018-10-16 23:35:15.000000000 +09:00
title: bug记录 - mysql写入dataframe注意事项
categories: 编程实践
tags: python&mysql
---

## 报错信息：

sqlalchemy.exc.InternalError: (pymysql.err.InternalError) (1054, "Unknown column 'index' in 'field list'") [SQL: 'INSERT INTO uc_anti_cheat_rule (`index`, id, name, description, remark) VALUES (%(index)s, %(id)s, %(name)s, %(description)s, %(remark)s)'] [parameters: ({'index': 0, 'id': '8', 'name': 'User4', 'description': '22', 'remark': 'ddd'}, {'index': 1, 'id': '9', 'name': '3', 'description': '5', 'remark': '44'})] (Background on this error at: http://sqlalche.me/e/2j85)


解决方案： 因为df在写入mysql时会将index也写入，所以需要删除索引，或者将索引的名字改成id

```
df1.set_index(["id"], inplace=True)

df.read_csv(filename ,  index = False)

重命名级别名称:
df1.index.names = ['index']
df1.columns.names = ['column']
```


- 解决方案更新
```
在pd.to_sql的接口中有一个参数为index,将其设置为False 

pd.to_sql(index=False)
# 问题即可解决
# 说明：由于dataframe 默认写入数据库时带有索引列，所以当数据库中存在默认id列时，会出现列数不匹配

```
## 总结

- 遇到问题还是要先看官方文档，网络上的解决方案可能不适用自己遇到的情况
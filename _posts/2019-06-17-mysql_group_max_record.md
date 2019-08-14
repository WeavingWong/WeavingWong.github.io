---

layout: post
date: 2019-06-17 23:35:15.000000000 +09:00
title: MySQL——分组查询表内某字段最大值所在的记录
categories: 编程实践
tags: MySQL

---


```mysql
# 示例 

CREATE TABLE orders
(id VARCHAR(10),
statu CHAR(1),
goods_id VARCHAR(10),
price DECIMAL(12,2)
);
 

INSERT INTO orders(id,goods_id,statu,price) VALUES('1','g','1',100);
INSERT INTO orders(id,goods_id,statu,price) VALUES('2','g','1',200);
INSERT INTO orders(id,goods_id,statu,price) VALUES('3','g','0',300);
INSERT INTO orders(id,goods_id,statu,price) VALUES('4','g','1',400);
INSERT INTO orders(id,goods_id,statu,price) VALUES('5','c','0',150);
INSERT INTO orders(id,goods_id,statu,price) VALUES('6','c','1',250);
INSERT INTO orders(id,goods_id,statu,price) VALUES('7','c','0',350);
INSERT INTO orders(id,goods_id,statu,price) VALUES('8','c','1',400);

# 方法一：使用子查询
SELECT a.* 
FROM 
orders a ,(SELECT b.goods_id,MAX(b.price) price 
FROM orders b GROUP BY b.goods_id) c 
WHERE a.goods_id = c.goods_id AND a.`price` = c.price


 # 使用in 和exists 
SELECT a.* FROM orders a 
WHERE NOT EXISTS
(SELECT 1 FROM orders b 
WHERE  a.goods_id = b.goods_id AND a.price <b.price)


SELECT * FROM orders 
WHERE price NOT IN 
(SELECT a.price FROM orders b,orders a 
WHERE  a.goods_id = b.goods_id AND a.price <b.price)
```
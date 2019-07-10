---
layout: post
date: 2018-08-17 23:35:15.000000000 +09:00
title: 机器学习（2）-Tensorflow初步
categories: tensorflow
tags: 机器学习 
---

## Tensorflow库函数的理解

很多时候对于tensorflow 官网的一些api的解释很不理解，有时候在各大论坛逛一遍才能明白真实的意思，虽然这样也是一个解决方案，但是人毕竟不是神。好记性确实不如烂笔头，偶尔记录一下，在本页汇集比较难理解的一些point.

<!-- more -->

### tf.matmul()

> ```python
> tf.matmul(
> a,
> b,
> transpose_a=False,
> transpose_b=False,
> adjoint_a=False,
> adjoint_b=False,
> a_is_sparse=False,
> b_is_sparse=False,
> name=None
> )
> a：Tensor类型的float16，float32，float64，int32，complex64， complex128和秩> 1。
> b：Tensor具有相同的类型和等级a。
> transpose_a：如果True，a则在乘法之前进行转置。
> transpose_b：如果True，b则在乘法之前进行转置。
> adjoint_a：如果True，a则在乘法之前进行共轭和转置。
> adjoint_b：如果True，b则在乘法之前进行共轭和转置。
> a_is_sparse：If True，a被视为稀疏矩阵。
> b_is_sparse：If True，b被视为稀疏矩阵。
> name：操作的名称（可选）。
> ```

这个函数的作用是用于两个矩阵a*b相乘，但要求b要与a的类型一致，如果一个或两个矩阵包含大量零(即可认为是稀疏矩阵)，则可以通过将相应a_is_sparse或b_is_sparse标志设置为true来使用更有效的乘法算法.

```
import tensorflow as tf
a=tf.matmul([[1, 2, 3],
             [4, 5, 6]],

            [[ 7,  8]
            ,[ 9, 10]
            ,[11, 12]])  +[3,4]

with tf.Session() as sess:
    print (sess.run(eval('a')))
    
#the result is:
# [[ 58,  64],
#  [139, 154]]
--->
# [[ 61  68]
 #[142 158]]
```

以上代码实现了矩阵加法的扩展，多维矩阵加一维数组，数组会扩张成与维数更高的矩阵一样的形状。

---

### tf.placeholder(dtype, shape=None, name=None)

此函数可以理解为形参，用于定义过程，在执行的时候再赋具体的值.

参数：

dtype：数据类型。常用的是tf.float32,tf.float64等数值类型
shape：数据形状。默认是None，就是一维值，也可以是多维，比如[2,3], [None, 3]表示列是3，行不定
name：名称。

```python
x = tf.placeholder(tf.float32, shape=(1024, 1024))
y = tf.matmul(x, x)

with tf.Session() as sess:
  print(sess.run(y))  # ERROR: 此处x还没有赋值.

  rand_array = np.random.rand(1024, 1024)
  print(sess.run(y, feed_dict={x: rand_array}))  # Will succeed.
```

```python
input1 = tf.placeholder(tf.float32)
input2 = tf.placeholder(tf.float32)
output = tf.mul(input1, input2)

with tf.Session() as sess:
  print(sess.run([output], feed_dict={input1:[7.], input2:[2.]}))

# output:
# [array([ 14.], dtype=float32)]
如果你没有提供 feed 的话，placeholder() 操作会错误。MNIST fully-connected feed tutorial (source code) 这里有一个大规模的 feed 示例。
```

---

### tf.constant
    tf.constant(
    value,
    dtype=None,
    shape=None,
    name='Const',
    verify_shape=False
    )
创建一个恒定值value执行填充。

---
### Softmax函数

> 在数学，尤其是概率论和相关领域中，Softmax函数，或称归一化指数函数，是逻辑函数的一种推广。它能将一个含任意实数的K维向量 “压缩”到另一个K维实向量 中，使得每一个元素的范围都在 {(0,1)}之间，并且所有元素的和为1.

---



### tf.reduce_sum

> ```
> tf.reduce_sum(
> ​```
> input_tensor,
> axis=None,
> keepdims=None,
> name=None,
> reduction_indices=None,
> keep_dims=None
> ​```
> )
> ```

```python
# 'x' is [[1, 1, 1]
#         [1, 1, 1]]
tf.reduce_sum(x) ==> 6
tf.reduce_sum(x, 0) ==> [2, 2, 2]
tf.reduce_sum(x, 1) ==> [3, 3]
tf.reduce_sum(x, 1, keep_dims=True) ==> [[3], [3]]
tf.reduce_sum(x, [0, 1]) ==> 6
```

> 可以理解为降维求和，
>
> input_tensor: 需要降维求和的张量
> axis: 需要降低的维度，如果参数无则表示将所有维度减去
> keep_dims: If true, retains reduced dimensions with length 1.
> name: A name for the operation (optional).
> reduction_indices: The old (deprecated) name for axis.

对于k维的，tf.reduce_sum(x, axis=k-1)的结果是对最里面一维所有元素进行求和。tf.reduce_sum(x, axis=k-2)是对倒数第二层里的向量对应的元素进行求和。tf.reduce_sum(x, axis=k-3)把倒数第三层的每个向量对应元素相加。[tf,reduce_sum](https://www.zhihu.com/question/51325408/answer/238082462)

### tf.argmax

```
tf.argmax(
    input,
    axis=None,
    name=None,
    dimension=None,
    output_type=tf.int64
)
```

>argmax返回的是索引值，返回每一行或者每一列的最大值的索引，当选择axis=1时。表示每一行的最大值，0表示每列的最大值索引;当axis为1，就是比每一行的值，返回最大值在该行的索引;[例子](https://www.jianshu.com/p/469789141af7)。



###  参考

[tf.matmul](https://www.tensorflow.org/api_docs/python/tf/matmul)

[tf.placeholder](https://blog.csdn.net/zj360202/article/details/70243127)

[tensorflow可译网学习指南](http://coyee.com/article/11637-tensorflow-get-started)

[tf,reduce_sum](https://www.zhihu.com/question/51325408/answer/238082462)
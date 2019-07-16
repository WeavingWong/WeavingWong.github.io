---
layout: post
date: 2019-07-16 23:36:15.000000000 +09:00
title: 使用XGBoost了解特征重要性以及特征筛选
categories: 编程实践
tags: 机器学习
---
**问题的提出：**

- 如何使用梯度增强算法计算特征重要性。
- 如何在XGBoost模型计算的Python中绘制特征重要性。
- 如何使用XGBoost计算的要素重要性来执行要素选择。

> 使用梯度增强的好处是，在构建增强树之后，检索每个属性的重要性分数是相对简单的

###  手动绘制重要性

```python
print(model.feature_importances_)

# plot 每一个特征的重要性，但是此处图中没有显示出特征的名称
pyplot.bar(range(len(model.feature_importances_)), model.feature_importances_)
pyplot.show()
```

### 利用自有库画图

```python
# plot feature importance using built-in function
from numpy import loadtxt
from xgboost import XGBClassifier
from xgboost import plot_importance
from matplotlib import pyplot

# load data
dataset = loadtxt('pima-indians-diabetes.data.csv', delimiter=",")

# split data into X and y
X = dataset[:,0:8]
y = dataset[:,8]
# fit model no training data
model = XGBClassifier()
model.fit(X, y)
# plot feature importance
plot_importance(model)
pyplot.show()
```

###  特征筛选

```python
# use feature importance for feature selection
from numpy import loadtxt
from numpy import sort
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.feature_selection import SelectFromModel

# load data
dataset = loadtxt('pima-indians-diabetes.data.csv', delimiter=",")

# split data into X and y
X = dataset[:,0:8]
Y = dataset[:,8]

# split data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.33, random_state=7)

# fit model on all training data
model = XGBClassifier()
model.fit(X_train, y_train)

# make predictions for test data and evaluate
y_pred = model.predict(X_test)
predictions = [round(value) for value in y_pred]
accuracy = accuracy_score(y_test, predictions)
print("Accuracy: %.2f%%" % (accuracy * 100.0))

# Fit model using each importance as a threshold
# 返回特征重要性排序
thresholds = sort(model.feature_importances_)
print(thresholds)
for thresh in thresholds:
    
	# select features using threshold，每次选择一个特征重要性分数，即排除一个特征
	selection = SelectFromModel(model, threshold=thresh, prefit=True)
	select_X_train = selection.transform(X_train)
    
	# train model
	selection_model = XGBClassifier()
	selection_model.fit(select_X_train, y_train)
    
	# eval model
	select_X_test = selection.transform(X_test)
	y_pred = selection_model.predict(select_X_test)
	predictions = [round(value) for value in y_pred]
	accuracy = accuracy_score(y_test, predictions)
	print("Thresh=%.3f, n=%d, Accuracy: %.2f%%" % (thresh, select_X_train.shape[1], accuracy*100.0))
```



### 参考文献：

https://machinelearningmastery.com/feature-importance-and-feature-selection-with-xgboost-in-python/
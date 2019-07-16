---
layout: post
date: 2019-07-16 23:35:15.000000000 +09:00
title: 探索性数据分析的步骤
categories: 编程实践
tags: 《机器学习实战》基于Scikit-Learn和tensorFlow
---
## 1.获取数据

```python
import os
import tarfile
from six.moves import urllib
DOWNLOAD_ROOT = "https://raw.githubusercontent.com/ageron/handson-ml/master/"
HOUSING_PATH = "datasets/housing"
HOUSING_URL = DOWNLOAD_ROOT + HOUSING_PATH + "/housing.tgz"
def fetch_housing_data(housing_url=HOUSING_URL, housing_path=HOUSING_PATH):
if not os.path.isdir(housing_path):
os.makedirs(housing_path)
tgz_path = os.path.join(housing_path, "housing.tgz")
urllib.request.urlretrieve(housing_url, tgz_path)
housing_tgz = tarfile.open(tgz_path)
housing_tgz.extractall(path=housing_path)
housing_tgz.close()
```

**加载数据**

```python
import pandas as pd
def load_housing_data(housing_path=HOUSING_PATH):
    csv_path = os.path.join(housing_path, "housing.csv")
    return pd.read_csv(csv_path)
```

**快速查看**

```python
df.head()
df.info()
df.describe() # 展示了数值属性的概括
```

![](/assets/images/EDA/desciribe.png)

**类别数据**

```python
housing["ocean_proximity"].value_counts() # 分析类别数据分布
```

**对整个数据集批量分析**

```python
%matplotlib inline   #  only in a Jupyter notebook
import matplotlib.pyplot as plt
housing.hist(bins=50, figsize=(20,15))
plt.show()
```

![](/assets/images/EDA/hist.png)

**创建测试集/验证集**

```python
方法1：
import numpy as np
def split_train_test(data, test_ratio):
    shuffled_indices = np.random.permutation(len(data))
    test_set_size = int(len(data) * test_ratio)
    test_indices = shuffled_indices[:test_set_size]
    train_indices = shuffled_indices[test_set_size:]
    return data.iloc[train_indices], data.iloc[test_indices]
    
# 使用方法如下：    
>>> train_set, test_set = split_train_test(housing, 0.2)
>>> print(len(train_set), "train +", len(test_set), "test")
16512 train + 4128 test

方法2：使用sklearn 自带方法
from sklearn.model_selection import train_test_split
train_set, test_set = train_test_split(housing, test_size=0.2,
                                       random_state=42)
```

**分层采样**

```python
# 根据业务逻辑 减小层数，即做分桶操作
#（新建一列仅用于分层，数值型数据不可分层参与训练）
housing["income_cat"] = np.ceil(housing["median_income"] / 1.5)
housing["income_cat"].where(housing["income_cat"] < 5, 5.0, inplace=True)

# 以下是分层采样
from sklearn.model_selection import StratifiedShuffleSplit
split = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
for train_index, test_index in split.split(housing, housing["income_cat"]):
    strat_train_set = housing.loc[train_index]
    strat_test_set = housing.loc[test_index]
    
# 总体数据集分布比例
housing["income_cat"].value_counts() / len(housing)

# 以下检查分层采样的测试集分布比例
strat_test_set ["income_cat"].value_counts() / len(strat_test_set )

# 删除用于分层的操作列
for set in (strat_train_set, strat_test_set):
    set.drop(["income_cat"], axis=1, inplace=True)
# 测试集通常被忽略，但实际是机器学习非常重要的一部分。
# 还有，生成测试集过程中的许多思路对于后面的交叉验证讨论是非常有帮助的
```

## 2.探索性数据分析、可视化、发现规律

**适合于散点图分析的数据类型**

- 地理经纬度分析（密度用颜色深浅区分alpha=0.1）

  ```python
  housing.plot(kind="scatter", x="longitude", y="latitude", alpha=0.1)
  ```

  ![](/assets/images/EDA/plot1.png)

- 地理-房价（颜色深浅）-人口（散点半径大小）

  ``` python
  housing.plot(kind="scatter", x="longitude", y="latitude", alpha=0.4,
      s=housing["population"]/100, label="population",
      c="median_house_value", cmap=plt.get_cmap("jet"), colorbar=True,)
  plt.legend()
  
  # 这张图# 说明房价和位置（比如，靠海）和人口密度联系密切，这点你可能早就知道。
  
  # 可以使用聚类算法来检测主要的聚集，用一个新的特征值测量聚集中心的距离。
  
  # 尽管北加州海岸区域的房价不是非常高，但离大海距离属性也可能很有用，
  
  # 所以这不是用一个简单的规则就可以定义的问题。
  ```

![](/assets/images/EDA/plot2.png)

**关联分析**

**方法一：**

```python
# 使用皮尔森相关系数 
corr_matrix = housing.corr()  
# 每个属性和房价中位数的关联度 
corr_matrix["median_house_value"].sort_values(ascending=False)
```

![](/assets/images/EDA/plot3.png)



**方法二：**检测属性间相关系数使用 Pandas 的scatter_matrix函数，它能画出每个数值属性对每个其它数值属性的图

```python
from pandas.tools.plotting import scatter_matrix
attributes = ["median_house_value", "median_income", "total_rooms",
              "housing_median_age"]
scatter_matrix(housing[attributes], figsize=(12, 8))
```



![](/assets/images/EDA/plot4.png)



单独画一张图：

```python
housing.plot(kind="scatter", x="median_income",y="median_house_value",              alpha=0.1) 
```

![](/assets/images/EDA/plot5.png)

这张图说明了几点。首先，相关性非常高；可以清晰地看到向上的趋势，并且数据点不是非常分散。第二，我们之前看到的最高价，清晰地呈现为一条位于 500000 美元的水平线。这张图也呈现了一些不是那么明显的直线：一条位于 450000 美元的直线，一条位于 350000 美元的直线，一条在 280000 美元的线，和一些更靠下的线。你可能希望去除对应的街区，以防止算法重复这些巧合。



**属性组合试验**

注意到一些属性具有长尾分布，因此你可能要将其进行转换（例如，计算其log对数）。当然，不同项目的处理方法各不相同，但大体思路是相似的。

给算法准备数据之前，你需要做的最后一件事是尝试多种属性组合。例如，如果你不知道某个街区有多少户，该街区的总房间数就没什么用。你真正需要的是每户有几个房间。相似的，总卧室数也不重要：你可能需要将其与房间数进行比较。每户的人口数也是一个有趣的属性组合。

```python
housing["rooms_per_household"] = housing["total_rooms"]/housing["households"]
housing["bedrooms_per_room"] = housing["total_bedrooms"]/housing["total_rooms"]
housing["population_per_household"]=housing["population"]/housing["households"]
```

```python
>>> corr_matrix = housing.corr()
>>> corr_matrix["median_house_value"].sort_values(ascending=False)
median_house_value          1.000000
median_income               0.687170
rooms_per_household         0.199343
total_rooms                 0.135231
housing_median_age          0.114220
households                  0.064702
total_bedrooms              0.047865
population_per_household   -0.021984
population                 -0.026699
longitude                  -0.047279
latitude                   -0.142826
bedrooms_per_room          -0.260070
Name: median_house_value, dtype: float64

```
看起来不错！与总房间数或卧室数相比，新的bedrooms_per_room属性与房价中位数的关联更强。显然，卧室数/总房间数的比例越低，房价就越高。每户的房间数也比街区的总房间数的更有信息，很明显，房屋越大，房价就越高。



**3.为算法准备数据**

还是先回到干净的训练集（通过再次复制strat_train_set），将预测量和标签分开，因为我们不想对预测量和目标值应用相同的转换（注意drop()创建了一份数据的备份，而不影响strat_train_set）：

```python
housing = strat_train_set.drop("median_house_value", axis=1)
housing_labels = strat_train_set["median_house_value"].copy()
```

**数据清洗**

**方法一：**

大多机器学习算法不能处理缺失的特征，因此先创建一些函数来处理特征缺失的问题。前面，你应该注意到了属性total_bedrooms有一些缺失值。有三个解决选项：

- 去掉对应的街区；
- 去掉整个属性；
- 进行赋值（0、平均值、中位数等等）。
```python
housing.dropna(subset=["total_bedrooms"])    #  选项1 去除部分
housing.drop("total_bedrooms", axis=1)       #  选项2 去除整列
median = housing["total_bedrooms"].median()
housing["total_bedrooms"].fillna(median)     #  选项3 进行填充
```

方法二：Scikit-Learn 提供了一个方便的类来处理缺失值：Imputer。下面是其使用方法：首先，需要创建一个Imputer实例，指定用某属性的中位数来替换该属性所有的缺失值：

```python
from sklearn.preprocessing import Imputer
imputer = Imputer(strategy="median")
# 去除类别特征

housing_num = housing.drop("ocean_proximity", axis=1)

# 训练出中位数

imputer.fit(housing_num)

# 开始填充转换，X为numpy数组

X = imputer.transform(housing_num)

# 将其重新转为DataFrame

housing_tr = pd.DataFrame(X, columns=housing_num.columns)
```

> Scikit-Learn 设计
>
> Scikit-Learn 设计的 API 设计的非常好。它的主要设计原则是：
>
> - 一致性：所有对象的接口一致且简单：
>
> - - 估计器（estimator）。任何可以基于数据集对一些参数进行估计的对象都被称为估计器（比如，imputer就是个估计器）。估计本身是通过fit()方法，只需要一个数据集作为参数（对于监督学习算法，需要两个数据集；第二个数据集包含标签）。任何其它用来指导估计过程的参数都被当做超参数（比如imputer的strategy），并且超参数要被设置成实例变量（通常通过构造器参数设置）。
>   - 转换器（transformer）。一些估计器（比如imputer）也可以转换数据集，这些估计器被称为转换器。API也是相当简单：转换是通过transform()方法，被转换的数据集作为参数。返回的是经过转换的数据集。转换过程依赖学习到的参数，比如imputer的例子。所有的转换都有一个便捷的方法fit_transform()，等同于调用fit()再transform()（但有时fit_transform()经过优化，运行的更快）。
>   - 预测器（predictor）。最后，一些估计器可以根据给出的数据集做预测，这些估计器称为预测器。例如，上一章的LinearRegression模型就是一个预测器：它根据一个国家的人均 GDP 预测生活满意度。预测器有一个predict()方法，可以用新实例的数据集做出相应的预测。预测器还有一个score()方法，可以根据测试集（和相应的标签，如果是监督学习算法的话）对预测进行衡器。
>
> - 可检验。所有估计器的超参数都可以通过实例的public变量直接访问（比如，imputer.strategy），并且所有估计器学习到的参数也可以通过在实例变量名后加下划线来访问（比如，imputer.statistics_）。
>
> - 类不可扩散。数据集被表示成 NumPy 数组或 SciPy 稀疏矩阵，而不是自制的类。超参数只是普通的 Python 字符串或数字。
>
> - 可组合。尽可能使用现存的模块。例如，用任意的转换器序列加上一个估计器，就可以做成一个流水线，后面会看到例子。
> - 合理的默认值。Scikit-Learn 给大多数参数提供了合理的默认值，很容易就能创建一个系统。
>



**处理文本和类别属性**

机器学习喜欢和数字打交道，所以需要对类别编码。

Scikit-Learn 为这个任务提供了一个转换器LabelEncoder：
```python
>>> from sklearn.preprocessing import LabelEncoder
>>> encoder = LabelEncoder()
>>> housing_cat = housing["ocean_proximity"]
>>> housing_cat_encoded = encoder.fit_transform(housing_cat)
>>> housing_cat_encoded
array([1, 1, 4, ..., 1, 0, 3])
译注:
在原书中使用LabelEncoder转换器来转换文本特征列的方式是错误的，
该转换器只能用来转换标签（正如其名）。
在这里使用LabelEncoder没有出错的原因是该数据只有一列文本特征值，
在有多个文本特征列的时候就会出错。应使用factorize()方法来进行操作：
housing_cat_encoded, housing_categories = housing_cat.factorize()
housing_cat_encoded[:10]
```
注意输出结果是一个 SciPy 稀疏矩阵，而不是 NumPy 数组。当类别属性有数千个分类时，这样非常有用。经过独热编码，我们得到了一个有数千列的矩阵，这个矩阵每行只有一个 1，其余都是 0。使用大量内存来存储这些 0 非常浪费，所以稀疏矩阵只存储非零元素的位置。你可以像一个 2D 数据那样进行使用，但是如果你真的想将其转变成一个（密集的）NumPy 数组，只需调用toarray()方法：
```python
housing_cat_1hot.toarray() 
```


为类别特征编码并独热的方式，本书中的作者存在一些错误，后续sklearn版本中已经修正

 

**自定义转换器（暂时还不知道怎么用）**

尽管 Scikit-Learn 提供了许多有用的转换器，你还是需要自己动手写转换器执行任务，比如自定义的清理操作，或属性组合。你需要让自制的转换器与 Scikit-Learn 组件（比如流水线）无缝衔接工作，因为 Scikit-Learn 是依赖鸭子类型的（而不是继承），你所需要做的是创建一个类并执行三个方法：fit()（返回self），transform()，和fit_transform()。通过添加TransformerMixin作为基类，可以很容易地得到最后一个。另外，如果你添加BaseEstimator作为基类（且构造器中避免使用*args和**kargs），你就能得到两个额外的方法（get_params()和set_params()），二者可以方便地进行超参数自动微调。例如，一个小转换器类添加了上面讨论的属性：

```python
from sklearn.base import BaseEstimator, TransformerMixin
rooms_ix, bedrooms_ix, population_ix, household_ix = 3, 4, 5, 6
class CombinedAttributesAdder(BaseEstimator, TransformerMixin):
def __init__(self, add_bedrooms_per_room = True): #  no *args or **kargs
self.add_bedrooms_per_room = add_bedrooms_per_room
def fit(self, X, y=None):
return self  #  nothing else to do
def transform(self, X, y=None):
rooms_per_household = X[:, rooms_ix] / X[:, household_ix]
population_per_household = X[:, population_ix] / X[:, household_ix]
if self.add_bedrooms_per_room:
bedrooms_per_room = X[:, bedrooms_ix] / X[:, rooms_ix]
return np.c_[X, rooms_per_household, population_per_household,
bedrooms_per_room]
else:
return np.c_[X, rooms_per_household, population_per_household]
attr_adder = CombinedAttributesAdder(add_bedrooms_per_room=False)
housing_extra_attribs = attr_adder.transform(housing.values)
```

在这个例子中，转换器有一个超参数add_bedrooms_per_room，默认设为True（提供一个合理的默认值很有帮助）。这个超参数可以让你方便地发现添加了这个属性是否对机器学习算法有帮助。更一般地，你可以为每个不能完全确保的数据准备步骤添加一个超参数。数据准备步骤越自动化，可以自动化的操作组合就越多，越容易发现更好用的组合（并能节省大量时间）。

**特征缩放**

有两种常见的方法可以让所有的属性有相同的量度：线性函数归一化（Min-Max scaling）和标准化（standardization）。

线性函数归一化（许多人称其为归一化（normalization））很简单：值被转变、重新缩放，直到范围变成 0 到 1。我们通过减去最小值，然后再除以最大值与最小值的差值，来进行归一化。Scikit-Learn 提供了一个转换器MinMaxScaler来实现这个功能。它有一个超参数feature_range，可以让你改变范围，如果不希望范围是 0 到 1。

标准化就很不同：首先减去平均值（所以标准化值的平均值总是 0），然后除以方差，使得到的分布具有单位方差。与归一化不同，标准化不会限定值到某个特定的范围，这对某些算法可能构成问题（比如，神经网络常需要输入值得范围是 0 到 1）。但是，标准化受到异常值的影响很小。例如，假设一个街区的收入中位数由于某种错误变成了100，归一化会将其它范围是 0 到 15 的值变为 0-0.15，但是标准化不会受什么影响。Scikit-Learn 提供了一个转换器StandardScaler来进行标准化。

**数据转换流水线**

标准化数据处理流程

```python
from sklearn.pipeline import FeatureUnion
num_attribs = list(housing_num)
cat_attribs = ["ocean_proximity"]
num_pipeline = Pipeline([
        ('selector', DataFrameSelector(num_attribs)),
        ('imputer', Imputer(strategy="median")),
        ('attribs_adder', CombinedAttributesAdder()),
        ('std_scaler', StandardScaler()),
    ])
cat_pipeline = Pipeline([
        ('selector', DataFrameSelector(cat_attribs)),
        ('label_binarizer', LabelBinarizer()),# LabelBinarizer存在错误
    ])
full_pipeline = FeatureUnion(transformer_list=[
        ("num_pipeline", num_pipeline),
        ("cat_pipeline", cat_pipeline),
    ])
```



# 你可以很简单地运行整个流水线：
```python
# 你可以很简单地运行整个流水线：
>>> housing_prepared = full_pipeline.fit_transform(housing)
>>> housing_prepared
array([[ 0.73225807, -0.67331551,  0.58426443, ...,  0.        ,
         0.        ,  0.        ],
       [-0.99102923,  1.63234656, -0.92655887, ...,  0.        ,
         0.        ,  0.        ],
       [...]
>>> housing_prepared.shape
(16513, 17)


# 每个子流水线都以一个选择转换器开始：通过选择对应的属性（数值或分类）、丢弃其它的，
# 来转换数据，并将输出DataFrame转变成一个 NumPy 数组。
# Scikit-Learn 没有工具来处理 PandasDataFrame，
# 因此我们需要写一个简单的自定义转换器来做这项工作：
from sklearn.base import BaseEstimator, TransformerMixin
class DataFrameSelector(BaseEstimator, TransformerMixin):
    def __init__(self, attribute_names):
        self.attribute_names = attribute_names
    def fit(self, X, y=None):
        return self
    def transform(self, X):
        return X[self.attribute_names].values
```
**选择并训练模型**

```python
# 开始训练
from sklearn.linear_model import LinearRegression
lin_reg = LinearRegression()
lin_reg.fit(housing_prepared, housing_labels)

# 做预测
>>> some_data = housing.iloc[:5]
>>> some_labels = housing_labels.iloc[:5]
>>> some_data_prepared = full_pipeline.transform(some_data)
>>> print("Predictions:\t", lin_reg.predict(some_data_prepared))
Predictions:     [ 303104.   44800.  308928.  294208.  368704.]
>>> print("Labels:\t\t", list(some_labels))
Labels:         [359400.0, 69700.0, 302100.0, 301300.0, 351900.0]

# 让我们使用 Scikit-Learn 的mean_squared_error函数，
# 用全部训练集来计算下这个回归模型的 RMSE：
>>> from sklearn.metrics import mean_squared_error
>>> housing_predictions = lin_reg.predict(housing_prepared)
>>> lin_mse = mean_squared_error(housing_labels, housing_predictions)
>>> lin_rmse = np.sqrt(lin_mse)
>>> lin_rmse
68628.413493824875


均方根误差 很大，效果很差，尝试决策树：
from sklearn.tree import DecisionTreeRegressor
tree_reg = DecisionTreeRegressor()
tree_reg.fit(housing_prepared, housing_labels)

# 做预测与验证
>>> housing_predictions = tree_reg.predict(housing_prepared)
>>> tree_mse = mean_squared_error(housing_labels, housing_predictions)
>>> tree_rmse = np.sqrt(tree_mse)
>>> tree_rmse
0.0
```

# 做预测与验证
>>> housing_predictions = tree_reg.predict(housing_prepared)
>>> tree_mse = mean_squared_error(housing_labels, housing_predictions)
>>> tree_rmse = np.sqrt(tree_mse)
>>> tree_rmse
0.0

等一下，发生了什么？没有误差？这个模型可能是绝对完美的吗？当然，更大可能性是这个模型严重过拟合数据。如何确定呢？如前所述，直到你准备运行一个具备足够信心的模型，都不要碰测试集，因此你需要使用训练集的部分数据来做训练，用一部分来做模型验证。

（上述训练采用了全部数据，预测训练数据，导致过拟合）

**使用交叉验证做更佳的评估**

评估决策树模型的一种方法是用函数train_test_split来分割训练集，得到一个更小的训练集和一个验证集，然后用更小的训练集来训练模型，用验证集来评估。这需要一定工作量，并不难而且也可行。

**另一种更好的方法**是使用 Scikit-Learn 的交叉验证功能。下面的代码采用了 K 折交叉验证（K-fold cross-validation）：它随机地将训练集分成十个不同的子集，成为“折”，然后训练评估决策树模型 10 次，每次选一个不用的折来做评估，用其它 9 个来做训练。结果是一个包含 10 个评分的数组：

```python
from sklearn.model_selection import cross_val_score
scores = cross_val_score(tree_reg, housing_prepared, housing_labels,
                         scoring="neg_mean_squared_error", cv=10)
rmse_scores = np.sqrt(-scores)

# 警告：Scikit-Learn 交叉验证功能期望的是效用函数（越大越好）而不是损失函数（越低越好），因此得分函数实际上与 MSE 相反（即负值），这就是为什么前面的代码在计算平方根之前先计算-scores。


>>> def display_scores(scores):
...     print("Scores:", scores)
...     print("Mean:", scores.mean())
...     print("Standard deviation:", scores.std())
...
>>> display_scores(tree_rmse_scores)
Scores: [ 74678.4916885   64766.2398337   69632.86942005  69166.67693232
          71486.76507766  73321.65695983  71860.04741226  71086.32691692
          76934.2726093   69060.93319262]
Mean: 71199.4280043
Standard deviation: 3202.70522793
# 效果更差了

# 重新用线性回归做交叉验证试试：
>>> lin_scores = cross_val_score(lin_reg, housing_prepared, housing_labels,
...                              scoring="neg_mean_squared_error", cv=10)
...
>>> lin_rmse_scores = np.sqrt(-lin_scores)
>>> display_scores(lin_rmse_scores)
Scores: [ 70423.5893262   65804.84913139  66620.84314068  72510.11362141
          66414.74423281  71958.89083606  67624.90198297  67825.36117664
          72512.36533141  68028.11688067]
Mean: 68972.377566
Standard deviation: 2493.98819069
# 判断没错：决策树模型过拟合很严重，它的性能比线性回归模型还差。

# 试试随机森林：
>>> from sklearn.ensemble import RandomForestRegressor
>>> forest_reg = RandomForestRegressor()
>>> forest_reg.fit(housing_prepared, housing_labels)
>>> [...] # 省略重复代码
>>> forest_rmse
22542.396440343684
>>> display_scores(forest_rmse_scores)
Scores: [ 53789.2879722   50256.19806622  52521.55342602  53237.44937943
          52428.82176158  55854.61222549  52158.02291609  50093.66125649
          53240.80406125  52761.50852822]
Mean: 52634.1919593
Standard deviation: 1576.20472269
```
> 提示：你要保存每个试验过的模型，以便后续可以再用。要确保有超参数和训练参数，以及交叉验证评分，和实际的预测值。这可以让你比较不同类型模型的评分，还可以比较误差种类。你可以用 Python 的模块pickle，非常方便地保存 Scikit-Learn 模型，或使用sklearn.externals.joblib，后者序列化大 NumPy 数组更有效率：

```python
from sklearn.externals import joblib
joblib.dump(my_model, "my_model.pkl")
#  然后
my_model_loaded = joblib.load("my_model.pkl")
```
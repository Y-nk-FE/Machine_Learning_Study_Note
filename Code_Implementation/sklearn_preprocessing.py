# 关于sklearn_preprocessing 的数据预处理
# 1. 标准化
# 2. 非线性变换
# 3. 正则化（ Normalizer）
# 4. 编码分类特征
# 5. 离散化
# 6. 缺失值处理（Imputer）
# 7. 生成多项式特征（PolynomialFeatures）
# 8. 自定义转换器（FunctionTransformer）
# 导入panda模块，用于数据的导入和读取
import pandas as pd
# 导入sklearn库中的preprocessing模块,用于对数据做预处理
from sklearn import preprocessing
import numpy as np

# 读取文件，提取信息
file_path = r'D:/sample/Machine_Learning_Study_Note/Code_Implementation/Related_Data/test_2.xlsx'
df = pd.read_excel(file_path, 0)
# 使用.values方法将表格中的数据存贮在数组中
data = df.values
print(data)
print('-----')

# 1.标准化：去均值，方差规模化
# sklearn.preprocessing中提供了scale方法,对传入参数的所有数据进行标准化,使数据服从标准正态分布
data_scale = preprocessing.scale(data)
print(data_scale)
print('-----')

# 计算均值&方差进行验证
ave = 0
for i in range(0, 7):
    for j in range(0, 5):
        ave = ave + data_scale[i, j]
print(data_scale[7, 5])
print('ave:', ave)
print('-----')

x = np.array([[1, 2, 3],
             [4, 5, 6]])
x_scale = preprocessing.scale(x)
print(x)
print(x_scale)
print('-----')

# .mean(axis)-取均值
# .std(axis)-去方差
# axis = 0时，对每一列操作，axis = 1时，对每一行操作
print(x_scale.mean(0))
print(x_scale.mean(1))
print(x_scale.std(0))
print(x_scale.mean(1))
print('-----')



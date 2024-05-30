# 五折交叉验证
# 导入pandas库用于读取Excel表格中的数据
import pandas as pd
# 导入sklearn库中的preprocessing模块，对数据进行预处理
# preprocessing.scale( ) 函数主要是对数据调整为均值为0，方差为1的正态分布，也就是高斯分布，属于数据标准化。
from sklearn import preprocessing


def data_pre_process():
    """"数据预处理函数 对Excel表格中的数据进行读取 提取特征值feature并标准化"""
    file_path = r'D:/sample/Machine_Learning_Study_Note/Code_Implementation/Related_Data/test_1.xlsx'
    original_data = pd.read_excel(file_path, 0)

    # 只提取表中的数据，不提取表头
    data = original_data.values

    # 去标签取特征值
    feature = data[:, 0:6]

    # 特征值数据标准化
    feature1 = preprocessing.scale(feature)
    print(feature1)

    # preprocessing.normalize( ) 函数可以实现L1正则化和L2正则化，也是很常用的方法
    feature2 = preprocessing.normalize(feature)
    print(feature2)


data_pre_process()

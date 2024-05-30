import numpy as np


def normalize(features):
    # astype函数是NumPy库中可以将一个NumPy数组转换成另一种数据类型的函数
    features_normalized = np.copy(features).astype(float)

    # 计算均值,np.mean(a, axis, dtype, out，keepdims)函数功能：求取均值,返回一个实数
    # axis=0，计算每一列的均值
    features_mean = np.mean(features, 0)

    # 计算标准差std，np.std(a, axis=None, dtype=None, out=None, ddof=0, keepdims=)求标准差的
    # axis=0时，表示求每一列标准差
    features_deviation = np.std(features, 0)

    # 标准化操作
    if features.shape[0] > 1:
        features_normalized -= features_mean

    # 防止除以0
    features_deviation[features_deviation == 0] = 1
    features_normalized /= features_deviation

    return features_normalized, features_mean, features_deviation

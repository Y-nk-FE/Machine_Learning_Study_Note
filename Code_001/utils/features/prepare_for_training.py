import numpy as np
from utils.features import normalize
from utils.features.generate_polynomials import generate_polynomials
from utils.features.generate_sinusoids import generate_sinusoids


def prepare_for_training(data, polynomial_degree=0, sinusoid_degree=0, normalize_data=True):
    """
    数据预处理
    :param data:
    :param polynomial_degree:
    :param sinusoid_degree:
    :param normalize_data:
    :return:
    """
    # 计算样本总数,shape[0]读取矩阵第一维度的长度,即数组的行数
    num_example = data.shape[0]
    data_processed = np.copy(data)

    # 预处理
    features_mean = 0
    features_deviation = 0  # deviation偏差
    data_normalized = data_processed

    # 是否需要对数据做标准化处理
    if normalize_data:
        (data_normalized,
         features_mean,
         features_deviation
         ) = normalize.normalize(data_processed)

    data_processed = data_normalized

    # 特征变换sinusoidal
    if sinusoid_degree > 0:
        sinusoids = generate_sinusoids(data_normalized, sinusoid_degree)
        data_processed = np.concatenate((data_processed, sinusoids), axis=1)

    # 特征变换polynomial
    if polynomial_degree>0:
        polynomials = generate_polynomials(data_normalized, polynomial_degree, normalize_data)
        data_processed = np.concatenate((data_processed, polynomials), axis=1)

    # 加一列1
    data_processed = np.hstack((np.ones((num_example, 1)), data_processed))
    return data_processed, features_mean, features_deviation

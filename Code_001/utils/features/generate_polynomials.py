"""
polynomials-多项式
generate polynomials - 生成多项式
"""
import numpy as np
from utils.features.normalize import normalize


def generate_polynomials(dataset, polynomial_degree, normalize_data=False):
    """
    变换方法：
    x1,x2,x1^2,x2^2,x1*x2,x1*x2^2,etc.
    :param dataset:
    :param polynomial_degree:
    :param normalize_data:
    :return:
    """

    # np.array_split()-对数组进行不均等划分,axis-分割方向，一维 二维 三维……
    features_split = np.array_split(dataset, 2, axis=1)
    dataset_1 = features_split[0]
    dataset_2 = features_split[1]

    (num_examples_1, num_features_1) = dataset_1.shape
    (num_examples_2, num_features_2) = dataset_2.shape

    # raise ValueError 自行引发异常
    if num_examples_1 != num_examples_2:
        raise ValueError('Can not generate polynomials for two sets with different number')

    if num_features_1 == 0 and num_features_2 == 0:
        raise ValueError('Can not generate polynomials for two sets with no columns')

    if num_features_1 == 0:
        dataset_1 = dataset_2
    elif num_features_2 == 0:
        dataset_2 = dataset_1

    num_features = num_features_1 if num_examples_1 < num_examples_2 else num_features_2
    dataset_1 = dataset_1[:, :num_features]
    dataset_2 = dataset_2[:, :num_features]

    polynomials = np.empty((num_examples_1, 0))

    for i in range(1, polynomial_degree + 1):
        for j in range(i + 1):
            polynomial_feature = (dataset_1 ** (i - j)) * (dataset_2 ** j)
            polynomials = np.concatenate((polynomials, polynomial_feature), axis=1)

    if normalize_data:
        polynomials = normalize(polynomials)[0]

    return polynomials

import numpy as np


def generate_sinusoids(dataset, sinusoid_degree):
    """
    生成正弦曲线
    :param dataset:
    :param sinusoid_degree:
    :return:
    """
    # 获取数据集的个数
    num_example = dataset.shape[0]
    sinusoids = np.empty((num_example, 0))

    for degree in range(1, sinusoid_degree + 1):
        sinusoids_features = np.sin(degree * dataset)
        sinusoids = np.concatenate((sinusoids, sinusoids_features), axis=1)

    return sinusoids

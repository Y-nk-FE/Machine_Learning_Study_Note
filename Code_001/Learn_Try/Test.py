import numpy as np
import random
import matplotlib.pyplot as plt


# sigmod函数，将值量化到0-1
def sigmoid(x):
    return 1.0 / (1 + np.exp(-x))


# 梯度下降算法
def gradientDescent(x, y, theta, alpha, m, numIteration):
    """
    :param x: 输入实例
    :param y: 分类标签
    :param theta: 学习的参数
    :param alpha: 学习率
    :param m: 实例个数,即样本个数
    :param numIteration: 迭代次数
    :return: 学习参数theta
    """
    xTrans = x.transpose()  # 矩阵的转置
    J = []  # 存储损失的列表，方便绘图
    for i in range(0, numIteration):
        hypothsis = sigmoid(np.dot(x, theta))  # 量化到0-1
        loss = hypothsis - y  # 计算误差
        cost = np.sum(loss ** 2) / (2 * m)  # 计算损失
        J.append(cost)  # 将损失存入列表
        # print("Iteration %d / Cost:%f" % (i, cost))
        gradient = np.dot(xTrans, loss) / m
        theta = theta - alpha * gradient  # 更新梯度
    plt.plot(J)  # 可视化损失变化
    plt.show()
    return theta


# 创建数据，用作测试
def generate_data(num_sample, num_features, variance):
    """
       :param num_sample: 样本数目
       :param num_features: 样本特征数目
       :param variance: 方差
       :return 创建的数据 x,y
       """
    # 实例（行数）、偏向、方差
    # 自变量
    x = np.zeros(shape=(num_sample, num_features))  # 初始化大小为(num_sample,num_feature)全零元素矩阵
    # 因变量
    y = np.zeros(shape=num_sample)  # 归类标签,0-1分类
    y_list = [0, 1]  # 标签列表y的取值
    for i in range(0, num_sample):
        x[i][0] = random.uniform(0, 1) * variance  # 创建X的特征对
        # x[i][1] = (i + num_features) + random.uniform(0, 1) * variance
        random.shuffle(y_list)  # 给X特征对附上随机的0 1 标签
        y[i] = y_list[0]
    return x, y


# numIteration = 100000
# alpha = 0.0005
# theta = np.ones(n)  # 初始化theta
# theta = gradientDescent(x, y, theta, alpha, m, numIteration)
# print(theta)  # output [-0.14718538  0.00381781]

if __name__ == '__main__':
    # 生成数据，用于回归
    x, y = generate_data(100, 1, 5)
    print('x:\n', x, '\n', x.shape)
    print('y:\n', y, '\n', y.shape)

    plt.scatter(x, y, label='数据')
    plt.legend()
    plt.show()

    # x和y的维度
    m, n = np.shape(x)
    n_y = np.shape(y)

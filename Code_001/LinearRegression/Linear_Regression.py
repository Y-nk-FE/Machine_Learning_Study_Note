import time

import numpy as np

from utils.features.prepare_for_training import prepare_for_training


class LinearRegression:
    def __init__(self, data, labels, polynomial_degree=0, sinusoid_degree=0, normalize_data=True):
        """
            1.对数据进行预处理操作
            2.先得到所有的特征个数
            3.初始化参数矩阵
            """
        # 数据预处理(标准化、归一化、去中心化)
        (data_processed,
         features_mean,
         features_deviation
         ) = prepare_for_training(data, polynomial_degree, sinusoid_degree, normalize_data)

        self.data = data_processed
        self.labels = labels
        self.features_mean = features_mean
        self.features_deviation = features_deviation
        self.polynomial_degree = polynomial_degree
        self.sinusoid_degree = sinusoid_degree
        self.normalize_data = normalize_data

        # 先预处理后确定参数theta大小
        num_features = self.data.shape[1]           # 列数，shape[1]=列数，shape[0]=行数
        self.theta = np.zeros((num_features, 1))    # 构造theta，shape： num_features×1 的矩阵

    def train(self, alpha, num_iterations=500):
        """
        训练模块，执行梯度下降模块
        :param alpha:学习率
        :param num_iterations:迭代次数，默认值为500
        :return:
        """
        cost_history = self.gradient_descent(alpha, num_iterations)
        return self.theta, cost_history

    def gradient_descent(self, alpha, num_iterations):
        """
        梯度下降模块，会迭代num_iterations次
        :param alpha:学习率
        :param num_iterations:迭代次数
        :return:
        """
        cost_history = []     # 记录每一次的损失值
        for i in range(num_iterations):
            self.gradient_step(alpha)
            cost_history.append(self.cost_function(self.data, self.labels))
        return cost_history

    def gradient_step(self, alpha):
        """
        梯度下降参数更新计算方法，注意是矩阵运算
        :param alpha:
        :return:
        """
        num_example = self.data.shape[0]
        prediction = LinearRegression.hypothesis(self.data, self.theta)
        delta = prediction - self.labels
        theta = self.theta
        theta = theta - alpha * (1/num_example) * (np.dot(delta.T, self.data)).T  # n行1列
        self.theta = theta

    def cost_function(self, data, labels):
        """
        损失计算方法
        :param data:
        :param labels:
        :return:
        """
        num_examples = data.shape[0]  # 总的样本的个数
        delta = LinearRegression.hypothesis(self.data, self.theta) - labels
        cost = (1/2) * np.dot(delta.T, data)/num_examples
        return cost[0][0]

    @staticmethod
    def hypothesis(data, theta):
        """
        预测值的计算
        :param data:
        :param theta:
        :return:
        """
        # np.dot(A, B) 矩阵乘法运算
        predictions = np.dot(data, theta)
        return predictions

    def get_cost(self, data, labels):
        # 数据预处理
        data_processed = \
            prepare_for_training(data, self.polynomial_degree, self.sinusoid_degree, self.normalize_data)[0]
        return self.cost_function(data_processed, labels)

    def predict(self, data):
        """
        用训练好的参数模型，预测的奥的回归值结果
        :param data:
        :return:
        """
        data_processed = \
            prepare_for_training(data, self.polynomial_degree, self.sinusoid_degree, self.normalize_data)[0]

        predictions = LinearRegression.hypothesis(data_processed, self.theta)

        return predictions


if __name__ == '__main__':
    begin = time.time()

    end = time.time()
    print('-' * 50, '\n', 'Running Time:', end-begin, 's')

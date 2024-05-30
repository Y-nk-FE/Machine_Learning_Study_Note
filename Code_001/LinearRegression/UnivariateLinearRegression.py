"""单特征回归"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from Linear_Regression import LinearRegression

data_filepath = "../data/World_Happiness_Data_2019.csv"
data = pd.read_csv(data_filepath)

# 训练数据和测试数据
train_data = data.sample(frac=0.8)
test_data = data.drop(train_data.index)

# 标签名
input_param_name = 'GDP per capita'
out_param_name = 'Score'

# [[]]行和列，矩阵要求matrix，必须是二维数组
x_train = train_data[[input_param_name]].values
y_train = train_data[[out_param_name]].values

x_test = test_data[input_param_name].values
y_test = test_data[out_param_name].values

plt.scatter(x_train, y_train, label='Train_data')
plt.scatter(x_test, y_test, label='Test_data')
plt.xlabel(input_param_name)
plt.ylabel(out_param_name)
plt.title('Happiness')
plt.legend()  # 显示label的内容
plt.show()

# 定义参数
num_iterations = 500
learning_rate = 0.01

# 初始化操作+数据预处理
linear_regression = LinearRegression(data=x_train, labels=y_train)
# 训练模型
(theta, cost_history) = linear_regression.train(alpha=learning_rate, num_iterations=num_iterations)

print('开始时的损失：', cost_history[0])       # 第一次
print('训练后的损失：', cost_history[-1])      # 最后一次
print('theta：\n', linear_regression.theta)    # theta

plt.plot(range(num_iterations), cost_history)
plt.xlabel('Iter')
plt.ylabel('cost')
plt.title('GD')
plt.show()

# 测试
prediction_num = 100
x_predictions = np.linspace(x_train.min(), x_train.max(), prediction_num).reshape(prediction_num,1)
y_predictions = linear_regression.predict(x_predictions)

plt.scatter(x_train, y_train, label='Train data')
plt.scatter(x_test, y_test, label='test data')
plt.plot(x_predictions, y_predictions, 'r', label='Prediction')
plt.xlabel(input_param_name)
plt.ylabel(out_param_name)
plt.title('Happy')
plt.show()

"""非线性回归"""
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from LinearRegression.Linear_Regression import LinearRegression

# 读取文件
data_filepath = "../data/Data Yield Curve.csv"
data = pd.read_csv(data_filepath)

# define the labels
input_param_name = 'number'
output_param_name = 'y'

# transform into two-dimension array,i.e. matrix
# train dataset
x = data[[input_param_name]].values
y = data[output_param_name].values.reshape((data.shape[0], 1))

data.head(10)

# plot
# plt.legend()显示label的内容
plt.scatter(x, y, label='Train_data')
plt.xlabel(input_param_name)
plt.ylabel(output_param_name)
plt.legend()
plt.show()

# configure the param
num_iterations = 500000
learning_rate = 0.0001
polynomial_degree = 15
sinusoid_degree = 15
normalize_data = True

linear_regression = LinearRegression(x, y, polynomial_degree, sinusoid_degree, normalize_data)

(theta, cost_history) = linear_regression.train(learning_rate, num_iterations)

print('开始损失:{:.2f}'.format(cost_history[0]))
print('损失结束:{:.2f}'.format(cost_history[-1]))

theta_table = pd.DataFrame({'Model Parameters': theta.flatten()})

plt.plot(range(num_iterations), cost_history)
plt.xlabel('Iterations')
plt.ylabel('Cost')
plt.title('Gradient Descent Progress')
plt.show()

predictions_num = 1000
x_predictions = np.linspace(x.min(), x.max(), predictions_num).reshape(predictions_num, 1)
y_predictions = linear_regression.predict(x_predictions)

plt.scatter(x, y, label='Training Dataset')
plt.plot(x_predictions, y_predictions, 'r', label='Prediction')
plt.xlabel(input_param_name)
plt.ylabel(output_param_name)
plt.title('Happy')
plt.show()

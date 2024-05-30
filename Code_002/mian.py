import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC


def test():
    print("Successful")


def reference_1():
    # 加载鸢尾花数据集
    iris = datasets.load_iris()

    # 获取特征和标签
    X = iris.data
    y = iris.target

    print(X.shape)

    # 数据集切分
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)

    # SVM分类器模型训练
    svm_model = SVC(kernel='linear', C=1.0)
    svm_model.fit(X_train, y_train)

    # 预测测试数据集
    predicted_y = svm_model.predict(X_test)

    # 打印预测结果及模型评分
    print("Predicted labels: \n", predicted_y, '\n---------------------')
    print("Accuracy score: \n", svm_model.score(X_test, y_test), '\n---------------------')


def plot_iris():
    # 加载鸢尾花数据集
    iris = datasets.load_iris()

    # 获取特征和标签
    x = iris.data
    y = iris.target

    # 分离特征
    x_1 = x[:, 0]
    x_2 = x[:, 1]
    x_3 = x[:, 2]
    x_4 = x[:, 3]

    # 绘图
    plt.scatter(x_1, y, label='X1')
    plt.scatter(x_2, y, label='X2')
    plt.scatter(x_3, y, label='X3')
    plt.scatter(x_4, y, label='X4')
    plt.legend()
    plt.show()


if __name__ == '__main__':
    test()
    plot_iris()


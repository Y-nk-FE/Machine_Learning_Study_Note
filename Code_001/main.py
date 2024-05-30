# 这是一个示例 Python 脚本。

# 按 Shift+F10 执行或将其替换为您的代码。
# 按 双击 Shift 在所有地方搜索类、文件、工具窗口、操作和设置。
import numpy as np


def print_hi(name):
    # 在下面的代码行中使用断点来调试脚本。
    print(f'Hi, {name}')  # 按 Ctrl+F8 切换断点。


def func_1():
    a = np.ones((5, 1))
    print(a)
    b = np.zeros((5, 3))
    print(b)

    c = np.hstack((a, b))
    print(c)


def func_2():
    a = np.array([[1, 2],
                  [1, 2],
                  [1, 2]])
    print(a, type(a), a.shape)

    b = np.array([[1, 1, 1],
                  [2, 2, 2]])

    c = np.dot(b, a)
    print(c)

# 按间距中的绿色按钮以运行脚本。
if __name__ == '__main__':
    func_2()

# 访问 https://www.jetbrains.com/help/pycharm/ 获取 PyCharm 帮助

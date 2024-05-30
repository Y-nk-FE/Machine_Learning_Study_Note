# 导入panda模块，用于数据的导入和读取
import pandas as pd
# 导入sklearn库中的preprocessing模块
from sklearn import preprocessing

# 读取文件，提取信息
file_path = r'D:/sample/Machine_Learning_Study_Note/Code_Implementation/Related_Data/test_1.xlsx'
df = pd.read_excel(file_path, 0)
# 使用.values方法将表格中的数据存贮在数组中
data = df.values
print(data)

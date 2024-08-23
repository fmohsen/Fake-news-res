import pandas as pd

# 读取CSV文件
df = pd.read_csv('data.csv')

# 计算值为0和1的数量
counts = df['Label'].value_counts()

# 输出结果
print('值为0的数量:', counts[0])
print('值为1的数量:', counts[1])

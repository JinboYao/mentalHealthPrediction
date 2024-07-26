import pandas as pd

# 加载数据
df = pd.read_csv(r'/dataset/Processed Data.csv')

# 计算分割点
train_size = int(len(df) * 0.7)
val_size = int(len(df) * 0.2)
test_size = len(df) - train_size - val_size

# 打乱数据
df = df.sample(frac=1).reset_index(drop=True)

# 切割数据集
train_df = df.iloc[:train_size]
val_df = df.iloc[train_size:train_size + val_size]
test_df = df.iloc[train_size + val_size:]

# 保存为CSV文件
train_df.to_csv(r'/Users/yaojinbo/Desktop/mentalHealthPrediction/dataset/train.csv', index=False)
val_df.to_csv(r'/Users/yaojinbo/Desktop/mentalHealthPrediction/dataset/val.csv', index=False)
test_df.to_csv(r'/Users/yaojinbo/Desktop/mentalHealthPrediction/dataset/test.csv', index=False)

print("Data split into train, validation, and test sets and saved as CSV files.")

# 导入所需库
import pandas as pd

# 加载数据
file_path = '../dataset/Processed Data.csv'  # 或者您之前所使用的具体路径
data = pd.read_csv(file_path)

# 转换前确认数据类型分布
print("Data types before conversion:")
print(data['statement'].apply(type).value_counts())

# 确保所有条目都是字符串类型
data['statement'] = data['statement'].astype(str)

# 转换后再次确认数据类型分布
print("Data types after conversion:")
print(data['statement'].apply(type).value_counts())

import pandas as pd
import numpy as np
from scipy.stats import spearmanr
import matplotlib.pyplot as plt

# 读取并预处理数据
data_x1 = np.load("D:\\Desktop\\20241029\\Experiments-using-PHM2010dataset-main\\features\\data_x1.npy")
data_x4 = np.load("D:\\Desktop\\20241029\\Experiments-using-PHM2010dataset-main\\features\\data_x4.npy")
data_x6 = np.load("D:\\Desktop\\20241029\\Experiments-using-PHM2010dataset-main\\features\\data_x6.npy")
data_y1 = np.load("D:\\Desktop\\20241029\\Experiments-using-PHM2010dataset-main\\features\\data_y1.npy")
data_y4 = np.load("D:\\Desktop\\20241029\\Experiments-using-PHM2010dataset-main\\features\\data_y4.npy")
data_y6 = np.load("D:\\Desktop\\20241029\\Experiments-using-PHM2010dataset-main\\features\\data_y6.npy")

# 定义标准化函数
def normalization(data):
    mu = np.mean(data, axis=0)
    sigma = np.std(data, axis=0)
    return (data - mu) / sigma

# 对所有特征进行标准化
def norm_all(data):
    d = np.empty((data.shape[0], data.shape[1], data.shape[2]))
    for i in range(data.shape[1]):
        data1 = data[:, i, :]
        for j in range(data1.shape[0]):
            data2 = data1[j, :]
            d[j, i, :] = normalization(data2)
    return d

# 对标签进行归一化
def normal_label(data):
    minVals = data.min(0)
    maxVals = data.max(0)
    ranges = maxVals - minVals
    normData = np.zeros(np.shape(data))
    m = data.shape[0]
    normData = data - np.tile(minVals, (m, 1))
    normData = normData / np.tile(ranges, (m, 1))
    return normData[0]

# 应用预处理函数
data_x1 = norm_all(data_x1)
data_x4 = norm_all(data_x4)
data_x6 = norm_all(data_x6)
data_y1 = normal_label(data_y1)
data_y4 = normal_label(data_y4)
data_y6 = normal_label(data_y6)

# 合并训练数据
train_x = np.append(np.append(data_x1, data_x4, axis=0), data_x6, axis=0)
train_y = np.append(np.append(data_y1, data_y4, axis=0), data_y6, axis=0)

# 将数据转换为DataFrame格式
num_samples, num_channels, num_features = train_x.shape
train_x_reshaped = train_x.reshape(num_samples, num_features * num_channels)
train_y_reshaped = np.repeat(train_y, 1)

df = pd.DataFrame(train_x_reshaped, columns=[f'feature_{i}' for i in range(num_features * num_channels)])
df['target'] = train_y_reshaped

# 计算斯皮尔曼相关系数并按单调性排序
def spearman_rank_correlation_sort(df, target_column):
    spearman_results = []
    for column in df.columns:
        if column != target_column:
            correlation, p_value = spearmanr(df[column], df[target_column])
            spearman_results.append({
                'feature': column,
                'spearman_correlation': correlation,
                'p_value': p_value
            })

    # 按相关系数绝对值排序
    sorted_features = sorted(spearman_results, key=lambda x: abs(x['spearman_correlation']), reverse=True)
    sorted_df = pd.DataFrame(sorted_features)

    return sorted_df

# 输出排序后的特征
sorted_features_df = spearman_rank_correlation_sort(df, target_column='target')
print("特征按斯皮尔曼等级相关性排序：")
print(sorted_features_df)

# 取相关性的绝对值
sorted_features_df['spearman_correlation'] = sorted_features_df['spearman_correlation'].abs()

# 绘制斯皮尔曼相关性条形图
plt.figure(figsize=(12, 6))

# 使用垂直条形图显示相关性
plt.bar(sorted_features_df['feature'], sorted_features_df['spearman_correlation'], color='blue', alpha=0.7, edgecolor='black')

# 添加标题和标签
plt.title('Spearman Correlation of Features with Target Variable (Absolute Values)')
plt.xlabel('Features')
plt.ylabel('Absolute Spearman Correlation')

# 自动调整边距
plt.xticks(rotation=90)  # 旋转x轴标签，以便更好地显示
plt.tight_layout()

# 显示图形
plt.show()

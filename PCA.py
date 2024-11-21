import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

# 加载数据
data_x1 = np.load("D:\\Desktop\\20241029\\Experiments-using-PHM2010dataset-main\\features\\data_x1.npy")
data_x4 = np.load("D:\\Desktop\\20241029\\Experiments-using-PHM2010dataset-main\\features\\data_x4.npy")
data_x6 = np.load("D:\\Desktop\\20241029\\Experiments-using-PHM2010dataset-main\\features\\data_x6.npy")

# 重塑数据形状
num_samples, num_channels, num_features = data_x1.shape
data_x1 = data_x1.reshape(num_samples, num_features * num_channels)
data_x4 = data_x4.reshape(num_samples, num_features * num_channels)
data_x6 = data_x6.reshape(num_samples, num_features * num_channels)

# 定义三次一阶指数平滑函数
def triple_exponential_smoothing(data, alpha):
    smoothed_1 = np.zeros_like(data)
    smoothed_1[0] = data[0]
    for t in range(1, len(data)):
        smoothed_1[t] = alpha * data[t] + (1 - alpha) * smoothed_1[t - 1]

    smoothed_2 = np.zeros_like(smoothed_1)
    smoothed_2[0] = smoothed_1[0]
    for t in range(1, len(smoothed_1)):
        smoothed_2[t] = alpha * smoothed_1[t] + (1 - alpha) * smoothed_2[t - 1]

    smoothed_3 = np.zeros_like(smoothed_2)
    smoothed_3[0] = smoothed_2[0]
    for t in range(1, len(smoothed_2)):
        smoothed_3[t] = alpha * smoothed_2[t] + (1 - alpha) * smoothed_3[t - 1]

    return smoothed_3

# 定义处理函数
def process_data(data, alpha, index_name):
    # 1. 标准化
    scaler = StandardScaler()
    X_norm = scaler.fit_transform(data)

    # 2. 进行 PCA
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X_norm)


    cov_matrix = np.cov(X_pca, rowvar=False)

    # 3. 计算特征值和特征向量
    eigenvalues, eigenvectors = np.linalg.eig(cov_matrix)

    # 4. 找到最大方差对应的特征向量
    max_index = np.argmax(eigenvalues)
    principal_vector = eigenvectors[:, max_index]

    # 5. 用最大特征值对应的特征向量对标准化后的矩阵重构
    X_reconstructed = X_pca @ principal_vector[:, np.newaxis]

    # 4. 对重构后的数据进行三次一阶指数平滑
    smoothed_data = triple_exponential_smoothing(X_reconstructed.flatten(), alpha)

    # 保存平滑后的健康指数为 NumPy 文件
    np.save(f'virtual_health_index_{index_name}.npy', smoothed_data)
    print(f"健康指数已保存到 'virtual_health_index_{index_name}.npy'。")

    # 可视化结果
    plt.figure(figsize=(12, 6))
    plt.plot(smoothed_data, label=f'Smoothed Virtual Health Index (HI_{index_name})', color='blue')
    plt.title(f'Virtual Health Index of Tool {index_name}')
    plt.xlabel('Time')
    plt.ylabel('Health Index')
    plt.legend()
    plt.show()

# 设置平滑系数
alpha = 0.3

# 处理每个数据集
process_data(data_x1, alpha, 'C1')
process_data(data_x4, alpha, 'C4')
process_data(data_x6, alpha, 'C6')

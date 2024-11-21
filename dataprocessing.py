import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import pywt
from scipy.stats import kurtosis, skew
import matplotlib.pyplot as plt

plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号 #有中文出现的情况，需要u'内容'

def stft(signal, window_size, overlap):
    step = window_size - overlap
    windows = [
        signal[i:i+window_size]
        for i in range(0, len(signal) - window_size + 1, step)
    ]
    stft_result = [np.fft.fft(window) for window in windows]
    return np.array(stft_result)
class Data_processing():
    def __init__(self, path):
        self.path = path

    def get_all_file(self):  # 获取文件夹中的所有文件名称
        files = os.listdir(path)
        files.sort(key=lambda x: int(x[4:-4]))
        s = []
        for file in files:
            if not os.path.isdir(path + file):  # 判断该文件是否是一个文件夹
                f_name = str(file)
                #             print(f_name)
                tr = '\\'
                filename = path + tr + f_name
                s.append(filename)  # 把当前文件名返加到列表里
        return s

    def get_data(self, i):  # 获得相应的数据
        list = self.get_all_file()
        data = pd.read_csv(list[i - 1], names=['Fx', 'Fy', 'Fz', 'Ax', 'Ay', 'Az', 'AE_rms'])
        return data

    def get_data_sample(self, i, column_num, num_points=5000):
        # 获取全部数据
        data = self.get_data(i=i)
        # 选择特定列
        data = data.iloc[:, column_num]
        # 随机采样5000个数据点（不放回采样）
        sampled_data = data.sample(n=num_points, replace=False, random_state=42)
        return sampled_data.reset_index(drop=True)

    def get_feature(self, i, column_num):  # 提取第i个文件第n列的特征
        data = self.get_data(i=i)
        data = data.iloc[:, column_num]  # 1.使用全部数据集，2.使用10万个数据量
        size = data.size
        # 绝对均值 0
        absolute_mean_value = np.sum(np.fabs(data)) / size
        # 峰峰值 0
        max = np.max(data)-min(data)
        # 均方根值 0
        root_mean_score = np.sqrt(np.sum(np.square(data)) / size)
        # 方根幅值
        Root_amplitude = np.square(np.sum(np.sqrt(np.fabs(data))) / size)
        # 歪度值
        skewness = np.sum(np.power((np.fabs(data) - absolute_mean_value), 3)) / size
        # 峰度值
        Kurtosis_value = np.sum(np.power(data, 4)) / size
        # 形状因子 0
        shape_factor = root_mean_score / absolute_mean_value
        # 脉冲因子 0
        pulse_factor = max / absolute_mean_value
        # 偏度  0
        skewness_factor = skewness / np.power(root_mean_score, 3)
        # 峰值因子 0
        crest_factor = max / root_mean_score
        # 边际因子 0
        clearance_factor = max / Root_amplitude
        # 峰度   0
        Kurtosis_factor = Kurtosis_value / np.power(root_mean_score, 4)
        #标准差 0
        biaozhun_cha=np.std(data)
        # 能量 0
        nengliang = np.sum(np.power(data, 2))



        # 频域
        data_fft = np.fft.fft(data)
        Y = np.abs(data_fft)
        freq = np.fft.fftfreq(size, 1 / 50000)
        ps = Y ** 2 / size
        # 重心频率
        FC = np.sum(freq * ps) / np.sum(ps)
        # 均方频率
        MSF = np.sum(ps * np.square(freq)) / np.sum(ps)
        # 均方根频率
        RMSF = np.sqrt(MSF)
        # 频率方差
        VF = np.sum(np.square(freq - FC) * ps) / np.sum(ps)

        # 计算 STFT 结果
        stft_result = stft(data, 256, 128)

        # 计算每个窗口频谱的幅值
        spectrum_amplitude = np.abs(stft_result)
        spectrum_kurtosis = kurtosis(spectrum_amplitude, fisher=False)

        # 计算谱峭度的均值、标准差、偏度和峰度
        mean_sk = np.mean(spectrum_kurtosis)
        std_sk = np.std(spectrum_kurtosis)
        skew_sk = skew(spectrum_kurtosis)
        kurtosis_sk = kurtosis(spectrum_kurtosis, fisher=False)



        # 时频域
        wp = pywt.WaveletPacket(data, wavelet='db3', mode='symmetric', maxlevel=3)
        aaa = wp['aaa'].data
        aad = wp['aad'].data
        ada = wp['ada'].data
        add = wp['add'].data
        daa = wp['daa'].data
        dad = wp['dad'].data
        dda = wp['dda'].data
        ddd = wp['ddd'].data
        ret1 = np.linalg.norm(aaa, ord=None)
        ret2 = np.linalg.norm(aad, ord=None)
        ret3 = np.linalg.norm(ada, ord=None)
        ret4 = np.linalg.norm(add, ord=None)
        ret5 = np.linalg.norm(daa, ord=None)
        ret6 = np.linalg.norm(dad, ord=None)
        ret7 = np.linalg.norm(dda, ord=None)
        ret8 = np.linalg.norm(ddd, ord=None)

        f = [absolute_mean_value, max, root_mean_score, Root_amplitude, skewness, Kurtosis_value,
             shape_factor, pulse_factor, skewness_factor, crest_factor, clearance_factor, Kurtosis_factor,
             FC, MSF, RMSF, VF,
             ret1, ret2, ret3, ret4, ret5, ret6, ret7, ret8]
        new_F=[absolute_mean_value,max,root_mean_score,shape_factor,
               pulse_factor,skewness_factor,crest_factor,clearance_factor,
               Kurtosis_factor,biaozhun_cha,nengliang,mean_sk, std_sk, skew_sk,
               kurtosis_sk]
        return new_F

    def get_all_ori_feature(self):
        features = np.empty([315, 6,5000])
        for i in range(315):
            for j in range(6):
                features[i, j,:] = self.get_data_sample(i, j)
        return features

    def get_all_feature(self):
        features = np.empty([315, 6,15])
        for i in range(315):
            for j in range(6):
                features[i, j, :] = self.get_feature(i, j)
        return features

    def get_label(self, filename=None):
        data = pd.read_csv(filename)
        y1 = np.array(data['flute_1'])
        y2 = np.array(data['flute_2'])
        y3 = np.array(data['flute_3'])
        y1 = y1.reshape(y1.shape[0], 1)
        y2 = y2.reshape(y2.shape[0], 1)
        y3 = y3.reshape(y3.shape[0], 1)
        y = np.concatenate((y1, y2, y3), axis=1)
        ya = np.mean(y, 1)
        return ya


def plot_features(data, column_num):  # column_num∈[1, 6]
    features = data[:, column_num-1, :]
    x1 = range(0, features.shape[0])
    plt.figure(num=0, figsize=(12, 5))
    plt.plot(x1, features[:, 0], '-g', label='均值')
    plt.plot(x1, features[:, 1], '--c', label='峰峰值')
    plt.plot(x1, features[:, 2], '-.k', label='均方根值')
    plt.plot(x1, features[:, 3], ':r', label='形状因子')
    plt.plot(x1, features[:, 4], '-y', label='脉冲因子')
    plt.plot(x1, features[:, 5], '-m', label='偏度')
    plt.plot(x1, features[:, 6], '-og', label='波峰因子')
    plt.plot(x1, features[:, 7], '-*c', label='边际因子')
    plt.plot(x1, features[:, 8], '-xk', label='峰度')
    plt.plot(x1, features[:, 9], '-vr', label='标准差')
    plt.plot(x1, features[:, 10], '-sy', label='能量')
    plt.xlabel('Times of cutting')
    plt.ylabel('Time domain features')
    plt.legend(loc=1)
    plt.show()
    plt.figure(num=1, figsize=(12, 5))
    plt.plot(x1, features[:, 11], '-vr', label='SK均值')
    plt.plot(x1, features[:, 12], '-k', label='SK标准差')
    plt.plot(x1, features[:, 13], '-xk', label='SK偏度')
    plt.plot(x1, features[:, 14], '-og', label='SK峰值')

    plt.xlabel('Times of cutting')
    plt.ylabel('Frequency domain features')
    plt.legend(loc=1)
    plt.show()


path = r'D:\Desktop\20241029\PHM 2010\c1\c1'
file1 = r'D:\Desktop\20241029\PHM 2010\c1\c1_wear.csv'
Data1 = Data_processing(path)
data_x1_origin=Data1.get_all_ori_feature()
data_x1 = Data1.get_all_feature()

data_y1 = Data1.get_label(file1)
print('data_x1:', data_x1.shape, 'data_y1:', data_y1.shape)
plot_features(data_x1, 0)

path = r'D:\Desktop\20241029\PHM 2010\c4\c4'
file4 = r'D:\Desktop\20241029\PHM 2010\c4\c4_wear.csv'
Data4 = Data_processing(path)
data_x4 = Data1.get_all_feature()
data_x4_origin=Data4.get_all_ori_feature()
data_y4 = Data1.get_label(file4)
print('data_x4:', data_x4.shape, 'data_y4:', data_y4.shape)
plot_features(data_x4, 0)

path = r'D:\Desktop\20241029\PHM 2010\c6\c6'
file6 = r'D:\Desktop\20241029\PHM 2010\c6\c6_wear.csv'
Data6 = Data_processing(path)
data_x6 = Data1.get_all_feature()
data_x6_origin=Data6.get_all_ori_feature()
data_y6 = Data1.get_label(file6)
print('data_x6:', data_x6.shape, 'data_y6:', data_y6.shape)
plot_features(data_x6, 0)

np.save("features\\data_x1.npy", data_x1)
np.save("features\\data_y1.npy", data_y1)
np.save("features\\data_x4.npy", data_x4)
np.save("features\\data_y4.npy", data_y4)
np.save("features\\data_x6.npy", data_x6)
np.save("features\\data_y6.npy", data_y6)

np.save("originfeature\\data_x1.npy",data_x1_origin)
np.save("originfeature\\data_x4.npy",data_x4_origin)
np.save("originfeature\\data_x6.npy",data_x6_origin)
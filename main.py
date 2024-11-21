import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from MultiChannel1DCNN import MultiChannel1DCNN
import torch.utils.data as Data
# 假设多通道1DCNN模型

# 粒子滤波类
class ParticleFilter:
    def __init__(self, num_particles, state_transition_func, observation_func):
        self.num_particles = num_particles
        self.particles = np.random.randn(num_particles)  # 初始化粒子
        self.weights = np.ones(num_particles) / num_particles  # 初始化权重
        self.state_transition_func = state_transition_func
        self.observation_func = observation_func

    def predict(self):
        # 使用状态转移函数预测
        self.particles = self.state_transition_func(self.particles)

    def update(self, observation):
        # 更新粒子权重
        likelihood = self.observation_func(self.particles, observation)
        self.weights *= likelihood
        self.weights += 1.e-300  # 防止下溢
        self.weights /= np.sum(self.weights)  # 归一化权重

    def resample(self):
        indices = np.random.choice(range(self.num_particles), size=self.num_particles, p=self.weights)
        self.particles = self.particles[indices]
        self.weights.fill(1.0 / self.num_particles)

# 状态转移函数示例
def state_transition_func(particles):
    # 使用状态转移方程（如公式（3-40））
    # 这里假设简单的随机游走模型
    return particles + np.random.normal(0, 1, size=particles.shape)

# 观测函数示例
def observation_func(particles, observation):
    # 计算似然函数
    # 假设观测值与粒子状态之间的关系为正态分布
    return np.exp(-0.5 * ((observation - particles) ** 2))

# 主函数
def main():
    num_particles = 1000
    EPOCH = 500
    BATCH_SIZE = 2
    LR = 0.002
    particle_filter = ParticleFilter(num_particles, state_transition_func, observation_func)
    model = MultiChannel1DCNN()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.MSELoss()
    model=model.cuda()

    data_x1 = np.load("D:\\Desktop\\20241029\\Experiments-using-PHM2010dataset-main\\originfeature\\data_x1.npy")
    data_x4 = np.load("D:\\Desktop\\20241029\\Experiments-using-PHM2010dataset-main\\originfeature\\data_x4.npy")
    data_x6 = np.load("D:\\Desktop\\20241029\\Experiments-using-PHM2010dataset-main\\originfeature\\data_x6.npy")
    data_y1 = np.load("D:\\Desktop\\20241029\\Experiments-using-PHM2010dataset-main\\originfeature\\data_y1.npy")
    data_y4 = np.load("D:\\Desktop\\20241029\\Experiments-using-PHM2010dataset-main\\originfeature\\data_y4.npy")
    data_y6 = np.load("D:\\Desktop\\20241029\\Experiments-using-PHM2010dataset-main\\originfeature\\data_y6.npy")
    train_x = np.append(data_x1, data_x6, axis=0)
    train_y = np.append(data_y1, data_y6, axis=0)
    test_x = data_x4
    test_y = data_y4

    train_x = torch.from_numpy(train_x)
    train_y = torch.from_numpy(train_y)
    test_x = torch.from_numpy(test_x)
    test_y = torch.from_numpy(test_y)

    train_dataset = Data.TensorDataset(train_x, train_y)
    all_num = train_x.shape[0]
    train_num = int(all_num * 0.8)
    train_data, val_data = Data.random_split(train_dataset, [train_num, all_num - train_num])
    train_loader = Data.DataLoader(dataset=train_data, batch_size=BATCH_SIZE, shuffle=True, )
    val_loader = Data.DataLoader(dataset=val_data, batch_size=BATCH_SIZE, shuffle=True, )

    test_dataset = Data.TensorDataset(test_x, test_y)
    test_loader = Data.DataLoader(dataset=test_dataset, batch_size=BATCH_SIZE, shuffle=False, )
    train_losses = []  # 用于记录训练损失
    val_losses = []  # 用于记录验证损失
    mae_list = []  # 用于记录验证的MAE
    rmse_list = []  # 用于记录验证的RMSE
    for epoch in range(100):  # 迭代次数
        for step, (x_batch, y_batch) in enumerate(train_loader):  # 假设您有批量数据

            model.train()
            epoch_train_loss = 0  # 记录当前epoch的训练损失
            optimizer.zero_grad()

            x_batch =x_batch.float()
            y_batch = y_batch.float()
            if torch.cuda.is_available():
                x_batch = x_batch.cuda()
                y_batch =y_batch.cuda()

            # 预测磨损值
            predicted = model(x_batch)  # 添加batch维度

            # 粒子滤波过程
            particle_filter.predict()
            predicted_single = predicted[0].item() if predicted.numel() > 1 else predicted.item()


            particle_filter.update(predicted_single)

            particle_filter.resample()

            # 计算损失
            loss = criterion(predicted, y_batch.unsqueeze(0))  # 添加batch维度
            loss.backward()
            optimizer.step()
            # 记录损失
            epoch_train_loss += loss.item()
            # print(f'Epoch [{epoch+1}/100], Loss: {loss.item():.4f}')
        model.eval()
        # 计算平均训练损失
        avg_train_loss = epoch_train_loss / len(train_loader)
        train_losses.append(avg_train_loss)
        print(f'Epoch [{epoch + 1}/100], Train Loss: {avg_train_loss:.4f}')

        model.eval()
        epoch_val_loss = 0  # 记录当前epoch的验证损失
        y_true, y_pred = [], []  # 用于存储真实值和预测值
        with torch.no_grad():
            for v_x, v_y in val_loader:
                v_x = v_x.float()
                v_y = v_y.float()
                if torch.cuda.is_available():
                    v_x = v_x.cuda()
                    v_y = v_y.cuda()

                predicted  = model(v_x).squeeze(-1)

                particle_filter.predict()
                predicted_single = predicted[0].item() if predicted.numel() > 1 else predicted.item()

                particle_filter.update(predicted_single)

                particle_filter.resample()

                # 计算验证损失
                val_loss = criterion(predicted , v_y)
                epoch_val_loss += val_loss.item()

                # 存储真实值和预测值
                y_true.extend(v_y.cpu().numpy())
                y_pred.extend(predicted .cpu().numpy())

         # 粒子滤波修正后的预测结果
        estimated_wear = np.average(particle_filter.particles, weights=particle_filter.weights)
        print("粒子滤波结果",estimated_wear)
        # 验证是否超过磨损阈值
        # if estimated_wear >= fail_threshold:
        #         print(f"刀具磨损已达到失效阈值 {fail_threshold}，建议更换刀具")
        #         break  # 终止流程
        # else:
        #         print(f"当前修正后预测磨损值: {estimated_wear:.4f}，低于失效阈值 {fail_threshold}")
        # 计算平均验证损失
        avg_val_loss = epoch_val_loss / len(val_loader)
        val_losses.append(avg_val_loss)
        print(f'Epoch [{epoch + 1}/100], Val Loss: {avg_val_loss:.4f}')

        # 计算MAE和RMSE
        mae = np.mean(np.abs(np.array(y_true) - np.array(y_pred)))
        rmse = np.sqrt(np.mean((np.array(y_true) - np.array(y_pred)) ** 2))
        mae_list.append(mae)
        rmse_list.append(rmse)
        print(f'Epoch [{epoch + 1}/100], Val MAE: {mae:.4f}, Val RMSE: {rmse:.4f}')


if __name__ == '__main__':
    main()

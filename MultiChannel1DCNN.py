# 3.2.2  多通道1DCNN预测模型代码
import torch
import torch.nn as nn
import torch.nn.functional as F

class MultiChannel1DCNN(nn.Module):
    def __init__(self):
        super(MultiChannel1DCNN, self).__init__()

        # 定义各层
        self.conv1 = nn.Conv1d(in_channels=6, out_channels=32, kernel_size=5, stride=2)
        self.pool1 = nn.MaxPool1d(kernel_size=3, stride=1)

        self.conv2 = nn.Conv1d(in_channels=32, out_channels=32, kernel_size=5, stride=2)
        self.pool2 = nn.MaxPool1d(kernel_size=2, stride=1)

        self.conv3 = nn.Conv1d(in_channels=32, out_channels=32, kernel_size=5, stride=2)
        self.pool3 = nn.MaxPool1d(kernel_size=3, stride=1)

        self.conv4 = nn.Conv1d(in_channels=32, out_channels=16, kernel_size=5, stride=2)
        self.pool4 = nn.MaxPool1d(kernel_size=2, stride=1)

        self.conv5 = nn.Conv1d(in_channels=16, out_channels=16, kernel_size=5, stride=2)
        self.pool5 = nn.MaxPool1d(kernel_size=3, stride=1)

        self.conv6 = nn.Conv1d(in_channels=16, out_channels=16, kernel_size=5, stride=2)
        self.pool6 = nn.MaxPool1d(kernel_size=3, stride=1)

        self.fc1 = nn.Linear(16 *71, 224)  # 计算扁平化后的输入特征数
        self.output = nn.Linear(224, 1)

    def forward(self, x):
        # 各层前向传播
        x = self.pool1(F.relu(self.conv1(x)))
        x = self.pool2(F.relu(self.conv2(x)))
        x = self.pool3(F.relu(self.conv3(x)))
        x = self.pool4(F.relu(self.conv4(x)))
        x = self.pool5(F.relu(self.conv5(x)))
        x = self.pool6(F.relu(self.conv6(x)))

        # 扁平化
        x = x.view(x.size(0), -1)  # 扁平化处理
        x = F.relu(self.fc1(x))
        x = self.output(x)
        return x

# 示例：创建模型并查看输出
if __name__ == "__main__":
    model = MultiChannel1DCNN()
    print(model)

    # 创建一个示例输入 (batch_size=1, input_length=5000)
    input_data = torch.randn(1, 5000)
    output = model(input_data)
    print("Output shape:", output.shape)  # 应该是 (1, 1)

# 3.2.2  多通道1DCNN预测模型的测试的代码
from sklearn.metrics import mean_squared_error, mean_absolute_error
import math
import torch
import numpy as np
import torch.nn as nn
import torch.utils.data as Data
import matplotlib.pyplot as plt
import math
from sklearn.metrics import mean_squared_error
from MultiChannel1DCNN import MultiChannel1DCNN

EPOCH =100
BATCH_SIZE = 2
LR = 0.002

data_x1 = np.load("D:\\Desktop\\20241029\\Experiments-using-PHM2010dataset-main\\originfeature\\data_x1.npy")
data_x4 = np.load("D:\\Desktop\\20241029\\Experiments-using-PHM2010dataset-main\\originfeature\\data_x4.npy")
data_x6 = np.load("D:\\Desktop\\20241029\\Experiments-using-PHM2010dataset-main\\originfeature\\data_x6.npy")
data_y1 = np.load("D:\\Desktop\\20241029\\Experiments-using-PHM2010dataset-main\\originfeature\\data_y1.npy")
data_y4 = np.load("D:\\Desktop\\20241029\\Experiments-using-PHM2010dataset-main\\originfeature\\data_y4.npy")
data_y6 = np.load("D:\\Desktop\\20241029\\Experiments-using-PHM2010dataset-main\\originfeature\\data_y6.npy")


def normalization(data):
    mu = np.mean(data, axis=0)
    sigma = np.std(data, axis=0)
    return (data - mu) / sigma


def norm_all(data):
    d = np.empty((data.shape[0], data.shape[1], data.shape[2]))
    for i in range(data.shape[1]):
        data1 = data[:, i, :]
        for j in range(data1.shape[0]):
            data2 = data1[j, :]
            d[j, i, :] = normalization(data2)
    return d


def normal_label(data):
    minVals = data.min(0)
    maxVals = data.max(0)
    ranges = maxVals - minVals
    normData = np.zeros(np.shape(data))
    m = data.shape[0]
    normData = data - np.tile(minVals, (m, 1))
    normData = normData / np.tile(ranges, (m, 1))
    return normData[0]


data_x1 = norm_all(data_x1)
data_x4 = norm_all(data_x4)
data_x6 = norm_all(data_x6)
data_y1 = normal_label(data_y1)
data_y4 = normal_label(data_y4)
data_y6 = normal_label(data_y6)

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





model = MultiChannel1DCNN()
if torch.cuda.is_available():
    model = model.cuda()

optimizer = torch.optim.Adam(model.parameters(), lr=LR)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=100, gamma=0.9)
loss_func = torch.nn.MSELoss()

train_loss = []
val_loss = []
lr_list = []

for epoch in range(EPOCH):
    total_loss = 0
    total_loss2 = 0
    model.train()
    for step, (b_x, b_y) in enumerate(train_loader):
        b_x = b_x.float()
        b_y = b_y.float()
        if torch.cuda.is_available():
            b_x = b_x.cuda()
            b_y = b_y.cuda()
        output = model(b_x).squeeze(-1)
        loss = loss_func(output, b_y)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        scheduler.step()

        total_loss += loss.cpu().item()

    total_loss /= len(train_loader.dataset)
    train_loss.append(total_loss)
    lr_list.append(optimizer.state_dict()['param_groups'][0]['lr'])

    model.eval()
    with torch.no_grad():
        for i, (v_x, v_y) in enumerate(val_loader):
            v_x = v_x.float()
            v_y = v_y.float()
            if torch.cuda.is_available():
                v_x = v_x.cuda()
                v_y = v_y.cuda()

            test_output = model(v_x).squeeze(-1)
            v_loss = loss_func(test_output, v_y)

            total_loss2 += v_loss.cpu().item()

        total_loss2 /= len(val_loader.dataset)
        val_loss.append(total_loss2)

    print('Train Epoch: {} \t Train Loss:{:.6f} \t Val Loss:{:.6f}'.format(epoch, total_loss, total_loss2))

X0 = np.array(train_loss).shape[0]
x1 = range(0, X0)
x2 = range(0, X0)
y1 = train_loss
y2 = val_loss
plt.subplot(2, 1, 1)
plt.plot(x1, y1, '-')
plt.ylabel('train_loss')
plt.subplot(2, 1, 2)
plt.plot(x2, y2, '-')
plt.ylabel('val_loss')
plt.show()

pred = torch.empty(1)
model.eval()
with torch.no_grad():
    for i, (tx, ty) in enumerate(test_loader):
        tx = tx.float()
        ty = ty.float()
        if torch.cuda.is_available():
            tx = tx.cuda()
            ty = ty.cuda()

        out = model(tx).squeeze(-1)
        pred = torch.cat((pred, out.cpu()))

pred = np.delete(pred.detach().numpy(), 0, axis=0)
yy1 = pred
yy2 = test_y.cpu().detach().numpy()

rmse = math.sqrt(mean_squared_error(pred, yy2))
mae = mean_absolute_error(pred, yy2)
print('Test RMSE: %.3f' % rmse)
print('Test MAE: %.3f' % mae)


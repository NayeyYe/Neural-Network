import os
import torch
import torch.nn as nn
import torch.utils.data as Data
import torchvision

epochs = 3
batch_size = 100
learning_rate = 0.001


def prepare_data():
    if not os.path.exists("./mnist/") or not os.listdir("./mnist/"):
        DOWNLOAD_MNIST = True
    else:
        DOWNLOAD_MNIST = False

    train_data = torchvision.datasets.MNIST(
        root="./mnist/",
        train=True,
        transform=torchvision.transforms.ToTensor(),
        download=DOWNLOAD_MNIST,
    )

    print("train_data.data: ", train_data.data)
    print("train_data.targets: ", train_data.targets)
    print("train_data.data.size: ", train_data.data.size())
    print("train_data.targets.size: ", train_data.targets.size())

    return train_data

def data_load(train_data):
    '''
    :param train_data: 数据的全部数据
    :return train_loader, test_x, test_y: 每一个batch的数据，测试集, 测试集标签
    '''
    train_loader = Data.DataLoader(dataset=train_data, batch_size=batch_size, shuffle=True)
    # Data.DataLoader函数:
    # 1. 用于自动分批, 将数据集分割成小批量(mini-batch)
    # 2. 将训练集打乱(shuffle=True), 如果是测试集和验证机就不打乱
    '''
    :param dataset: 指定要加载的数据集对象(此处为之前定义的 MNIST 数据集)
    :param batch_size: 定义每个批次包含的样本数
    :param shuffle: 每个epoch开始时是否随机打乱数据顺序
    '''

    # 用2000个样本作为测试
    # 注意！这里的train是False, 表示不需要训练, 加载测试集
    test_data = torchvision.datasets.MNIST(root='./mnist/', train=False)
    # shape from (2000, 28, 28) to (2000, 1, 28, 28), value in range(0, 1)
    test_x = torch.unsqueeze(test_data.data, dim=1).type(torch.FloatTensor)[:2000] / 255.
    '''
    test_data.test_data: 获取原始图像数据
    torch.unsqueeze(dim=1): 在维度1通道拓展, 将形状从(N, 28, 28)-> (N, 1, 28, 28)
    .type(torch.FloatTensor):将数据类型从uint8转为float32
    [:2000]: 只要前2000个样本
    / 255.: 归一化
    '''
    test_y = test_data.targets[:2000]
    '''
    获取测试集标签
    '''
    return train_loader, test_x, test_y

class MLP(nn.Module):
    def __init__(self):
        super(MLP, self).__init__()
        self.mlp = nn.Sequential(
            nn.Linear(784, 28*28),
            nn.Linear(28*28, 10),
        )

    def forward(self, x):
        output = self.mlp(x)
        return output, x

def train(train_loader, test_x, test_y):
    mlp = MLP()
    print(mlp)

    optimizer = torch.optim.Adam(mlp.parameters(), lr=learning_rate)
    loss_func = nn.CrossEntropyLoss()

    for epoch in range(epochs):
        for step, (data, labels) in enumerate(train_loader):
            data = data.view(-1, 784)

            output = mlp(data)[0]
            loss = loss_func(output, labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if step % 50 == 0:

                test_output, last_layer = mlp(test_x.view(-1, 784))
                pred_y = torch.max(test_output, 1)[1].data.numpy()
                accuracy = float((pred_y == test_y.data.numpy()).astype(int).sum()) / float(test_y.size(0))
                print('Epoch: ', epoch, '| train loss: %.4f' % loss.data.numpy(), '| test accuracy: %.2f' % accuracy)
    return mlp

def test(mlp, test_x, test_y,):
    test_output, _ = mlp(test_x[:10].view(-1, 28 * 28))
    pred_y = torch.max(test_output, 1)[1].data.numpy()
    print(pred_y, 'prediction number')
    print(test_y[:10].numpy(), 'real number')

if __name__ == '__main__':
    train_data = prepare_data()
    train_loader, test_x, test_y = data_load(train_data)
    mlp = train(train_loader, test_x, test_y)
    test(mlp, test_x, test_y)
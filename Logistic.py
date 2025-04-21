import os
import torch
import torch.nn as nn
import torch.utils.data as Data
import torchvision
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE


epochs = 10
batch_size = 100
learning_rate = 0.001
HAS_SK = True

def prepare_data():
    # 下载 MNIST 数据集
    # 如果没有mnist文件夹或者mnist文件夹是空的
    if not(os.path.exists( './mnist/')) or not os.listdir('./mnist/'):
        # 下载MNIST数据集
        DOWNLOADER_MNIST = True
    else:
        DOWNLOADER_MNIST = False


    # 训练数据集
    train_data = torchvision.datasets.MNIST(
        root = './mnist/', # 数据集的根目录
        train=True, # 加载训练集, 如果是False, 就是加载测试集
        transform = torchvision.transforms.ToTensor(), # 将图像转化为张量
        # target_transform: 对标签数据进行处理, 比如one-hot编码
        # target_transform=Lambda(y: torch.tensor(y))
        download=DOWNLOADER_MNIST, # 如果DOWNLOADER_MNIST是True, 数据不存在就自动下载
    )

    # 输出一个例子
    # 创建存储目录（如果不存在）
    savedir = './images/Logistic'
    os.makedirs(savedir, exist_ok=True) # 自动创建目录，存在时不报错
    image_name = 'num_example.png'
    image_path = os.path.join(savedir, image_name)
    print(train_data.data.size()) # (60000, 28, 28)
    print(train_data.targets.size()) # (60000)
    plt.imshow(train_data.data[0].numpy(), cmap='gray')
    plt.title('%i' % train_data.targets[0])
    plt.savefig(image_path)
    plt.show()

    return train_data

# 加载数据
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


class logistic(nn.Module):
    def __init__(self):
        super(logistic, self).__init__()
        self.linear = nn.Linear(
            in_features=1*28*28, # 输入形状为1*28*28
            out_features=10 # 输出形状为10， 表示为10个数字
        )

    def forward(self, x):
        output = self.linear(x)
        return output, x # 返回x是为了可视化

def plot_with_labels(lowDWeights, labels, epoch, step):
    """
    可视化高维数据的二维嵌入结果（如t-SNE降维后的特征）

    参数:
        lowDWeights : numpy数组 或 torch.Tensor, 形状为 [N, 2]
            降维后的二维坐标数据，每行代表一个样本的坐标
        labels : numpy数组 或 torch.Tensor, 形状为 [N]
            每个样本对应的类别标签（整数形式）
        epoch : int - 当前训练轮次
        step : int - 当前训练步数
    """
    plt.cla()  # 清除当前axes，准备绘制新图（适用于动态更新场景）

    # 解包二维坐标为X,Y轴数据
    X = lowDWeights[:, 0]  # 所有样本的第一维坐标
    Y = lowDWeights[:, 1]  # 所有样本的第二维坐标

    # 创建存储目录（如果不存在）
    save_dir = './images/Logistic'
    os.makedirs(save_dir, exist_ok=True)  # 自动创建目录，存在时不报错

    # 生成唯一文件名
    image_name = f'datapoints_epoch_{epoch}_step_{step}.png'
    image_path = os.path.join(save_dir, image_name)
    # 遍历每个数据点绘制带背景色的文本标签
    from matplotlib import cm


    for x, y, s in zip(X, Y, labels):
        # 计算颜色映射:将标签s映射到彩虹色系的0-255范围
        # 假设labels为0-9的整数，s/9将归一化到0~1，乘以255后转为整数索引
        c = cm.rainbow(int(255 * s / 9))

        # 在坐标(x,y)处绘制文本标签，背景色为c
        plt.text(
            x, y, str(s),  # 显示标签数值
            backgroundcolor=c,  # 按类别着色的背景
            fontsize=9,  # 字体大小
            alpha=0.8  # 建议添加透明度以提升重叠点的可视性
        )

    # 设置坐标轴范围（根据数据范围自动调整）
    plt.xlim(X.min(), X.max())  # X轴显示范围覆盖所有数据
    plt.ylim(Y.min(), Y.max())  # Y轴显示范围覆盖所有数据

    # 添加标题并显示图形
    plt.title('Visualize last layer')  # 图表标题
    # plt.show()  # 渲染显示图形
    # 图片保存
    plt.savefig(image_path, dpi=150, bbox_inches='tight')

    # 短暂暂停（允许图形界面更新，适用于动态可视化）
    # plt.pause(0.01)  # 暂停0.01秒（配合plt.ion()实现动态更新）


"""
使用说明:
1. 需配合降维方法（如TSNE）使用，输入应为二维坐标数据
2. labels应为整数型类别标签（如MNIST的0-9）
3. 建议在Jupyter或支持交互的Python环境中使用plt.ion()启用交互模式

注意事项:
- 数据量较大时plt.text会导致渲染卡顿，建议限制plot_only=500个点
- 若标签范围非0-9，需调整颜色计算中的分母（如s/max_label）
- 可添加alpha参数提升文本重叠时的可读性（见代码注释）
"""


def train(train_loader, test_x, test_y):
    model = logistic() # 实例化自定义的卷积神经网络
    print(model) # 打印网络结构（显示各层参数）

    # 使用Adam优化器: 自适应调节学习率，适合大多数深度学习任务
    optimizer = torch.optim.Adam(model.parameters(),  # # 待优化的模型参数（通常为 model.parameters()）
                                 lr=learning_rate
                                 )
    # 交叉熵损失函数: 专为分类任务设计，自动处理Softmax和类别概率计算
    loss_func = nn.CrossEntropyLoss()
    # training and testing
    # 遍历所有训练周期

    for epoch in range(epochs):
        for step, (data, target) in enumerate(train_loader):
            data = data.view(-1, 28*28)

            output = model(data)[0]
            loss = loss_func(output, target)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # 验证和可视化
            if step % 50 == 0:
                # 获取测试集预测结果和最后一层特征
                test_output, last_layer = model(test_x.view(-1, 28*28))
                # 取预测类别
                pred_y = torch.max(test_output, 1)[1].data.numpy()
                # 计算准确率
                accuracy = float((pred_y == test_y.data.numpy()).astype(int).sum()) / float(test_y.size(0))
                print('Epoch: ', epoch, '| train loss: %.4f' % loss.data.numpy(), '| test accuracy: %.2f' % accuracy)
                if HAS_SK:
                    # Visualization of trained flatten layer (T-SNE)
                    tsne = TSNE(perplexity=30, n_components=2, init='pca', max_iter=5000)
                    plot_only = 500
                    low_dim_embs = tsne.fit_transform(last_layer.data.numpy()[:plot_only, :])
                    labels = test_y.numpy()[:plot_only]
                    plot_with_labels(low_dim_embs, labels, epoch, step)
    return model


def test(logistic_model, test_x, test_y):
    # 对前10个测试样本预测
    test_output, _ = logistic_model(test_x[:10].view(-1, 28*28))
    # 获取预测数字
    pred_y = torch.max(test_output, 1)[1].data.numpy()
    # 打印预测结果
    print(pred_y, 'prediction number')
    # 打印真实标签
    print(test_y[:10].numpy(), 'real number')


if __name__ == '__main__':
    train_data = prepare_data()
    train_loader, test_x, test_y = data_load(train_data)
    logistic_model = train(train_loader, test_x, test_y)
    test(logistic_model, test_x, test_y)
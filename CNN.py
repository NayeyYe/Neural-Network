# 卷积神经网络
# 卷积神经网络（CNN）详解
# CNN是计算机视觉领域的核心深度学习模型，
# 擅长处理网格状数据（如图像、音频频谱、时间序列）。
# 以下是其核心要点：
# ------------------------------------------------------------------
# 输入层 (Input Layer)
# ------------------------------------------------------------------
# 作用：接收并预处理原始输入数据
# 关键技术：
#   - 标准化(Normalization)：将像素值缩放到[0,1]或[-1,1]区间
# 示例参数：
#   - 形状：(224, 224, 3) 对应(height, width, channels)
#   - 常见数据类型：float32（归一化后的图像数据）
# 注意：输入尺寸需与后续卷积核参数兼容

# ------------------------------------------------------------------
# 卷积层 (Convolution Layer)
# ------------------------------------------------------------------
# 作用：通过卷积核提取空间特征
# 关键技术：
#   - 滤波器(Filter)：权重矩阵，用于特征检测
#   - 步长(Stride)：控制滑动步幅（常用1或2）
#   - 填充(Padding)：'same'保持尺寸/'valid'缩减尺寸
# 示例参数：
#   - filters=32（卷积核数量）
#   - kernel_size=(3,3)  # 常用3x3或5x5
#   - strides=1          # 通常1或2
#   - padding='same'     # 输出尺寸与输入相同
# 注意：使用kernel_initializer设置权重初始化方式

# ------------------------------------------------------------------
# 激活函数 (Activation Function)
# ------------------------------------------------------------------
# 作用：引入非线性变换，增强模型表达能力
# 关键技术：
#   - ReLU：f(x)=max(0,x)（解决梯度消失问题）
#   - Leaky ReLU：负区间保留微小梯度（α通常取0.01）
#   - Swish：x*sigmoid(βx)（Google提出的平滑ReLU变体）
# 示例参数：
#   - tf.keras.layers.ReLU(negative_slope=0.01) # Leaky ReLU实现
#   - activation='swish' （TensorFlow 2.4+原生支持）
# 注意：最后一层需根据任务选择激活函数（如sigmoid/softmax）

# ------------------------------------------------------------------
# 池化层 (Pooling Layer)
# ------------------------------------------------------------------
# 作用：降维和空间信息压缩，增强平移不变性
# 关键技术：
#   - 最大池化：取窗口内最大值（保留显著特征）
#   - 平均池化：取窗口内平均值（平滑特征）
# 示例参数：
#   - pool_size=(2,2)  # 常用2x2窗口
#   - strides=2        # 通常等于pool_size
#   - padding='valid'  # 默认不填充
# 注意：GlobalAveragePooling可用于替代全连接层

# ------------------------------------------------------------------
# 全连接层 (Fully Connected Layer)
# ------------------------------------------------------------------
# 作用：整合特征并进行最终分类/回归
# 关键技术：
#   - Dropout：随机失活神经元（rate通常0.2-0.5）
#   - Softmax：多分类概率归一化
# 示例参数：
#   - units=1000       # ImageNet分类任务输出维度
#   - activation=None  # 通常配合单独激活层使用
#   - kernel_regularizer=l2(0.01) # L2正则化
# 注意：最后一层维度需与分类数一致，回归任务使用线性激活
# 读取当前文件夹
import os

# 导入torch库
import torch
import torchvision
import torch.nn as nn
import torch.utils.data as Data
import matplotlib.pyplot as plt
from matplotlib import cm
# 参数
'''
    EPOCH: 训练轮数: 把数据训练的次数, 1个EPOCH有(数据个数 / BATCH_SIZE = BATCH)个迭代过程
    BATCH_SIZE: 1个BATCH有BATCH_SIZE个数据
    LR: 学习率
    DOWNLOADER_MNIST: 是否下载 MNIST 数据集
'''
EPOCH = 3
BATCH_SIZE = 100


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
    savedir = './images/CNN'
    os.makedirs('./images/CNN', exist_ok=True) # 自动创建目录，存在时不报错
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
    train_loader = Data.DataLoader(dataset=train_data, batch_size=BATCH_SIZE, shuffle=True)
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


class CNN(nn.Module):
    def __init__(self):
        # 父类是nn.Module, 要重写Module的私有属性
        super(CNN, self).__init__()

        # 第一卷积块: 提取低级特征(边缘, 纹理)
        self.conv1 = nn.Sequential( # 输入形状为(1, 28, 28)[通道x高x宽]
            nn.Conv2d(
                in_channels=1, # 通道数 = 1, 因为MNIST是灰度图, 只有1个通道
                out_channels=16, # 输出通道数 或者说是 卷积核数量
                kernel_size=5, # 卷积核尺寸 = 5x5
                stride=1, # 滑动步长 = 1
                padding=2, # 填充2像素, 保持空间分辨率
            ), 
            nn.ReLU(), # 非线性激活, 解决线性不可分问题
            nn.MaxPool2d(kernel_size=2) # 最大池化(下采样, 抗噪声)
        ) # 输出形状: (16, 14, 14)
        '''
        现在, 我会详细说明这些参数的选择
        1. in_channels=1, # 通道数 = 1, 因为MNIST是灰度图, 只有1个通道
            决定因素: 输入数据的通道维度
            比如: 因为MNIST数据集中的图像都是灰度图像, 没有彩色, 所以通道数就是1
            如果是彩色图像, 那么就是RGB三个通道, 在一个像素点他们有3个值, 通道数就是3
            想知道通道个数, 可以使用print(train_data[0][0].shape) , 输出torch.Size([1, 28, 28]), 通道个数就是1
        2. out_channels=16, # 输出通道数或者说是卷积核数量=16
            这个16并不是计算出来的, 而是在保证不出现欠拟合或者过拟合的情况下的经验结果
            一般来说, 通道数我们会通过参数量的下载来进行权衡
            参数量 = (kernel_w × kernel_h × in_channels + 1) × out_channels
            第一层参数量 = (5×5×1 + 1) × 16 = 26×16 = 416
            如果out_channels增加到32, 每个epochs时间会变成, 参数量增加到832, 一不留神就过拟合了
                简单任务(MNIST/CIFAR-10):16-64通道
                中等任务(CIFAR-100):64-256通道
                复杂任务(ImageNet):256-1024通道
            选择out_channels=16是平衡模型容量、计算效率和MNIST任务特性的实践最优解。
            实际应用中, 可通过实验对比、资源评估和自动化搜索进一步优化。
            核心原则是:在不过度增加参数量的前提下, 选择能稳定提取关键特征的最小通道数。
        3. kernel_size=5, # 卷积核尺寸 = 5x5
            同样的, 这个值不是算出来的, 是经验值
        4. stride=1, # 滑动步长 = 1
        输出尺寸 = (输入尺寸 + 2 x padding - kernel_size / stride) + 1 向下取整
        
            Stride值	适用场景	优缺点
            1	保持分辨率(浅层网络、语义分割)	计算量大, 但信息保留完整
            2	下采样(分类网络、降低计算成本)	高效但可能丢失细节
            ≥3	快速降维(处理高分辨率输入)	信息丢失风险大, 需谨慎使用
        5. padding=2, # 填充2像素, 保持空间分辨率
        当 stride=1 且 输入尺寸 = 输出尺寸 时, 解得:
            padding = (kernel_size - 1) / 2
        当使用奇数尺寸的卷积核时, 设置 padding=(kernel_size-1)/2 可保持尺寸不变。
        
        nn.ReLU(): 
        ReLU(x) = max(0, x)
        
        nn.MaxPool2d(kernel_size=2) # 最大池化(下采样, 抗噪声)
        输入:[16, 28, 28] → 输出:[16, 14, 14]
        MaxPool2d 是一个下采样的操作, 可以提取局部区域的最大值, 减少参数量
        参数说明:
        :param kernel_size: 池化窗口尺寸(决定每个池化区域的大小)
        :param stride: 	窗口滑动步长(控制下采样率)
        :param padding: 输入边缘填充的像素数(可选, 池化中较少使用)
        :param dilation: 窗口元素的间距(默认1, 保持连续区域)
        :return 特征矩阵
        计算公式:
        输出尺寸 = [输入尺寸 + 2*padding - dilation*(kernel_size-1) - 1] / stride + 1 向下取整
        
        比如: 参数:kernel_size=2, stride=2, padding=0
        输入:
        [[1, 2, 3, 4],              [1, 2 | 3, 4]          
         [5, 6, 7, 8],              [5, 6 | 7, 8]  
         [9, 10, 11, 12],   ----->  -----------     
         [13, 14, 15, 16]]           [9, 10 | 11, 12]  
                                    [13, 14 | 15, 16]
        左上窗口:max(1, 2, 5, 6) = 6
        右上窗口:max(3, 4, 7, 8) = 8
        左下窗口:max(9, 10, 13, 14) = 14
        右下窗口:max(11, 12, 15, 16) = 16
        
        输出:
        [[6, 8], 
        [14, 16]]
        '''

        self.conv2 = nn.Sequential( # 定义第二个卷积块, 输入形状是self.conv1的输出形状:[16, 14, 14]
            # 卷积层: 提取深层特征
            nn.Conv2d(
                in_channels=16, # 和 self.conv1中的out_channels相同
                out_channels=32, # 输出通道数(扩展特征维度至32通道)
                kernel_size=5, # 卷积核尺寸5x5(保持与第一层相同的感受野)
                stride=1, # 滑动步长1(保持特征图分辨率不变)
                padding=2, # 填充2像素(计算公式:(5-1)/2=2, 维持宽高)
            ), # 输出形状 (32, 14, 14)
            # 激活函数:引入非线性
            nn.ReLU(), # 修正线性单元(保留正值, 抑制负值)
            # 池化层:下采样关键特征
            nn.MaxPool2d(2)
            # 2x2最大池化窗口
        )# 输出形状 (32, 7, 7)
        # 全连接分类层, 将高维特征映射到类别空间
        self.out = nn.Linear(in_features=32 * 7 * 7, out_features=10)
        """
        参数详解:
        - 输入维度 32*7*7: 
          由前一层的输出决定(32个通道的7x7特征图, 展平后的向量长度 = 32×7×7 = 1568)
        - 输出维度 10: 
          对应MNIST的10个数字类别(0-9)

        功能说明:
        1. 特征整合:将卷积层提取的二维空间特征(32@7x7)转换为一维向量
        2. 分类决策:通过线性变换 + 隐含偏置项, 计算每个类别的得分(logits)
        3. 参数量:32*7*7*10 + 10 = 15, 690 个可训练参数

        数据流示例:
        输入:[batch_size, 32, 7, 7] → 展平 → [batch_size, 1568] → 线性变换 → [batch_size, 10]
        """

    def forward(self, x):
        # 前向传播过程定义
        x = self.conv1(x)  # 输入通过第一个卷积块(包含卷积、激活、池化)
        # 输入形状:[batch, 1, 28, 28] → 输出:[batch, 16, 14, 14]

        x = self.conv2(x)  # 通过第二个卷积块进一步提取特征
        # 输入形状:[batch, 16, 14, 14] → 输出:[batch, 32, 7, 7]

        x = x.view(x.size(0), -1)  # 展平操作:将多维特征图转换为一维特征向量
        # 输入形状:[batch, 32, 7, 7] → 输出:[batch, 32*7*7=1568]
        # x.size(0) 获取批次大小, -1 自动计算展平后维度

        output = self.out(x)  # 全连接层分类决策(输出各类别得分)
        # 输入形状:[batch, 1568] → 输出:[batch, 10]

        return output, x  # 返回分类结果和展平后的特征(后者用于可视化分析)

    """
    功能详解:
    1. 特征提取流水线:通过两个卷积块逐步提取从低级到高级的图像特征
    2. 维度变换:将空间特征(高度×宽度×通道数)转换为全连接层所需的向量形式
    3. 双返回值设计:
       - output:模型预测结果(未归一化的类别得分)
        - x:展平后的高维特征(可用于特征可视化、迁移学习等)

    数据流示例:
    输入 → [batch_size, 1, 28, 28] 
    → conv1 → [batch_size, 16, 14, 14] 
    → conv2 → [batch_size, 32, 7, 7] 
    → view → [batch_size, 1568] 
    → out → [batch_size, 10]
    """


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
    save_dir = './images/CNN'
    os.makedirs(save_dir, exist_ok=True)  # 自动创建目录，存在时不报错

    # 生成唯一文件名
    image_name = f'datapoints_epoch_{epoch}_step_{step}.png'
    image_path = os.path.join(save_dir, image_name)
    # 遍历每个数据点绘制带背景色的文本标签
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
    try: from sklearn.manifold import TSNE; HAS_SK = True
    except: HAS_SK = False; print('Please install sklearn for layer visualization')
    LR = 0.001
    cnn = CNN()  # 实例化自定义的卷积神经网络
    print(cnn)  # 打印网络结构（显示各层参数）

    # 使用Adam优化器: 自适应调节学习率，适合大多数深度学习任务
    optimizer = torch.optim.Adam(cnn.parameters(), # # 待优化的模型参数（通常为 model.parameters()）
                                 lr=LR
                                 )
    # 交叉熵损失函数: 专为分类任务设计，自动处理Softmax和类别概率计算
    loss_func = nn.CrossEntropyLoss()
    # training and testing
    # 遍历所有训练周期
    # plt.ion()
    for epoch in range(EPOCH):
        # 遍历所有批次数据
        for step, (batch_x, batch_y) in enumerate(train_loader):
            # batch_x: 图像数据(形状:[batch_size, 1, 28, 28])
            # batch_y: 对应标签 (形状:[batch_size])
            # === 核心训练步骤 ===
            output = cnn(batch_x)[0] # 前向传播（获取预测结果）
            loss = loss_func(output, batch_y) # 计算损失
            optimizer.zero_grad() # 清空梯度缓存
            loss.backward() # 反向传播计算梯度
            optimizer.step() # 更新网络参数

            # 验证和可视化
            if step % 50 == 0:
                # 获取测试集预测结果和最后一层特征
                test_output, last_layer = cnn(test_x)
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
    # plt.ioff()
    return cnn


def test(cnn_model, test_x, test_y):
    # 对前10个测试样本预测
    test_output, _ = cnn_model(test_x[:10])
    # 获取预测数字
    pred_y = torch.max(test_output, 1)[1].data.numpy()
    # 打印预测结果
    print(pred_y, 'prediction number')
    # 打印真实标签
    print(test_y[:10].numpy(), 'real number')


if __name__ == '__main__':
    train_data = prepare_data()
    train_loader, test_x, test_y = data_load(train_data)
    cnn_model = train(train_loader, test_x, test_y)
    test(cnn_model, test_x, test_y)
import torch
import torchvision
import torchvision.transforms as transforms
from torch import nn
from torch.autograd import Variable

import LSTMnet

"""RNN网络公式 ht=tanh(Wih * Xt + Bih + Whh * Ht-1 + Bhh)"""
basic_rnn = nn.RNN(
    input_size=20,
    hidden_size=50,
    num_layers=10,
)
print(basic_rnn.weight_ih_l0.size())  # 因为输入是20，要求输出是50，所以第一层权重应该是50x20的
print(basic_rnn.weight_hh_l0.size())  # whh是与ht相乘，h的size是50，所以whh与h相乘后
# 得到的新h要保持size不变
# 所以也是50x50
print(basic_rnn.bias_ih_l0.size())
test_input = Variable(torch.randn(100, 32, 20))  # 长100 批量（batch）32 维度20
# 批量32可以理解为一次性拿32个样本进行输入。长度100可以当作epoch为100次
h_0 = Variable(torch.randn(10, 32, 50))  # layer*direction,batch,hidden
test_output, h_n = basic_rnn(test_input, h_0)
out = test_output[-1, :, :]
print(out)
print(out.size())
print(test_output.size())
print(h_n.size())
'将数据标准化为(-1,1)'
transform = transforms.Compose(
    [transforms.ToTensor()]  # (x-mean)/std
)
# 创建训练集
# root -- 数据存放的目录
# train -- 明确是否是训练集
# download -- 是否需要下载
# transform -- 转换器，将数据集进行转换
trainset = torchvision.datasets.MNIST(root='./data',
                                      train=True, transform=transform, download=True)
# 创建数据加载器
# trianset/testset --数据集
# batch_size --一次输入的图片数目
# shuffle --是否打乱顺序
trainloader = torch.utils.data.DataLoader(trainset, batch_size=100, shuffle=True)

# 创建测试集
testset = torchvision.datasets.MNIST(root='./data',
                                     train=False, transform=transform, download=True
                                     )

# 创建测试集数据加载器
testloader = torch.utils.data.DataLoader(testset, batch_size=100, shuffle=True)

net = LSTMnet.rnnNet(in_dim=28, hidden_dim=50, n_layer=10, n_class=10)
optimizer = torch.optim.Adam(net.parameters(), lr=0.001)
criterion = nn.CrossEntropyLoss()
for epoch in range(20):
    running_loss = 0.0
    for i, data in enumerate(trainloader, start=0):
        inputs, labels = data  # data由输入和标签（真实值）组成
        optimizer.zero_grad()  # 所有梯度清零
        outputs = net(inputs.view(-1,28,28))  # 正向传播
        loss = criterion(outputs, labels)  # 计算损失
        loss.backward()  # 反向传播
        optimizer.step()  # 更新模型

        # print statistics
        running_loss += loss.item()
        if i % 20 == 19:  # print every 2000 mini-batches
            print('[%d, %5d] loss: %.3f' % (epoch + 1, i + 1, running_loss / 20))
            running_loss = 0.0
print('Finished')

import torch
# import torchvision as tv
import matplotlib.pyplot as plt
import shutil
import os
import time
from torch.utils.data import Dataset
from torchvision import datasets, transforms

from PIL import Image
# 现在test文件的形式不方便处理，将test中的图片全部提取出来，并整理成好处理的形式
old_test = "./test"
new_test = "./new_test"
# 如果目标文件new_Test不存在，则创建
os.makedirs("./new_test", False)
# 创建每个类的文件
os.makedirs("./new_test/baihe")
os.makedirs("./new_test/dangshen")
os.makedirs("./new_test/gouqi")
os.makedirs("./new_test/huaihua")
os.makedirs("./new_test/jinyinhua")
# 遍历旧测试集中的文件，转移到对应的类中
for filename in os.listdir(old_test):
    source_path = os.path.join(old_test, filename)
    if "baihe" in filename:
        dest_path = "./new_test/baihe"
    elif "dangshen" in filename:
        dest_path = "./new_test/dangshen"
    elif "gouqi" in filename:
        dest_path = "./new_test/gouqi"
    elif "huaihua" in filename:
        dest_path = "./new_test\huaihua"
    elif "jinyinhua" in filename:
        dest_path = "./new_test/jinyinhua"
    dest_path = os.path.join(dest_path, filename)
    shutil.move(source_path, dest_path)

# 初始化loss,loss_list,correct
loss_total = 0
correct = 0
loss_list = list()
acc_list = list()

# 读取训练集
train_dir = "./train"
# 裁剪图片为统一尺寸
train_transform = transforms.Compose([transforms.Resize([64, 64]), transforms.ToTensor()])
# 加载训练集
train_dataset = datasets.ImageFolder(root="./train", transform=train_transform)
train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=40, shuffle=True)

# 读取测试集
test_dir = "./new_test"
test_transform = transforms.Compose([transforms.Resize([64, 64]), transforms.ToTensor()])
# 加载测试集
test_dataset = datasets.ImageFolder(root="./new_test", transform=test_transform)
test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=10, shuffle=False)


#模型
class Model(torch.nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.Conv = torch.nn.Sequential(  # 输入3*64*64
            torch.nn.Conv2d(
                in_channels=3,
                out_channels=64,
                kernel_size=3,
                stride=1,
                padding=1
            ),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(kernel_size=2)  # 输出64*32*32
        )
        self.Conv1 = torch.nn.Sequential(
            torch.nn.Conv2d(
                in_channels=64,
                out_channels=128,
                kernel_size=3,
                stride=1,
                padding=1
            ),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(kernel_size=2)  # 输出128*16*16
        )
        self.out = torch.nn.Sequential(
            torch.nn.Linear(128 * 16 * 16, 512),
            torch.nn.ReLU(),
            torch.nn.Dropout(p=0.5),
            torch.nn.Linear(512, 5)
        ) 

# 前向传播
    def forward(self, x):
        x = self.Conv(x)
        x = self.Conv1(x)
        x = x.view(x.size(0), -1)
        out = self.out(x)
        return out, x


model = Model()
print(model)
# 遍历次数
traverse_time = 30
# 开始时间
time_start = time.time()
# loss
loss_function = torch.nn.CrossEntropyLoss()
# optimizer
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

Use_gpu = torch.cuda.is_available()
if Use_gpu:
    model = model.cuda()

for traverse in range(traverse_time):
    # 如果是训练模式，就进行训练；否则不进行训练
    for batch, data in enumerate(train_loader, 0):
        if batch % 40 != 0:
            # print("train")
            model.train(True)
            X, Y = data
            # 放到GPU上
            X = X.cuda()
            Y = Y.cuda()
            # 搭建模型
            output = model(X)[0]
            # 得到标签值
            predict = torch.max(output, 1)[1]
            # 收回到CPU上
            predict = predict.cpu().numpy()
            # 梯度归零
            optimizer.zero_grad()
            # 计算损失
            loss = loss_function(output, Y)
            loss_list.append(loss.item())
            # 反向传播
            loss.backward()
            # 梯度更新
            optimizer.step()
            # 计算loss
            loss_total = loss_total + loss.data.item()
            correct = correct + (predict == Y.data.detach().cpu().numpy()).astype(int).sum().item()
            accuracy = float(correct / Y.size(0))
            acc_list.append(accuracy)
            correct = 0
        else:
            print("test")
            model.train(False)
            for batch_test, data_test in enumerate(test_loader, 0):
                X, Y = data_test
                # 放到GPU上
                X = X.cuda()
                Y = Y.cuda()
                # 搭建模型
                output = model(X)[0]
                # 得到标签值
                predict = torch.max(output, 1)[1]
                # 收回到CPU
                predict = predict.cpu().numpy()
                # 梯度归零
                optimizer.zero_grad()
                # 计算损失
                loss = loss_function(output, Y)
                # 计算loss
                loss_total = loss_total + loss.data.item()
                accuracy = float((predict == Y.data.detach().cpu().numpy()).astype(int).sum()) / float(Y.size(0))
                acc_list.append(accuracy)
                print('Epoch: ', traverse, ' test accuracy:', accuracy)
time_end = time.time() - time_start
print("run time:", time_end)

# 最后再测试一次
test_count = 0
test_sum = 0
for step, data in enumerate(test_loader):
    X, Y = data
    X = X.cuda()
    Y = Y.cuda()
    output, last_layer = model(X)
    predict = torch.max(output, 1)[1].data.detach().cpu().numpy()
    test_sum += (Y.size(0))
    test_count += (predict == Y.data.detach().cpu().numpy()).astype(int).sum()
print("final_test_accuracy:", float(test_count) / test_sum, "(correct=", test_count, ")")

# 画图
figure1 = plt.figure(1)
plt.xlabel('train and test times')
plt.ylabel('loss')
y = loss_list
x = list(range(len(loss_list)))
plt.plot(x, y, color='black', label='Loss')

fig2 = plt.figure(2)
plt.xlabel("train and test times")
plt.ylabel("Accuracy")
y = acc_list
x = list(range(len(acc_list)))
plt.plot(x, y, color='black', label='Accuracy')
# 显示
plt.show()

import os
import torch
import torch.nn as nn
from torch.nn import DataParallel
# from torchvision.ops import SupervisedContrastiveLoss
from torch.utils.data import DataLoader
from tensorboardX import SummaryWriter
from torchvision import transforms
import numpy as np
import math
import shutil
import csv

from torch.nn import functional as F

import data_utils.transform as tr
from data_utils.data_loader import DataGenerator

from utils import dfs_remove_weight
from torch.cuda.amp import autocast, GradScaler


# GPU version.


# VolumeClassifier 类定义了一个完整的深度学习模型训练流水线，
# 包括网络设置、数据预处理、损失计算、优化、模型保存等。
# 它还支持学习率调度、早停和训练过程中的日志记录等高级功能。
class VolumeClassifier(object):
    '''
    Control the training, evaluation, and inference process.
    Args:
    - net_name: string, __all__ = ["resnet18", "resnet34", "resnet50",...].
    - lr: float, learning rate.
    - n_epoch: integer, the epoch number
    - num_classes: integer, the number of class
    - image_size: integer, input size
    - batch_size: integer
    - num_workers: integer, how many subprocesses to use for data loading.
    - device: string, use the specified device
    - pre_trained: True or False, default False
    '''

    def __init__(self,
                 net_name=None,
                 lr=1e-3,
                 n_epoch=1,
                 num_classes=3,
                 image_size=None,
                 batch_size=6,
                 train_mean=0,
                 train_std=0,
                 num_workers=0,
                 device=None,
                 pre_trained=True,
                 weight_decay=0.,
                 momentum=0.95,
                 gamma=0.1,
                 milestones=[40, 80],
                 T_max=5,
                 use_fp16=True,
                 dropout=0.01,
                 use_supcon=False,
                 supcon_weight=0.1):
        super(VolumeClassifier, self).__init__()

        # 初始化模型参数
        self.net_name = net_name
        self.lr = lr
        self.n_epoch = n_epoch
        self.num_classes = num_classes
        self.image_size = image_size
        self.batch_size = batch_size
        self.train_mean = train_mean
        self.train_std = train_std

        self.num_workers = num_workers
        self.device = device

        self.pre_trained = pre_trained
        self.start_epoch = 0
        self.global_step = 0
        self.loss_threshold = 1.0
        self.metric_threshold = 0.0
        # 保存中间输出
        self.feature_in = []
        self.feature_out = []

        # 优化器参数设置
        self.weight_decay = weight_decay
        self.momentum = momentum
        self.gamma = gamma
        self.milestones = milestones
        self.T_max = T_max
        self.use_fp16 = use_fp16
        # self.use_fp16 = False
        self.dropout = dropout
        self.use_supcon = use_supcon
        self.supcon_weight = supcon_weight

        os.environ['CUDA_VISIBLE_DEVICES'] = self.device

        # 获取网络模型
        self.net = self._get_net(self.net_name, self.pre_trained)

    # 训练函数
    def trainer(self,
                train_path,
                val_path,
                label_dict,
                use_supcon,
                supcon_weight,
                output_dir=None,
                log_dir=None,
                optimizer='Adam',
                loss_fun='Cross_Entropy',
                class_weight=None,
                lr_scheduler=None,
                cur_fold=0):
        # 设置随机种子以确保可重复性
        torch.manual_seed(0)
        np.random.seed(0)
        torch.cuda.manual_seed_all(0)
        print('Device:{}'.format(self.device))
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.enabled = True
        torch.backends.cudnn.benchmark = True
        # 创建日志和输出目录
        log_dir = os.path.join(log_dir, f'fold{str(cur_fold)}')
        output_dir = os.path.join(output_dir, f'fold{str(cur_fold)}')

        if not os.path.exists(log_dir):
            os.makedirs(log_dir)

        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        # 初始化TensorBoard记录器
        self.writer = SummaryWriter(log_dir)
        self.global_step = self.start_epoch * math.ceil(
            len(train_path) / self.batch_size)

        net = self.net
        lr = self.lr
        loss = self._get_loss(loss_fun, class_weight)
        loss_sup = self._get_SupervisedContrastiveLoss(temperature=0.07)
        print(use_supcon)
        # 如果使用多GPU，则使用DataParallel
        if len(self.device.split(',')) > 1:
            net = DataParallel(net)

        # 数据预处理设置
        train_transformer = transforms.Compose([
            tr.ToCVImage(),
            tr.RandomResizedCrop(size=self.image_size, scale=(1.0, 1.0), ratio=(3 / 4, 4 / 3)),
            # tr.RandomHorizontalFlip(),
            tr.ToTensor(),
            tr.Normalize(self.train_mean, self.train_std)
        ])

        val_transformer = transforms.Compose([
            tr.ToCVImage(),
            tr.Resize(resized=self.image_size),
            tr.ToTensor(),
            tr.Normalize(self.train_mean, self.train_std)
        ])
        train_dataset = DataGenerator(train_path,
                                      label_dict,
                                      transform=train_transformer)

        train_loader = DataLoader(train_dataset,
                                  batch_size=self.batch_size,
                                  shuffle=True,
                                  num_workers=self.num_workers,
                                  pin_memory=True)

        val_dataset = DataGenerator(val_path,
                                    label_dict,
                                    transform=val_transformer)

        val_loader = DataLoader(val_dataset,
                                batch_size=self.batch_size,
                                shuffle=False,
                                num_workers=self.num_workers,
                                pin_memory=True)

        # 将模型和损失函数移动到GPU
        net = net.cuda()
        loss = loss.cuda()

        # 优化器设置
        optimizer = self._get_optimizer(optimizer, net, lr)
        scaler = GradScaler()
        # 学习率调度器设置
        if lr_scheduler is not None:
            lr_scheduler = self._get_lr_scheduler(lr_scheduler, optimizer)
        # 早停设置
        early_stopping = EarlyStopping(patience=50,
                                       verbose=True,
                                       monitor='val_acc',
                                       best_score=self.metric_threshold,
                                       op_type='max')

        # 开始正式训练
        for epoch in range(self.start_epoch, self.n_epoch):

            # train_loss, train_acc = self._train_on_epoch(epoch, net, loss, optimizer, train_loader, scaler)
            train_loss, train_acc = self._train_on_epoch(epoch, net, loss, loss_sup, use_supcon, supcon_weight,
                                                         optimizer, train_loader, scaler)
            # torch.cuda.empty_cache()

            val_loss, val_acc = self._val_on_epoch(epoch, net, loss, val_loader)

            # 更新学习率
            if lr_scheduler is not None:
                lr_scheduler.step()
            # 打印训练和验证损失
            print('epoch:{},train_loss:{:.5f},val_loss:{:.5f}'.format(epoch, train_loss, val_loss))
            # 打印训练和验证精度
            print('epoch:{},train_acc:{:.5f},val_acc:{:.5f}'.format(epoch, train_acc, val_acc))
            # 记录训练和验证损失到TensorBoard
            self.writer.add_scalars('data/loss', {
                'train': train_loss,
                'val': val_loss
            }, epoch)
            # 记录训练和验证精度到TensorBoard
            self.writer.add_scalars('data/acc', {
                'train': train_acc,
                'val': val_acc
            }, epoch)
            self.writer.add_scalar('data/lr', optimizer.param_groups[0]['lr'], epoch)

            early_stopping(val_acc)
            # 如果验证精度达到阈值，保存模型
            if val_acc > self.metric_threshold:
                self.metric_threshold = val_acc

                if len(self.device.split(',')) > 1:
                    state_dict = net.module.state_dict()
                else:
                    state_dict = net.state_dict()

                saver = {
                    'epoch': epoch,
                    'save_dir': output_dir,
                    'state_dict': state_dict,
                    'optimizer': optimizer.state_dict()
                }

                file_name = 'epoch={}-train_loss={:.5f}-val_loss={:.5f}-train_acc={:.5f}-val_acc={:.5f}.pth'.format(
                    epoch, train_loss, val_loss, train_acc, val_acc)
                print('Save as :', file_name)
                save_path = os.path.join(output_dir, file_name)

                torch.save(saver, save_path)

            csv_path = os.path.join(output_dir, "train_metrics.csv")
            write_header = not os.path.exists(csv_path)

            with open(csv_path, mode="a", newline='') as file:
                writer = csv.writer(file)
                if write_header:
                    writer.writerow(["epoch", "train_loss", "val_loss", "train_acc", "val_acc"])
                writer.writerow([epoch, train_loss, val_loss, train_acc, val_acc])

            # 如果早停条件满足，停止训练
            if early_stopping.early_stop:
                print('Early Stopping!')
                break

        self.writer.close()
        # 清理权重文件，只保留最好的5个模型
        dfs_remove_weight(output_dir, 5)

    # 训练一个epoch的函数
    def _train_on_epoch(self, epoch, net, criterion, loss_sup, use_supcon, supcon_weight, optimizer, train_loader,
                        scaler):
        net.train()

        train_loss = AverageMeter()
        train_acc = AverageMeter()

        for step, sample in enumerate(train_loader):

            data = sample['image']
            target = sample['label']

            data = data.cuda()
            target = target.cuda()

            if torch.isnan(data).any() or torch.isinf(data).any():
                print("nan of inf input", step)

            #             if self.use_fp16:
            #                 with autocast():
            #                     output = net(data)
            #                     loss = criterion(output, target)

            #                 optimizer.zero_grad()
            #                 # 使用scaler进行梯度缩放
            #                 scaler.scale(loss).backward()
            #                 scaler.step(optimizer)
            #                 scaler.update()
            #             else:
            #                 output = net(data)
            #                 loss = criterion(output, target)

            #                 optimizer.zero_grad()
            #                 loss.backward()
            #                 optimizer.step()
            if self.use_fp16:
                with autocast():
                    if use_supcon:
                        # print("supcon")
                        output, features = net(data, return_features=True)
                        ce_loss = criterion(output, target)
                        supcon_loss = loss_sup(features, target)
                        loss = ce_loss + supcon_weight * supcon_loss
                    else:
                        # print("loss")
                        output = net(data)
                        loss = criterion(output, target)

                optimizer.zero_grad()
                # 使用scaler进行梯度缩放
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                if use_supcon:
                    # print("no fp16")
                    # print("supcon")
                    output, features = net(data, return_features=True)
                    ce_loss = criterion(output, target)
                    supcon_loss = loss_sup(features, target)
                    loss = ce_loss + supcon_weight * supcon_loss
                else:
                    # print("no fp16")
                    # print("loss")
                    output = net(data)
                    loss = criterion(output, target)
            output = F.softmax(output, dim=1)
            # output = output.float()
            # loss = loss.float()

            # 计算精度并记录损失
            acc = accuracy(output, target)
            train_loss.update(loss.item(), data.size(0))
            train_acc.update(acc, data.size(0))

            # torch.cuda.empty_cache()

            print('epoch:{},step:{},train_loss:{:.5f},train_acc:{:.5f},lr:{}'.
                  format(epoch, step, loss.item(), acc,
                         optimizer.param_groups[0]['lr']))

            if self.global_step % 10 == 0:
                self.writer.add_scalars('data/train_loss_acc', {
                    'train_loss': loss.item(),
                    'train_acc': acc
                }, self.global_step)

            self.global_step += 1

        return train_loss.avg, train_acc.avg

        # 验证一个epoch的函数

    def _val_on_epoch(self, epoch, net, criterion, val_loader):

        net.eval()

        val_loss = AverageMeter()
        val_acc = AverageMeter()

        with torch.no_grad():
            for step, sample in enumerate(val_loader):
                data = sample['image']
                target = sample['label']

                data = data.cuda()
                target = target.cuda()

                output = net(data)
                loss = criterion(output, target)

                output = F.softmax(output, dim=1)
                # output = output.float()
                # loss = loss.float()

                # 计算精度并记录损失
                acc = accuracy(output, target)
                print('epoch:{},step:{},val_loss:{:.5f},val_acc:{:.5f}'.format(epoch, step, loss.item(), acc))
                val_loss.update(loss.item(), data.size(0))
                val_acc.update(acc, data.size(0))

                # torch.cuda.empty_cache()

        return val_loss.avg, val_acc.avg

    # 钩子函数是用来获取神经网络中间层特征的。
    # 当网络前向传播时，钩子函数会被调用，并返回输入和输出的特征。
    # 在这里，它保存了输入特征和输出特征，主要用于分析模型的行为或者调试。
    def hook_fn_forward(self, module, input, output):

        for i in range(input[0].size(0)):
            self.feature_in.append(input[0][i].cpu().numpy())
            self.feature_out.append(output[i].cpu().numpy())

    # 推理函数用于加载测试数据，执行推理过程，计算模型在测试集上的准确率，并返回预测结果。
    def inference(self,
                  test_path,
                  label_dict,
                  net=None,
                  hook_fn_forward=False):

        if net is None:
            net = self.net

        if hook_fn_forward:
            net.layer4.register_forward_hook(self.hook_fn_forward)

        net = net.cuda()
        net.eval()

        test_transformer = transforms.Compose([
            tr.ToCVImage(),
            tr.RandomResizedCrop(size=self.image_size, scale=(1.0, 1.0)),
            tr.ToTensor(),
            tr.Normalize(self.train_mean, self.train_std)
        ])

        test_dataset = DataGenerator(test_path,
                                     label_dict,
                                     transform=test_transformer)

        test_loader = DataLoader(test_dataset,
                                 batch_size=self.batch_size,
                                 shuffle=False,
                                 num_workers=self.num_workers,
                                 pin_memory=True)

        result = {'true': [], 'pred': [], 'prob': []}

        test_acc = AverageMeter()

        with torch.no_grad():
            for step, sample in enumerate(test_loader):
                data = sample['image']
                target = sample['label']

                data = data.cuda()
                target = target.cuda()  # N
                output = net(data)
                output = F.softmax(output, dim=1)
                # output = output.float()  #N*C

                acc = accuracy(output, target)
                test_acc.update(acc, data.size(0))

                result['true'].extend(target.detach().tolist())
                result['pred'].extend(torch.argmax(output, 1).detach().tolist())
                result['prob'].extend(output.detach().tolist())

                print('step:{},test_acc:{:.5f}'.format(step, acc))

                torch.cuda.empty_cache()

        print('average test_acc:{:.5f}'.format(test_acc.avg))

        return result, np.array(self.feature_in), np.array(self.feature_out)

    # 获取网络模型的函数，根据提供的 net_name 加载对应的网络模型，
    # 并根据需要修改最后一层（全连接层）以适应当前的分类任务（即 num_classes）
    def _get_net(self, net_name, pretrained):
        if net_name.startswith('res') or net_name.startswith('wide_res'):
            import model.resnet as resnet
            from model.resnet import ResNetWithFeatures
            print("pre_trained:", pretrained)
            # if net_name in ['resnet18', 'resnet34', 'resnet50']:

            # net = ResNetWithFeatures(base_name=net_name,
            #                  num_classes=self.num_classes,
            #                  pretrained=pretrained)
            net = resnet.__dict__[net_name](
                pretrained=pretrained,
            )
            new_fc = nn.Linear(net.fc.in_features, self.num_classes)
            net.fc = new_fc

        elif net_name.startswith('vit_'):
            import model.vision_transformer as vit
            net = vit.__dict__[net_name](
                pretrained=pretrained,
                image_size=self.image_size,
                dropout=self.dropout
            )
            new_fc = nn.Linear(net.heads.head.in_features, self.num_classes)
            net.heads.head = new_fc
        print("load model")
        return net

    # 获取损失函数的函数
    def _get_loss(self, loss_fun, class_weight=None):
        if class_weight is not None:
            class_weight = torch.tensor(class_weight)

        if loss_fun == 'Cross_Entropy':
            loss = nn.CrossEntropyLoss(class_weight)
        return loss

    def _get_SupervisedContrastiveLoss(self, temperature=0.07):
        return SupervisedContrastiveLoss(temperature)

    # 获取优化器的函数
    def _get_optimizer(self, optimizer, net, lr):
        if optimizer == 'Adam':
            optimizer = torch.optim.Adam(net.parameters(),
                                         lr=lr,
                                         weight_decay=self.weight_decay)

        elif optimizer == 'SGD':
            optimizer = torch.optim.SGD(net.parameters(),
                                        lr=lr,
                                        momentum=self.momentum,
                                        weight_decay=self.weight_decay)

        elif optimizer == 'AdamW':
            optimizer = torch.optim.AdamW(net.parameters(),
                                          lr=lr, weight_decay=self.weight_decay)

        return optimizer

    # 获取学习率调度器的函数
    def _get_lr_scheduler(self, lr_scheduler, optimizer):
        if lr_scheduler == 'ReduceLROnPlateau':
            lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer, mode='min', patience=5, verbose=True)
        elif lr_scheduler == 'MultiStepLR':
            lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(
                optimizer, self.milestones, gamma=self.gamma)
        elif lr_scheduler == 'CosineAnnealingLR':
            lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer, T_max=self.T_max)
        elif lr_scheduler == 'CosineAnnealingWarmRestarts':
            lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
                optimizer, 20, T_mult=2)

        return lr_scheduler

    # def _get_pre_trained(self, weight_path):
    #     checkpoint = torch.load(weight_path)
    #     self.net.load_state_dict(checkpoint['state_dict'])
    #     self.start_epoch = checkpoint['epoch'] + 1


class SupervisedContrastiveLoss(nn.Module):
    def __init__(self, temperature=0.1):
        super(SupervisedContrastiveLoss, self).__init__()
        self.temperature = temperature

    def forward(self, features, labels):
        features = nn.functional.normalize(features, p=2, dim=1)
        device = features.device
        labels = labels.contiguous().view(-1, 1)
        mask = torch.eq(labels, labels.T).float().to(device)

        contrast = torch.div(torch.matmul(features, features.T), self.temperature)
        logits_max, _ = torch.max(contrast, dim=1, keepdim=True)
        logits = contrast - logits_max.detach()

        logits_mask = torch.ones_like(mask) - torch.eye(mask.size(0), device=device)
        mask = mask * logits_mask

        exp_logits = torch.exp(logits) * logits_mask
        log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True) + 1e-12)

        mean_log_prob_pos = (mask * log_prob).sum(1) / (mask.sum(1) + 1e-12)

        loss = -mean_log_prob_pos.mean()
        return loss


# 计算工具类

# 用于计算和存储一组数值的平均值和当前值。
class AverageMeter(object):
    '''
    Computes and stores the average and current value
    '''

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


import torch


# 用于计算分类任务的准确率。
def accuracy(scores, target):
    """
    计算分类任务的精度。

    参数:
    scores (torch.Tensor): 经过softmax后的分类分数，形状为 [batch_size, num_classes]。
    target (torch.Tensor): 真实标签，形状为 [batch_size]。

    返回:
    float: 精度，范围在 [0, 1] 之间。
    """
    # 获取预测的类别（即分数最大的类别）
    _, predicted = torch.max(scores, dim=1)

    # 计算预测正确的数量
    correct = (predicted == target).sum().item()

    # 计算精度
    accuracy = correct / target.size(0)

    return accuracy


# 用于在训练过程中监控模型的性能，如果模型在若干个训练周期内没有明显改善（即性能指标没有提升），则提前停止训练。
class EarlyStopping(object):
    """Early stops the training if performance doesn't improve after a given patience."""

    def __init__(self,
                 patience=10,
                 verbose=True,
                 delta=0,
                 monitor='val_loss',
                 best_score=None,
                 op_type='min'):
        """
        Args:
            patience (int): How long to wait after last time performance improved.
                            Default: 10
            verbose (bool): If True, prints a message for each performance improvement.
                            Default: True
            delta (float): Minimum change in the monitored quantity to qualify as an improvement.
                            Default: 0
            monitor (str): Monitored variable.
                            Default: 'val_loss'
            op_type (str): 'min' or 'max'
        """
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = best_score
        self.early_stop = False
        self.delta = delta
        self.monitor = monitor
        self.op_type = op_type

        if self.op_type == 'min':
            self.val_score_min = np.Inf
        else:
            self.val_score_min = 0

    def __call__(self, val_score):

        score = -val_score if self.op_type == 'min' else val_score

        if self.best_score is None:
            self.best_score = score
            self.print_and_update(val_score)
        elif score < self.best_score + self.delta:
            self.counter += 1
            print(
                f'EarlyStopping counter: {self.counter} out of {self.patience}'
            )
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.print_and_update(val_score)
            self.counter = 0

    def print_and_update(self, val_score):
        '''print_message when validation score decrease.'''
        if self.verbose:
            print(
                self.monitor,
                f'optimized ({self.val_score_min:.6f} --> {val_score:.6f}).  Saving model ...'
            )
        self.val_score_min = val_score


import json
import torch
import torch.utils.data as Data
from torch import nn, optim
import numpy as np
import time
from tqdm import tqdm
from gpt_model import *
import os


# 加上<sep>符号
def make_data(datas):
    train_datas =[]
    for data in datas:
        data=data.strip()
        train_data = [i if i!='\t' else "<sep>" for i in data]+['<sep>']
        train_datas.append(train_data)

    return train_datas


class MyDataSet(Data.Dataset):
    # 是类的构造方法，接受一个参数 datas，是数据集的主体，它存储了所有样本的数据
    def __init__(self,datas):
        self.datas = datas

    def __getitem__(self, item):
        data = self.datas[item]
        # 解码器的输入
        decoder_input = data[:-1]
        # 标准输出（去掉第一列是为了和输入对齐）
        decoder_output = data[1:]
        # 计算解码器输入和标准输出的长度
        decoder_input_len = len(decoder_input)
        decoder_output_len = len(decoder_output)
        # 方法的返回值是一个字典，包含了 decoder_input、
        # decoder_input_len、decoder_output 和 decoder_output_len，这些是模型训练或推理时所需要的输入和输出
        return {"decoder_input":decoder_input,"decoder_input_len":decoder_input_len,
                "decoder_output":decoder_output,"decoder_output_len":decoder_output_len}

    def __len__(self):
        return len(self.datas)

    # 返回填充后的解码器输入和输出
    def padding_batch(self,batch):
        decoder_input_lens = [d["decoder_input_len"] for d in batch]
        decoder_output_lens = [d["decoder_output_len"] for d in batch]

        decoder_input_maxlen = max(decoder_input_lens)
        decoder_output_maxlen = max(decoder_output_lens)


        for d in batch:
            d["decoder_input"].extend([word2id["<pad>"]]*(decoder_input_maxlen-d["decoder_input_len"]))
            d["decoder_output"].extend([word2id["<pad>"]]*(decoder_output_maxlen-d["decoder_output_len"]))
        decoder_inputs = torch.tensor([d["decoder_input"] for d in batch],dtype=torch.long)
        decoder_outputs = torch.tensor([d["decoder_output"] for d in batch],dtype=torch.long)

        return decoder_inputs,decoder_outputs


# 计算一个训练过程或任务的耗时
def epoch_time(start_time, end_time):
    elapsed_time = end_time - start_time
    elapsed_mins = int(elapsed_time / 60)
    elapsed_secs = int(elapsed_time - (elapsed_mins * 60))
    return elapsed_mins, elapsed_secs


def train_step(model,data_loader,optimizer,criterion,clip=1,print_every=None):
    # 设置模型为训练模式
    model.train()

    if print_every == 0:
        print_every = 1

    print_loss_total = 0  # 每次打印都重置

    epoch_loss = 0

# dec_inputs 和 dec_outputs，分别是解码器输入和输出
    # 按批次（batch）加载数据
    # tqdm 是用于显示进度条
    for i, (dec_inputs, dec_outputs) in enumerate(tqdm(data_loader)):
        '''
        dec_inputs: [batch_size, tgt_len]
        dec_outputs: [batch_size, tgt_len]
        '''
        # 将优化器的梯度清零
        optimizer.zero_grad()
        # 解码器输入和输出转移到合适的设备
        dec_inputs, dec_outputs = dec_inputs.to(device), dec_outputs.to(device)
        # outputs: [batch_size * tgt_len, tgt_vocab_size]
        # 前向传播
        # outputs 是模型的输出，形状通常是 [batch_size, tgt_len, vocab_size]，即预测的每个位置的词汇概率分布
        outputs, dec_self_attns = model(dec_inputs)
        # 计算损失
        loss = criterion(outputs, dec_outputs.view(-1))
        # 累积损失,是用来打印的
        print_loss_total += loss.item()
        epoch_loss += loss.item()
        # 反向传播
        loss.backward()
        # 梯度裁剪
        torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
        #  使用反向传播计算的梯度来更新模型的权重
        optimizer.step()
        # 打印损失信息
        if print_every and (i + 1) % print_every == 0:
            print_loss_avg = print_loss_total / print_every
            print_loss_total = 0
            print('\tCurrent Loss: %.4f' % print_loss_avg)
    # 返回平均损失
    return epoch_loss / len(data_loader)


def train(model, data_loader):
    # 定义损失函数
    criterion = nn.CrossEntropyLoss(ignore_index=0).to(device)
    # 设置学习率
    optimizer = optim.Adam(model.parameters(), lr=1e-4)
    start_epoch = 0

    if os.path.exists('GPT2.pt'):
        checkpoint = torch.load('GPT2.pt')
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_epoch = checkpoint['epoch']  # 恢复epoch

    for epoch in range(start_epoch, epochs):
        start_time = time.time()
        # 训练
        train_loss = train_step(model, data_loader, optimizer, criterion, CLIP, print_every=10)
        end_time = time.time()
        # 保存训练参数
        if epoch % 3 == 0:
            torch.save({
                'epoch': epoch + 1,  # 保存当前epoch
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
            }, 'GPT2.pt')

        epoch_mins, epoch_secs = epoch_time(start_time, end_time)
        # 打印轮数和这一轮训练的用时
        print(f'Epoch: {epoch + 1:02} | Time: {epoch_mins}m {epoch_secs}s')
        print(f'\tTrain Loss: {train_loss:.3f}')


def print_num_parameters(model):
    # Find total parameters and trainable parameters
    total_params = sum(p.numel() for p in model.parameters())
    print(f'{total_params:,} total parameters.')
    total_trainable_params = sum(

        p.numel() for p in model.parameters() if p.requires_grad)
    print(f'{total_trainable_params:,} training parameters.')


# 第一步：打开处理后的dataset.txt，并读取其中数据，存储到datas里
# 第二步：调用make_data，替换掉 \t 字符并添加 <sep>
# 第三步：将word2id中的单词按行（即一个短对话）转换成对应的id，存储到train_num_data中
# 第四步：创建MyDataSet对象dataset。dataset中存储的data是train_num_data。
# 第五步：使用dataloader批量处理数据，batch_size 控制了每个批次包含多少个样本，collate_fn使用了自定义的批次处理函数 padding_batch
# dataloader按批次（batch_size）加载数据集 dataset 中的样本，padding_batch把每个批次填充到相同长度
# 第六步：搭建模型
# 第七步：训练
if __name__ == '__main__':
    with open('dataset.txt', 'r', encoding='utf-8') as f:
        datas = f.readlines()

    train_data = make_data(datas)
    train_num_data = [[word2id[word] for word in line] for line in train_data]
    batch_size = 8
    epochs = 30
    dataset = MyDataSet(train_num_data)
    data_loader = Data.DataLoader(dataset, batch_size=batch_size, collate_fn=dataset.padding_batch)
    print(torch.__version__)
    print(torch.cuda.is_available())

    model = GPT().to(device)
    print(model)
    # model.load_state_dict(torch.load('GPT2.pt'))

    train(model,data_loader)


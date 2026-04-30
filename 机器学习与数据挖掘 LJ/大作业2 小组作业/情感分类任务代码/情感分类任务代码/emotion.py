import json
import pandas as pd
import torch
import time
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from torch.optim import Adam
from sklearn.model_selection import train_test_split
from gpt_model import *
from train import epoch_time

# 加上<sep>符号
def make_data(dataframe):
    train_datas = []
    datas = dataframe['review'].tolist()
    labels = dataframe['label'].tolist()
    for i, data in enumerate(datas):  # 使用enumerate来同时获取索引和数据
        data = str(data)
        train_data = [word if word != ' ' else "<sep>" for word in data] + ['<sep>']
        train_datas.append((train_data, labels[i]))  # 保存文本和对应的标签
    return train_datas

class EmoDataSet(Data.Dataset):
    def __init__(self, datas):
        self.datas = datas

    def __getitem__(self, item):
        sentence = self.datas[item]
        label = sentence[-1]
        decoder_input = sentence[:-2]  # 除去最后的label和<sep>
        decoder_output = sentence[1:-1]  # 从第二个词开始
        decoder_input_len = len(decoder_input)
        decoder_output_len = len(decoder_output)
        return {
            "decoder_input": decoder_input,
            "decoder_input_len": decoder_input_len,
            "decoder_output": decoder_output,
            "decoder_output_len": decoder_output_len,
            "label": label  # 这里返回label
        }

    def __len__(self):
        return len(self.datas)

    def padding_batch(self, batch):
        decoder_input_lens = [d["decoder_input_len"] for d in batch]
        decoder_output_lens = [d["decoder_output_len"] for d in batch]
        decoder_input_maxlen = max(decoder_input_lens)
        decoder_output_maxlen = max(decoder_output_lens)

        for d in batch:
            d["decoder_input"].extend([word2id["<pad>"]] * (decoder_input_maxlen - d["decoder_input_len"]))
            d["decoder_output"].extend([word2id["<pad>"]] * (decoder_output_maxlen - d["decoder_output_len"]))
        decoder_inputs = torch.tensor([d["decoder_input"] for d in batch], dtype=torch.long)
        decoder_outputs = torch.tensor([d["decoder_output"] for d in batch], dtype=torch.long)
        labels = torch.tensor([d["label"] for d in batch], dtype=torch.long)  # 获取标签
        return decoder_inputs, decoder_outputs, labels

def train_step(model, data_loader, optimizer, criterion, clip=1, print_every=None):
    model.train()

    if print_every == 0:
        print_every = 1
    print_loss_total = 0  # 每次打印都重置
    epoch_loss = 0

    for i, (dec_inputs, dec_outputs, labels) in enumerate(tqdm(data_loader)):
        optimizer.zero_grad()
        dec_inputs, dec_outputs, labels = dec_inputs.to(device), dec_outputs.to(device), labels.to(device)
        outputs, dec_self_attns = model(dec_inputs, classify = True)  # 获取模型输出 logits
        # 计算交叉熵损失，输出 logits 和标签
        loss = criterion(outputs, labels)
        print_loss_total += loss.item()
        epoch_loss += loss.item()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
        optimizer.step()

        if print_every and (i + 1) % print_every == 0:
            print_loss_avg = print_loss_total / print_every
            print_loss_total = 0
            print(f'\tCurrent Loss: {print_loss_avg:.4f}')

    return epoch_loss / len(data_loader)

def train(model, data_loader):
    # 处理数据不平衡
    class_counts = df['label'].value_counts().to_dict()
    total_samples = sum(class_counts.values())
    class_weights = {label: total_samples / count for label, count in class_counts.items()}
    weights = torch.tensor([class_weights[0], class_weights[1]], dtype=torch.float).to(device)
    criterion = nn.CrossEntropyLoss(weight=weights).to(device)

    # 设置学习率
    optimizer = torch.optim.Adam([
        {'params': model.classifier.parameters(), 'lr': 1e-3},  # 仅训练分类头
        # {'params': model.decoder.layers.parameters(), 'lr': 1e-5}   # 微调解码器所有层
        {'params': model.decoder.layers[-1:].parameters(), 'lr': 1e-5}  # 微调解码器最后一层
    ])

    for epoch in range(epochs):
        start_time = time.time()
        # 训练
        train_loss = train_step(model, data_loader, optimizer, criterion, CLIP, print_every=10)
        end_time = time.time()
        # 保存训练参数
        torch.save(model.state_dict(), 'GPT2_emo_last_banmask.pt')

        epoch_mins, epoch_secs = epoch_time(start_time, end_time)
        # 打印轮数和这一轮训练的用时
        print(f'Epoch: {epoch + 1:02} | Time: {epoch_mins}m {epoch_secs}s')
        print(f'\tTrain Loss: {train_loss:.3f}')

### 参考了train的代码
if __name__ == '__main__':
    # 加载数据并处理数据
    df = pd.read_csv('cleaned_emo.csv')
    train_data = make_data(df)
    train_num_data = []
    # 遍历 train_data 中的每个文本
    for line in train_data:
        train_line_data = []
        for word in line[0]:
            # 只有当单词存在于预训练的word2id中时，才使用
            if word in word2id:
                train_line_data.append(word2id[word])
            else:
                train_line_data.append(word2id.get('<unk>', 1))  # 处理未知词
        train_line_data.append(int(line[1]))
        train_num_data.append(train_line_data)

    # 参数
    batch_size = 8
    epochs = 30
    dataset = EmoDataSet(train_num_data)
    data_loader = Data.DataLoader(dataset, batch_size=batch_size, collate_fn=dataset.padding_batch)
    print(torch.__version__)
    print(torch.cuda.is_available())

    # 使用GPU
    device = torch.device('cuda')
    model = GPT(use_mask=False).to(device)
    model.load_state_dict(torch.load('GPT2.pt', weights_only=True), strict=False) # 因为GPT类有修改，所以要strict=False

    # train start
    train(model, data_loader)
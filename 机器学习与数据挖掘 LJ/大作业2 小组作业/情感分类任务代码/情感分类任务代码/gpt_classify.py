import json
import torch
import torch.utils.data as Data
from torch import nn, optim
import numpy as np
import time
from tqdm import tqdm

# 训练准备
# 使用GPU
device = torch.device("cuda")
# device = torch.device('cpu') # for debug
# 打开一个JSON格式的文件dict_datas.json并将其中的数据加载到dict_datas变量中。
# json.load()会将文件中的JSON数据解析为Python字典对象。这个字典通常包含了一些数据，比如词汇表的映射。
dict_datas = json.load(open('dict_datas.json', 'r'))
# word2id和id2word分别从字典dict_datas中提取出来。
# word2id通常是一个词到ID的映射（比如一个词表），而id2word是一个ID到词的映射。
# 这两个映射通常在词嵌入（embedding）和序列处理模型中使用。
word2id, id2word = dict_datas['word2id'], dict_datas['id2word']
# 词汇表中不同词的数量
vocab_size = len(word2id)

# 参数
# 输入序列的最大长度
max_pos = 1800
# 模型的嵌入维度
d_model = 768  # Embedding Size
# 前馈神经网络层的维度
d_ff = 2048  # FeedForward dimension
# 注意力机制中键（Key）、查询（Query）和值（Value）的维度
d_k = d_v = 64  # dimension of K(=Q), V
# 编码器（Encoder）或解码器（Decoder）的层数
n_layers = 6  # number of Encoder of Decoder Layer
# 多头注意力机制中的头数
n_heads = 8  # number of heads in Multi-Head Attention
CLIP = 1


# 生成一个注意力掩码（Attention Mask），用来处理填充（padding）部分的注意力。
# 在序列数据处理中，填充部分（通常是0）不应该对模型的计算产生影响，因此需要通过掩码将其屏蔽掉
def get_attn_pad_mask(seq_q, seq_k):
    '''
    seq_q: [batch_size, seq_len]
    seq_k: [batch_size, seq_len]
    seq_len could be src_len or it could be tgt_len
    seq_len in seq_q and seq_len in seq_k maybe not equal
    '''
    batch_size, len_q = seq_q.size()
    batch_size, len_k = seq_k.size()
    # eq(zero) is PAD token
    pad_attn_mask = seq_k.data.eq(0).unsqueeze(1)  # [batch_size, 1, len_k], True is masked
    return pad_attn_mask.expand(batch_size, len_q, len_k)  # [batch_size, len_q, len_k]


# 没看懂为什么这样的矩阵可以屏蔽未来
def get_attn_subsequence_mask(seq):
    #seq 是一个张量，形状为 [batch_size, tgt_len]，即一个批次中每个序列的目标长度。
    # batch_size 是批次大小，tgt_len 是目标序列的长度（可能是解码过程中的时间步长）。
    '''
    seq: [batch_size, tgt_len]
    '''
    # attn_shape 是生成注意力掩码时需要的张量形状，形状为 [batch_size, tgt_len, tgt_len]。
    # seq.size(0) 是批次大小（batch_size）。
    # seq.size(1) 是目标序列的长度（tgt_len）。
    attn_shape = [seq.size(0), seq.size(1), seq.size(1)]
    # 成一个上三角矩阵，并将对角线及以下部分的值设为 0。参数 k=1 表示将对角线以上（包括对角线本身）的部分设为 1，其余部分设为 0。
    subsequence_mask = np.triu(np.ones(attn_shape), k=1)  # Upper triangular matrix
    #  numpy 数组转换为 PyTorch 张量，并使用 .byte() 方法将其转换为 uint8 类型（即 torch.uint8），这是因为掩码通常以 0 和 1 表示。
    subsequence_mask = torch.from_numpy(subsequence_mask).byte()
    # 移动到指定的设备上
    subsequence_mask = subsequence_mask.to(device)
    # 返回形状为 [batch_size, tgt_len, tgt_len] 的掩码张量，它表示每个时间步对其他时间步的可见性
    return subsequence_mask  # [batch_size, tgt_len, tgt_len]


# 这段代码实现了 缩放点积注意力 的计算过程
# 计算输入的查询（Q）、键（K）和值（V）之间的加权关系。
# 通过计算查询与键的点积来评估它们的相关性，再通过 softmax 计算注意力分布，最后得到基于该分布加权的值向量。
class ScaledDotProductAttention(nn.Module):
    def __init__(self, d_k):
        super(ScaledDotProductAttention, self).__init__()
        self.d_k = d_k

    def forward(self, q, k, v, attention_mask):
        ##
        # q: [batch_size, n_heads, len_q, d_k]
        # k: [batch_size, n_heads, len_k, d_k]
        # v: [batch_size, n_heads, len_v, d_v]
        # attn_mask: [batch_size, n_heads, seq_len, seq_len]
        ##
        # 计算每个Q与K的分数，计算出来的大小是 [batch_size, n_heads, len_q, len_q]
        scores = torch.matmul(q, k.transpose(-1, -2)) / np.sqrt(self.d_k)
        # 把被mask的地方置为无限小，softmax之后基本就是0，也就对q不起作用
        scores.masked_fill_(attention_mask, -1e9)
        attn = nn.Softmax(dim=-1)(scores)
        # 注意力后的大小 [batch_size, n_heads, len_q, d_v]
        context = torch.matmul(attn, v)
        return context, attn


class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, n_heads, d_k, d_v):
        super(MultiHeadAttention, self).__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_k = d_k
        self.d_v = d_v
        self.w_q = nn.Linear(d_model, d_k * n_heads, bias=False)
        self.w_k = nn.Linear(d_model, d_k * n_heads, bias=False)
        self.w_v = nn.Linear(d_model, d_v * n_heads, bias=False)
        self.fc = nn.Linear(n_heads * d_v, d_model, bias=False)
        self.layernorm = nn.LayerNorm(d_model)

    def forward(self, q, k, v, attention_mask):
        ##
        # q: [batch_size, seq_len, d_model]
        # k: [batch_size, seq_len, d_model]
        # v: [batch_size, seq_len, d_model]
        # attn_mask: [batch_size, seq_len, seq_len]
        ##
        # 记录原始值, 后续计算残差
        residual, batch_size = q, q.size(0)
        # 先映射 q、k、v, 然后后分头
        # q: [batch_size, n_heads, len_q, d_k]
        q = self.w_q(q).view(batch_size, -1, self.n_heads, self.d_k).transpose(1, 2)
        # k: [batch_size, n_heads, len_k, d_k]
        k = self.w_k(k).view(batch_size, -1, self.n_heads, self.d_k).transpose(1, 2)
        # v: [batch_size, n_heads, len_v(=len_k), d_v]
        v = self.w_v(v).view(batch_size, -1, self.n_heads, self.d_v).transpose(1, 2)
        # attn_mask : [batch_size, n_heads, seq_len, seq_len]
        attention_mask = attention_mask.unsqueeze(1).repeat(1, self.n_heads, 1, 1)
        # 点积注意力分数计算，  [batch_size, n_heads, len_q, d_v]
        context, attn = ScaledDotProductAttention(self.d_k)(q, k, v, attention_mask)
        # context: [batch_size, len_q, n_heads * d_v]
        context = context.transpose(1, 2).reshape(batch_size, -1, self.n_heads * self.d_v)
        # 还原为原始大小
        output = self.fc(context)
        # LN + 残差计算
        return self.layernorm(output + residual), attn


# 前馈神经网络层，
# 由两个线性全连接层组成，中间使用 ReLU 激活函数衔接，主要在做一个升维再降维的操作，
# 可以学习到更为抽象的特征。
class PoswiseFeedForwardNet(nn.Module):
    def __init__(self, d_model, d_ff):
        super(PoswiseFeedForwardNet, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(d_model, d_ff, bias=False),
            nn.ReLU(),
            nn.Linear(d_ff, d_model, bias=False)
        )
        self.layernorm = nn.LayerNorm(d_model)

    def forward(self, inputs):
        ##
        # inputs: [batch_size, seq_len, d_model]
        ##
        residual = inputs
        output = self.fc(inputs)
        # # LN + 残差计算, [batch_size, seq_len, d_model]
        return self.layernorm(output + residual)


# 一个解码层由一个多头注意力层和一个前馈神经网络层组成
class DecoderLayer(nn.Module):
    def __init__(self, d_model, n_heads, d_ff, d_k, d_v):
        super(DecoderLayer, self).__init__()
        # 多头注意力层
        self.attention = MultiHeadAttention(d_model, n_heads, d_k, d_v)
        # 前馈神经网络层
        self.pos_ffn = PoswiseFeedForwardNet(d_model, d_ff)

    def forward(self, inputs, attention_mask):
        ##
        # inputs: [batch_size, seq_len, d_model]
        # attention_mask: [batch_size, seq_len, seq_len]
        ##
        # outputs: [batch_size, seq_len, d_model]
        # self_attn: [batch_size, n_heads, seq_len, seq_len]
        outputs, self_attn = self.attention(inputs, inputs, inputs, attention_mask)
        # [batch_size, seq_len, d_model]
        outputs = self.pos_ffn(outputs)
        return outputs, self_attn


# 解码器主要将多个解码层堆叠，形成一个特征提取链路。
# 首先解码器接收输入的 Token，然后通过 Embedding 转为高维向量表示
# 由于注意力机制没有位置信息，因此这里还需要加上位置编码。
class Decoder(nn.Module):
    def __init__(self, d_model, n_heads, d_ff, d_k, d_v, vocab_size, max_pos, n_layers, use_mask=True):
        super(Decoder, self).__init__()
        self.tgt_emb = nn.Embedding(vocab_size, d_model)
        self.pos_emb = nn.Embedding(max_pos, d_model)
        self.layers = nn.ModuleList([DecoderLayer(d_model, n_heads, d_ff, d_k, d_v) for _ in range(n_layers)])
        self.use_mask = use_mask  # 控制是否使用mask

    def forward(self, dec_inputs):
        '''
        dec_inputs: [batch_size, tgt_len]
        '''
        seq_len = dec_inputs.size(1)
        pos = torch.arange(seq_len, dtype=torch.long, device=device)
        pos = pos.unsqueeze(0).expand_as(dec_inputs)  # [seq_len] -> [batch_size, seq_len]

        dec_outputs = self.tgt_emb(dec_inputs) + self.pos_emb(pos)  # [batch_size, tgt_len, d_model]

        if self.use_mask:
            dec_self_attn_pad_mask = get_attn_pad_mask(dec_inputs, dec_inputs)  # [batch_size, tgt_len, tgt_len]
            dec_self_attn_subsequence_mask = get_attn_subsequence_mask(dec_inputs)  # [batch_size, tgt_len, tgt_len]
            dec_self_attn_mask = torch.gt((dec_self_attn_pad_mask + dec_self_attn_subsequence_mask),
                                      0)  # [batch_size, tgt_len, tgt_len]
        else:
            # 如果禁用mask，创建一个全为False的mask
            dec_self_attn_mask = torch.zeros((dec_inputs.size(0), seq_len, seq_len), dtype=torch.bool, device=device)
        dec_self_attns = []
        for layer in self.layers:
            dec_outputs, dec_self_attn = layer(dec_outputs, dec_self_attn_mask)
            dec_self_attns.append(dec_self_attn)
        return dec_outputs, dec_self_attns


# 解码器（Decoder）部分，负责生成目标序列。
# 一个线性投影层（projection），将解码器输出映射到词汇表大小。
# 贪心解码算法（greedy_decoder），用于根据输入生成文本，直到预测到<sep>为止。
# answer方法，将输入句子转化为回答，回答是由模型通过greedy_decoder生成的
class GPT(nn.Module):
    def __init__(self, use_mask=True):
        super(GPT, self).__init__()
        self.decoder = Decoder(d_model, n_heads, d_ff, d_k, d_v, vocab_size, max_pos, n_layers, use_mask)
        self.projection = nn.Linear(d_model, vocab_size)
        self.classifier = nn.Linear(d_model, 2) # classify task

    def forward(self, dec_inputs, classify=False):
        # dec_inputs: [batch_size, tgt_len]
        # classify: 是否进行分类任务 (False 表示生成任务)
        dec_outputs, dec_self_attns = self.decoder(dec_inputs)  # [batch_size, tgt_len, d_model]

        if classify:
            # 分类任务: 取最后一个时间步的输出进行分类
            last_token_output = dec_outputs[:, -1, :]  # [batch_size, d_model]
            class_logits = self.classifier(last_token_output)  # [batch_size, 2]
            return class_logits, dec_self_attns
        else:
            # 生成任务: 使用投影层生成词汇分布
            dec_logits = self.projection(dec_outputs)  # [batch_size, tgt_len, vocab_size]
            return dec_logits.view(-1, dec_logits.size(-1)), dec_self_attns

    def greedy_decoder(self, dec_input):

        terminal = False
        start_dec_len = len(dec_input[0])
        # 一直预测下一个单词，直到预测到"<sep>"结束，如果一直不到"<sep>"，则根据长度退出循环，并在最后加上”<sep>“字符
        while not terminal:
            if len(dec_input[0]) - start_dec_len > 100:
                next_symbol = word2id['<sep>']
                dec_input = torch.cat(
                    [dec_input.detach(), torch.tensor([[next_symbol]], dtype=dec_input.dtype, device=device)], -1)
                break
            dec_outputs, _ = self.decoder(dec_input)
            projected = self.projection(dec_outputs)
            prob = projected.squeeze(0).max(dim=-1, keepdim=False)[1]
            next_word = prob.data[-1]
            next_symbol = next_word
            if next_symbol == word2id["<sep>"]:
                terminal = True

            dec_input = torch.cat(
                [dec_input.detach(), torch.tensor([[next_symbol]], dtype=dec_input.dtype, device=device)], -1)

        return dec_input

    def answer(self, sentence):
        # 把原始句子的\t替换成”<sep>“
        dec_input = [word2id.get(word, 1) if word != '\t' else word2id['<sep>'] for word in sentence]
        dec_input = torch.tensor(dec_input, dtype=torch.long, device=device).unsqueeze(0)

        output = self.greedy_decoder(dec_input).squeeze(0)
        out = [id2word[int(id)] for id in output]
        # 统计"<sep>"字符在结果中的索引
        sep_indexs = []
        for i in range(len(out)):
            if out[i] == "<sep>":
                sep_indexs.append(i)

        # 取最后两个sep中间的内容作为回答

        answer = out[sep_indexs[-2] + 1:-1]

        answer = "".join(answer)
        return answer

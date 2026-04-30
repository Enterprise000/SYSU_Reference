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
d_model = 768
# 前馈神经网络层的维度
d_ff = 2048
# 注意力机制中键（Key）、查询（Query）和值（Value）的维度
d_k = d_v = 64
# 编码器（Encoder）或解码器（Decoder）的层数
n_layers = 6
# 多头注意力机制中的头数
n_heads = 8
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


# 缩放点积注意力的计算过程
class ScaledDotProductAttention(nn.Module):
    def __init__(self, d_k):
        super(ScaledDotProductAttention, self).__init__()
        self.d_k = d_k

    def forward(self, q, k, v, attention_mask):
        # 计算输入的查询（Q）、键（K）和值（V）之间的加权关系。
        # 通过计算查询与键的点积来评估它们的相关性，再通过 softmax 计算注意力分布，最后得到基于该分布加权的值向量。
        scores = torch.matmul(q, k.transpose(-1, -2)) / np.sqrt(self.d_k)
        scores.masked_fill_(attention_mask, -1e9)
        attn = nn.Softmax(dim=-1)(scores)
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
        residual, batch_size = q, q.size(0)
        # 将输入的查询、键和值分别映射到多个注意力头
        q = self.w_q(q).view(batch_size, -1, self.n_heads, self.d_k).transpose(1, 2)
        k = self.w_k(k).view(batch_size, -1, self.n_heads, self.d_k).transpose(1, 2)
        v = self.w_v(v).view(batch_size, -1, self.n_heads, self.d_v).transpose(1, 2)
        attention_mask = attention_mask.unsqueeze(1).repeat(1, self.n_heads, 1, 1)
        # 并行计算每个头的注意力，最终通过全连接层将多头的结果合并起来。
        context, attn = ScaledDotProductAttention(self.d_k)(q, k, v, attention_mask)
        context = context.transpose(1, 2).reshape(batch_size, -1, self.n_heads * self.d_v)
        output = self.fc(context)
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
        # 首先将输入经过fc网络，随后应用残差连接和层归一化，得到输出。
        residual = inputs
        output = self.fc(inputs)
        return self.layernorm(output + residual)


# 一个解码层由一个多头注意力层和一个前馈神经网络层组成
class DecoderLayer(nn.Module):
    def __init__(self, d_model, n_heads, d_ff, d_k, d_v):
        super(DecoderLayer, self).__init__()
        self.attention = MultiHeadAttention(d_model, n_heads, d_k, d_v)
        self.pos_ffn = PoswiseFeedForwardNet(d_model, d_ff)

    def forward(self, inputs, attention_mask):
        # 将输入通过多头自注意力层和前馈神经网络层，得到输出。
        outputs, self_attn = self.attention(inputs, inputs, inputs, attention_mask)
        outputs = self.pos_ffn(outputs)
        return outputs, self_attn


# 解码器主要将多个解码层堆叠，形成一个特征提取链路。
# 首先解码器接收输入的 Token，然后通过 Embedding 转为高维向量表示
# 由于注意力机制没有位置信息，因此这里还需要加上位置编码。
class Decoder(nn.Module):
    def __init__(self, d_model, n_heads, d_ff, d_k, d_v, vocab_size, max_pos, n_layers):
        super(Decoder, self).__init__()
        self.tgt_emb = nn.Embedding(vocab_size, d_model)
        self.pos_emb = nn.Embedding(max_pos, d_model)
        self.layers = nn.ModuleList([DecoderLayer(d_model, n_heads, d_ff, d_k, d_v) for _ in range(n_layers)])

    def forward(self, dec_inputs):
        # 首先生成位置编码，得到加了位置信息的嵌入结果 dec_outputs。
        # 随后，生成填充掩码和注意力掩码，并将它们组合起来。
        seq_len = dec_inputs.size(1)
        pos = torch.arange(seq_len, dtype=torch.long, device=device)
        pos = pos.unsqueeze(0).expand_as(dec_inputs)

        dec_outputs = self.tgt_emb(dec_inputs) + self.pos_emb(pos)

        dec_self_attn_pad_mask = get_attn_pad_mask(dec_inputs, dec_inputs)
        dec_self_attn_subsequence_mask = get_attn_subsequence_mask(dec_inputs)
        dec_self_attn_mask = torch.gt((dec_self_attn_pad_mask + dec_self_attn_subsequence_mask),0)
        dec_self_attns = []
        # 解码层共有n_layers个。
        # 对于每个解码层，将输入放入解码层进行处理。
        # 解码层中的输入是加了位置编码的输入，
        # 解码层中的掩码是在之前组合起来的填充掩码和注意力掩码。
        # 每一层的输出会变成下一层的输入。
        for layer in self.layers:
            dec_outputs, dec_self_attn = layer(dec_outputs, dec_self_attn_mask)
            dec_self_attns.append(dec_self_attn)
        return dec_outputs, dec_self_attns


# 解码器（Decoder）部分，负责生成目标序列。
# 一个线性投影层（projection），将解码器输出映射到词汇表大小。
# 贪心解码算法（greedy_decoder），用于根据输入生成文本，直到预测到<sep>为止。
# answer方法，将输入句子转化为回答，回答是由模型通过greedy_decoder生成的
class GPT(nn.Module):
    def __init__(self):
        super(GPT, self).__init__()
        # 一个decoder模块和一个将解码器输出映射到词表的线性层
        self.decoder = Decoder(d_model, n_heads, d_ff, d_k, d_v, vocab_size, max_pos, n_layers)
        self.projection = nn.Linear(d_model, vocab_size)

    def forward(self, dec_inputs):
        # 输入通过Decoder模块得到解码器的输出和自注意力权重，
        # 再通过线性层得到每个位置上词汇表的概率分布。
        dec_outputs, dec_self_attns = self.decoder(dec_inputs)
        dec_logits = self.projection(dec_outputs)
        return dec_logits.view(-1, dec_logits.size(-1)), dec_self_attns

    def greedy_decoder(self, dec_input):
        # 不断循环直到terminal变为true。
        # 将输入放入解码器获得输出，
        # 再通过线性投影层获得词表的概率分布，
        # 随后获取概率最大的词，
        # 并将这个词作为输入拼接到输入序列中。
        # 最终返回生成的序列。
        terminal = False
        start_dec_len = len(dec_input[0])
        while not terminal:
            # 如果遇到`<sep>`或生成的长度超过100，就终止。
            if len(dec_input[0]) - start_dec_len > 100:
                next_symbol = word2id['<sep>']
                dec_input = torch.cat([dec_input.detach(), torch.tensor([[next_symbol]], dtype=dec_input.dtype, device=device)], -1)
                break
            dec_outputs, _ = self.decoder(dec_input)
            projected = self.projection(dec_outputs)
            prob = projected.squeeze(0).max(dim=-1, keepdim=False)[1]
            next_word = prob.data[-1]
            next_symbol = next_word
            if next_symbol == word2id["<sep>"]:
                terminal = True
            dec_input = torch.cat([dec_input.detach(), torch.tensor([[next_symbol]], dtype=dec_input.dtype, device=device)], -1)
        return dec_input

    def answer(self, sentence):
        # 首先将输入的句子转换成词表的形式（word2id），并转换为张量。
        # 调用greedy_decoder生成输出序列，再通过id2word转换成单词。
        # 提取`<sep>`符之间的部分作为回答，将回答拼接成字符串输出。
        dec_input = [word2id.get(word, 1) if word != '\t' else word2id['<sep>'] for word in sentence]
        dec_input = torch.tensor(dec_input, dtype=torch.long, device=device).unsqueeze(0)

        output = self.greedy_decoder(dec_input).squeeze(0)
        out = [id2word[int(id)] for id in output]
        sep_indexs = []
        for i in range(len(out)):
            if out[i] == "<sep>":
                sep_indexs.append(i)
        answer = out[sep_indexs[-2] + 1:-1]
        answer = "".join(answer)
        return answer

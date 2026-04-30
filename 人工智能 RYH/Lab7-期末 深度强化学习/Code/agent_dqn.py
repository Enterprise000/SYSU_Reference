import random
import copy
import numpy as np
import torch
from torch import nn, optim
from agent import Agent


class QNetwork(nn.Module):
    # 搭建一个神经网络
    def __init__(self, input_size, hidden_size, output_size):
        super(QNetwork, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, output_size)

    def forward(self, inputs):
        out = torch.relu(self.fc1(inputs))
        out = self.fc2(out)
        return out


class ReplayBuffer:
    def __init__(self, buffer_size):
        # 缓冲区大小
        self.buffer_size = buffer_size
        # 初始缓冲区为空
        self.buffer = []
        # 访问位置
        self.pos = 0
        # 缓冲区中各元素的优先级，初始化为0
        self.priorities = np.zeros((buffer_size,), dtype=np.float32)
        # 最大优先级
        self.max_priority = 1.0
        return

    def __len__(self):
        # 返回缓冲区长度
        return len(self.buffer)

    # 填入缓冲区
    def push(self, *transition):
        # 填入“数据”
        if len(self.buffer) < self.buffer_size:
            self.buffer.append(None)
        # 在当前位置放入转换后的数据
        self.buffer[self.pos] = transition
        # 将新数据的优先级设置为最高
        self.priorities[self.pos] = self.max_priority
        # 用循环方式更改pos
        self.pos = (self.pos + 1) % self.buffer_size
        return

    # 抽样
    def sample(self, batch_size):
        # 删去未使用的priority
        priorities = self.priorities[:len(self.buffer)]
        # 计算抽样概率
        probs = priorities ** 0.5
        probs /= probs.sum()
        # 抽取一个批次
        indices = np.random.choice(len(self.buffer), batch_size, p=probs)
        batch = [self.buffer[idx] for idx in indices]
        return batch, indices

    # 更新优先级
    def update_priorities(self, indices, priorities):
        # 更新指定位置的priority
        self.priorities[indices] = priorities
        # 更新max
        self.max_priority = max(self.max_priority, np.max(priorities))
        return

    # 清空缓冲区
    def clean(self):
        # 清空缓冲区
        self.buffer = []
        # 复位
        self.pos = 0
        return


class AgentDQN(Agent):
    def __init__(self, env, args):
        super(AgentDQN, self).__init__(env)
        self.env = env
        self.args = args
        # 选择GPU或者CPU
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        # 神经网络相关
        # 创建神经网络
        self.q_network = QNetwork(4, 256, 2).to(self.device)
        # 创建优化器，学习率0.001
        self.optimizer = optim.Adam(self.q_network.parameters(), lr=0.001)
        # 创建目标网络
        self.target_network = copy.deepcopy(self.q_network)
        # 缓冲区相关
        # 创建一个容量4000的缓冲区
        self.replay_buffer = ReplayBuffer(5000)
        # 缓冲区最小大小为400
        self.minimal_size = 500
        # Qlearning相关
        # 批次大小
        self.batch_size = 50
        # 折扣因子
        self.gamma = 0.9
        # 探索率
        self.epsilon = 0.9
        # 探索率的衰减率
        self.epsilon_decay = 0.95
        # 最小探索率
        self.epsilon_min = 0.001
        # 目标网络更新频率
        self.update_target_every = 10
        # 步数
        self.steps = 0
        return

    # def init_game_setting(self):

    def train(self):
        # 如果缓冲区长度小于最小长度，不训练
        if len(self.replay_buffer) < self.minimal_size:
            return
        # 从缓冲区中抽样，获得抽样结果和这些结果的index
        transitions, indices = self.replay_buffer.sample(self.batch_size)
        # 将抽样结果分类组成元组，再放进一个列表，方便处理
        batch = list(zip(*transitions))
        # 第一类，状态
        states = torch.tensor(np.array(batch[0]), dtype=torch.float32).to(self.device)
        # 第二类，行动
        actions = torch.tensor(batch[1], dtype=torch.int64).to(self.device)
        # 第三类，奖励
        rewards = torch.tensor(batch[2], dtype=torch.float32).to(self.device)
        # 第四类，下一个状态
        next_states = torch.tensor(np.array(batch[3]), dtype=torch.float32).to(self.device)
        # 第五类，完成标志
        dones = torch.tensor(batch[4], dtype=torch.float32).to(self.device)
        # 计算当前state下选择的action对应的q值
        q_values = self.q_network(states).gather(1, actions.unsqueeze(1)).squeeze(1)
        # next_state下目标网络计算的最大q值
        next_q_values = self.target_network(next_states).max(1)[0]
        # 计算预期的q值，如果dones为真，那么q值是reward，否则要折现
        expected_q_values = rewards + (self.gamma * next_q_values * (1 - dones))
        # 均方误差计算loss
        loss = nn.MSELoss()(q_values, expected_q_values.detach())
        # 更新参数
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        # 更新目标网络
        if self.steps % self.update_target_every == 0:
            self.target_network.load_state_dict(self.q_network.state_dict())
        # 更新缓冲区的优先级
        self.replay_buffer.update_priorities(indices, (q_values - expected_q_values).abs().cpu().detach().squeeze(-1).numpy())
        return

    def make_action(self, observation, test=True):
        # 随机数>探索率
        if random.random() > self.epsilon:
            # 预测动作
            observation = torch.tensor(observation, dtype=torch.float32).unsqueeze(0).to(self.device)
            with torch.no_grad():
                # 最大q值动作
                action = self.q_network(observation).argmax().item()
        # 随机选择一个动作
        else:
            action = self.env.action_space.sample()
        return action

    def run(self):
        # 每个episode的总reward
        total_rewards = []
        for episode in range(100):
            # 重置state为初始状态
            state = self.env.reset()
            # 取出实际的状态值
            state = state[0]
            # 初始化奖励为0
            total_reward = 0
            # 初始化完成标志为false
            done = False
            # 未完成
            while not done:
                # 训练模式，根据当前状态选择动作
                action = self.make_action(state, test=False)
                # 执行动作，获得下一个状态，奖励，完成标志
                next_state, reward, terminated, truncated, _ = self.env.step(action)
                done = terminated or truncated
                # 把信息放进缓冲区
                self.replay_buffer.push(state, action, reward, next_state, done)
                # state转移
                state = next_state
                # 计算reward
                total_reward += reward
                # 训练神经网络
                self.train()
                self.steps += 1
            # 更新探索率
            self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)
            # 添加reward
            total_rewards.append(total_reward)
            # 打印信息
            print("Episode: ", episode + 1, "Total Reward: ", total_reward, "Epsilon: ", self.epsilon)
        return total_rewards

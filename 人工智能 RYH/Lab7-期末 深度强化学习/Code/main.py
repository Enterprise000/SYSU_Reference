import argparse
import matplotlib.pyplot as plt
import gym
from argument import dqn_arguments


def parse():
    parser = argparse.ArgumentParser(description="SYSU_RL_HW2")
    parser.add_argument('--train_dqn', default=True, type=bool, help='whether train DQN')

    parser = dqn_arguments(parser)
    # parser = pg_arguments(parser)
    args = parser.parse_args()
    return args


def run(args):
    if args.train_dqn:
        # 环境是CartPole-v0
        env_name = args.env_name
        # 创建env对象
        env = gym.make(env_name)
        # 创建agent实例
        from agent_dqn import AgentDQN
        agent = AgentDQN(env, args)
        # 运行
        total_rewards = agent.run()
        # 画图
        draw(total_rewards)


def draw(total_rewards):
    plt.plot(total_rewards, label='Total Reward')
    plt.xlabel('Episode')
    plt.ylabel('Total Reward')
    plt.title(' Total Reward and Episode')
    plt.legend()
    plt.show()
    return


if __name__ == '__main__':
    args = parse()
    run(args)

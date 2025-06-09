"""
The DQN improvement: Prioritized Experience Replay
Using TensorFlow 2.x
"""

import gym
from RL_brain import DQNPrioritizedReplay
import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np

# 确保使用 TensorFlow 1.x 的兼容模式
tf.compat.v1.disable_eager_execution()

env = gym.make('MountainCar-v0')
env = env.unwrapped
observation, info = env.reset(seed=21)  # 新版本 gym 的 reset 方法
MEMORY_SIZE = 10000

sess = tf.compat.v1.Session()
with tf.compat.v1.variable_scope('natural_DQN'):
    RL_natural = DQNPrioritizedReplay(
        n_actions=3, n_features=2, memory_size=MEMORY_SIZE,
        e_greedy_increment=0.00005, sess=sess, prioritized=False,
    )

with tf.compat.v1.variable_scope('DQN_with_prioritized_replay'):
    RL_prio = DQNPrioritizedReplay(
        n_actions=3, n_features=2, memory_size=MEMORY_SIZE,
        e_greedy_increment=0.00005, sess=sess, prioritized=True, output_graph=True,
    )
sess.run(tf.compat.v1.global_variables_initializer())


def train(RL):
    total_steps = 0
    steps = []
    episodes = []
    for i_episode in range(20):
        # 修改 reset 的调用方式
        observation, _ = env.reset()
        while True:
            # env.render()  # 如果需要渲染环境，可以取消注释

            action = RL.choose_action(observation)

            # 修改 step 的返回值解包
            observation_, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated  # 合并终止条件

            if done: 
                reward = 10

            RL.store_transition(observation, action, reward, observation_)

            if total_steps > MEMORY_SIZE:
                RL.learn()

            if done:
                print('episode ', i_episode, ' finished')
                steps.append(total_steps)
                episodes.append(i_episode)
                break

            observation = observation_
            total_steps += 1
    return np.vstack((episodes, steps))

his_natural = train(RL_natural)
his_prio = train(RL_prio)

# compare based on first success
plt.plot(his_natural[0, :], his_natural[1, :] - his_natural[1, 0], c='b', label='natural DQN')
plt.plot(his_prio[0, :], his_prio[1, :] - his_prio[1, 0], c='r', label='DQN with prioritized replay')
plt.legend(loc='best')
plt.ylabel('total training time')
plt.xlabel('episode')
plt.grid()
plt.show()



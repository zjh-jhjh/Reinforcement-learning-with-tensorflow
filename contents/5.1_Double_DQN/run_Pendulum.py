"""
Double DQN & Natural DQN comparison
"""
import gym
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from RL_brain import DoubleDQN

# 设置随机种子
np.random.seed(1)
tf.random.set_seed(1)

# 创建并初始化环境
env = gym.make('Pendulum-v1')
env = env.unwrapped
# 新版本 gym 使用 reset 时设置随机种子
observation, info = env.reset(seed=1)

MEMORY_SIZE = 3000
ACTION_SPACE = 11

# 使用 TF 1.x 兼容模式创建会话
sess = tf.compat.v1.Session()

with tf.compat.v1.variable_scope('Natural_DQN'):
    natural_DQN = DoubleDQN(
        n_actions=ACTION_SPACE, n_features=3, memory_size=MEMORY_SIZE,
        e_greedy_increment=0.001, double_q=False, sess=sess
    )

with tf.compat.v1.variable_scope('Double_DQN'):
    double_DQN = DoubleDQN(
        n_actions=ACTION_SPACE, n_features=3, memory_size=MEMORY_SIZE,
        e_greedy_increment=0.001, double_q=True, sess=sess, output_graph=True)

sess.run(tf.compat.v1.global_variables_initializer())

def train(RL):
    total_steps = 0
    observation, _ = env.reset()  # 新版本 gym reset 返回值
    while True:
        action = RL.choose_action(observation)

        f_action = (action-(ACTION_SPACE-1)/2)/((ACTION_SPACE-1)/4)   # convert to [-2 ~ 2] float actions
        # 更新 step 方法的返回值解包
        observation_, reward, terminated, truncated, info = env.step(np.array([f_action]))
        done = terminated or truncated  # 合并终止条件

        reward /= 10     # normalize to a range of (-1, 0)
        
        RL.store_transition(observation, action, reward, observation_)

        if total_steps > MEMORY_SIZE:   # learning
            RL.learn()

        if total_steps - MEMORY_SIZE > 20000:   # stop game
            break

        observation = observation_
        total_steps += 1
    return RL.q

q_natural = train(natural_DQN)
q_double = train(double_DQN)

plt.plot(np.array(q_natural), c='r', label='natural')
plt.plot(np.array(q_double), c='b', label='double')
plt.legend(loc='best')
plt.ylabel('Q eval')
plt.xlabel('training steps')
plt.grid()
plt.show()

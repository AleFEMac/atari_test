import gym
#from Qfunction import Qfunction
import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np
import random

env = gym.make('Asterix-v0')

#######################
#CNN
#######################
img = tf.placeholder(tf.float32, shape=(None, 210, 160, 3), name='img')
Y = tf.placeholder(tf.int32, (None,), name='actions')
DR = tf.placeholder(tf.float32, (None,), name='discounted_reward')

alpha = 0.5
value_scale = 0.5
entropy_scale = 0.00

conv1 = tf.layers.conv2d(inputs=img, filters=32, kernel_size=[5, 5], padding="same", activation=tf.nn.relu, name='conv1')
pool1 = tf.layers.max_pooling2d(inputs=conv1, pool_size=[2, 2], strides=2, name='pool1')
conv2 = tf.layers.conv2d(inputs=pool1, filters=64, kernel_size=[5, 5], padding="same", activation=tf.nn.relu, name='conv2')
pool2 = tf.layers.max_pooling2d(inputs=conv2, pool_size=[2, 2], strides=2, name='pool2')

flat = tf.layers.flatten(pool2)
dense = tf.layers.dense(inputs = flat, units = 256, activation = tf.nn.relu, name = 'fc')
logits = tf.layers.dense(inputs = dense, units = 9, name='logits')
value = tf.layers.dense(inputs=dense, units = 1, name='value')

calc_action = tf.multinomial(logits, 1)
aprob = tf.nn.softmax(logits)
action_logprob = tf.nn.log_softmax(logits)

# Define Losses
pg_loss         = tf.reduce_mean((DR + value) * tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits, labels=Y))
value_loss      = value_scale * tf.reduce_mean(tf.square(DR - value))
entropy_loss    = -entropy_scale * tf.reduce_sum(aprob * tf.exp(aprob))
loss            = pg_loss + value_loss - entropy_loss

optim = tf.train.AdamOptimizer(alpha)
nextstep = optim.minimize(loss)

sess = tf.Session()
init = tf.global_variables_initializer()
sess.run(init)

def main():

    gamma = 0.9
    eps = 0.1
    nep = 100
    ep_size = 500
    results = []

    # Initialize Session
    sess = tf.Session()
    init = tf.global_variables_initializer()
    sess.run(init)

    repeat = 100
    r = []
    for rep in range(0, repeat):
        print("Iteration", rep+1)
        S, A, R, steps = getep(eps, gamma, ep_size)
        r.append(sum(R)/len(R))
        sess.run(nextstep, feed_dict={img:S, Y:A, DR: R})

    # Plot of the rewards
    x = []

    for  i in range(0, nep):
        x.append(i)

    plt.plot(x, r, 'b-')
    plt.show()
    env.close()


def getep(eps, gamma, size):
    S, A, R = [], [], []

    s = env.reset()
    step = 0
    rews = []
    while step < size:
        env.render()
        e = random.random()
        if e < eps:
            action = env.action_space.sample()
        else:
            inp = {img: [s]}
            action = sess.run(calc_action, feed_dict=inp)
            action = action[0][0]
        s1, r, done, lives = env.step(action)
        S.append(s1)
        A.append(action)
        rews.append(r)
        step += 1
        if done:
            break
        s = s1
    R = disc_rews(rews, gamma)

    return S, A, R, step

def disc_rews(r, gamma):
    dr = np.zeros_like(r)
    G = 0.0
    for i in reversed(range(0, len(r))):
        G = G * gamma + r[i]
        dr[i] = G

    # Rewards normalization
    mean = np.mean(dr)
    std = np.std(dr)
    dr = (dr - mean) / (std)

    return dr


if __name__ == "__main__":
    main()

import gym
env = gym.make('Blackjack-v0')

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def color_plot(arr, x_bot=None, x_top=None, y_bot=None, y_top=None):
  fig = plt.figure()
  ax = fig.add_subplot(111)
  plt.imshow(arr)
  ax.set_aspect('equal')
  plt.colorbar(orientation='vertical')
  if x_bot and x_top:
    plt.xlim(x_bot, x_top)
  if y_bot and y_top:
    plt.ylim(y_bot, y_top)

## First Visit MC

# Only stick if our hand is 20 or 21
policy = np.ones((32, 11, 2))
policy[20:22, :, :] = 0
policy[9:11, :, 1] = 0

V_s = np.zeros((32, 11, 2))
ret = np.zeros((32, 11, 2))
count = np.zeros((32, 11, 2))

DISCOUNT = 1

for _ in range(500000):
  hand, show, ace = env.reset()
  done = False
  episode = []
  
  while not done:
    state = (hand, show, int(ace))
    (hand, show, ace), reward, done, _ = env.step(int(policy[state]))
    episode.append((state, reward))
  
  g = 0
  while len(episode) > 0:
    state, reward = episode.pop()
    g = DISCOUNT * g + reward
    if (state, reward) not in episode:
      count[state] += 1
      V_s[state] += (g - V_s[state])/count[state]

color_plot(V_s[:,:,0], 0.5, 10.5, 11.5, 21.5)
plt.savefig('first_no_ace.png')
color_plot(V_s[:,:,1], 0.5, 10.5, 11.5, 21.5)
plt.savefig('first_ace.png')

## Monte Carlo Exploring Starts

usable = np.zeros((32, 11, 2, 2))
usable[1:22, 1:12] = 1

q = np.random.random((32, 11, 2, 2)) * usable
policy = np.argmax(q, axis=3)
ret = np.zeros((32, 11, 2, 2))
count = np.zeros((32, 11, 2, 2))

DISCOUNT = 1

for _ in range(10000000):
  # Environment already has positive chance for all states
  hand, show, ace = env.reset()
  state = (hand, show, int(ace))
  done = False
  episode = []

  action = np.random.randint(0, 2)
  (hand, show, ace), reward, done, _ = env.step(action)
  episode.append((state, action, reward))
  while not done:
    state = (hand, show, int(ace))
    action = int(policy[state])
    (hand, show, ace), reward, done, _ = env.step(action)
    episode.append((state, action, reward))
  
  g = 0
  while len(episode) > 0:
    state, action, reward = episode.pop()
    g = DISCOUNT * g + reward
    
    if (state, action, reward) not in episode:
      count[state + tuple([action])] += 1
      q[state + tuple([action])] += (g - q[state + tuple([action])])/count[state + tuple([action])]
      policy[state] = np.argmax(q[state])

for i in range(2):
  for j in range(2):
    color_plot(q[:,:,i,j], 0.5, 10.5, 11.5, 21.5)
    str_i = 'ace' if i else 'no_ace'
    str_j = 'hit' if j else 'stick'
    plt.savefig('es_' + str_i + '_' + str_j + '.png')

## On-Policy First-Visit Monte Carlo

usable = np.zeros((32, 11, 2, 2))
usable[1:22, 1:12] = 1
q = np.random.random((32, 11, 2, 2)) * usable
policy = np.argmax(q, axis=3)
ret = np.zeros((32, 11, 2, 2))
count = np.zeros((32, 11, 2, 2))

epsilon = 0.1
DISCOUNT = 1

for _ in range(1000000):
  hand, show, ace = env.reset()  
  done = False
  g = 0
  episode = []

  while not done:
    state = (hand, show, int(ace))
    action = int(policy[state])
    (hand, show, ace), reward, done, _ = env.step(action)
    episode.append((state, action, reward))
  
  while len(episode) > 0:
    state, action, reward = episode.pop()
    g = DISCOUNT * g + reward
    
    if (state, action, reward) not in episode:
      count[state + tuple([action])] += 1
      q[state + tuple([action])] += (g - q[state + tuple([action])])/count[state + tuple([action])]
      g_action = np.argmax(q[state])

      if np.random.random() < epsilon:
        policy[state] = np.random.randint(0, 2)
      else:
        policy[state] = g_action

for i in range(2):
  for j in range(2):
    color_plot(q[:,:,i,j], 0.5, 10.5, 11.5, 21.5)
    str_i = 'ace' if i else 'no_ace'
    str_j = 'hit' if j else 'stick'
    plt.savefig('on_' + str_i + '_' + str_j + '.png')

## Off-Policy Monte Carlo Prediction 

pi = (np.random.random((32, 11, 2)) < 0.5).astype(int)
b = np.random.random((32, 11, 2))
b = np.stack((b, 1-b), axis=3)
q = np.zeros((32, 11, 2, 2))
count = np.zeros((32, 11, 2, 2))
DISCOUNT = 1

for _ in range(1000000):
  hand, show, ace = env.reset()  
  done = False
  episode = []
  while not done:
    state = (hand, show, int(ace))
    action = 0 if np.random.random() < b[state][0] else 1
    (hand, show, ace), reward, done, _ = env.step(action)
    episode.append((state, action, reward))
  
  g = 0
  w = 1
  while len(episode) > 0 and w != 0:
    state, action, reward = episode.pop()
    g = DISCOUNT * g + reward
    sa = state + tuple([action])
    count[sa] += w
    q[sa] += (w * (g - q[sa])) / count[sa]

    # pi[state] = np.argmax(q[state])

    w *= pi[state]/b[sa]

for i in range(2):
  for j in range(2):
    color_plot(q[:,:,i,j], 0.5, 10.5, 11.5, 21.5)
    str_i = 'ace' if i else 'no_ace'
    str_j = 'hit' if j else 'stick'
    plt.savefig('off_eval_' + str_i + '_' + str_j + '.png')

## Off Policy Monte Carlo Control

b = np.ones((32, 11, 2, 2)) * 0.5
q = np.random.random((32, 11, 2, 2))
count = np.zeros((32, 11, 2, 2))
pi = np.argmax(q, axis=3)
DISCOUNT = 1

for _ in range(10000000):
  hand, show, ace = env.reset()
  done = False
  episode = []
  while not done:
    state = (hand, show, int(ace))
    action = np.random.choice(range(len(b[state])), p=b[state])
    (hand, show, ace), reward, done, _ = env.step(action)
    episode.append((state, action, reward))
  g = 0.0
  w = 1.0
  while len(episode) > 0:
    state, action, reward = episode.pop()
    sa = state + tuple([action])

    g = DISCOUNT * g + reward
    count[sa] += w
    q[sa] += (w * (g - q[sa])) / count[sa]
    pi[state] = np.argmax(q[state])
    if pi[state] != action:
      break
       
    w *= 1/b[sa]

for i in range(2):
  for j in range(2):
    color_plot(q[:,:,i,j], 0.5, 10.5, 11.5, 21.5)
    str_i = 'ace' if i else 'no_ace'
    str_j = 'hit' if j else 'stick'
    plt.savefig('off_' + str_i + '_' + str_j + '.png')


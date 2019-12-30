import numpy as np

q = np.random.normal(0.0, 2.0, size=10)
std = 0.5


## Action Value Estimation

q_a = np.array([0.0] * len(q))
n_a = np.array([0] * len(q))

for _ in range(1000):
    action = np.random.randint(10)
    reward = np.random.normal(q[action], std)
    n_a[action] += 1
    q_a[action] += (reward - q_a[action]) / n_a[action]

print(q)
print(q_a)


## epsilon-greedy Methods

q_a = np.array([0.0] * len(q))
n_a = np.array([0] * len(q))

def greedy_epsilon(epsilon):
  for _ in range(5000):
    action = None
    if np.random.random() < 1 - epsilon:
      action = np.argmax(q_a)
    else:
      action = np.random.randint(10)
    reward = np.random.normal(q[action], std)
    n_a[action] += 1
    q_a[action] += (reward - q_a[action]) / n_a[action]

greedy_epsilon(epsilon = 0.1)

print(q)
print(q_a)


## Optimistic Initialization

q_a = np.array([5.0] * len(q))
greedy_epsilon(epsilon = 0.1)

print(q)
print(q_a)


## Moving Rewards

def alpha(action):
  # return 1/n_a[action]
  return 0.1


## Softmax

q_a = np.array([0.0] * len(q))
n_a = np.array([0] * len(q))

def softmax(x):
    return np.exp(x) / np.sum(np.exp(x), axis=0)

def softmax_action(beta):
  for t in range(5000):
    action = np.random.choice(
        np.arange(len(q_a)), 
        p = softmax(q_a)
    )
    reward = np.random.normal(q[action], std)
    n_a[action] += 1
    q_a[action] += alpha(action) * (reward - q_a[action])

softmax_action(beta = 1)
print(q)
print(q_a)


## Upper-Confidence Bound Action Selection

q_a = np.array([0.0] * len(q))
n_a = np.array([0] * len(q))

def ucb(c):
  for t in range(5000):
    action = np.argmax([q_a[i] + c * np.sqrt(np.log(t+1)/np.max([n_a[i], 1])) for i in range(len(q_a))])
    reward = np.random.normal(q[action], std)
    n_a[action] += 1
    q_a[action] += alpha(action) * (reward - q_a[action])

ucb(c = 2)
print(q)
print(q_a)


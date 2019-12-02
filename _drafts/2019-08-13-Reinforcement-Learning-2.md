---
layout: post
title: 'Basics of Banditry 1: Reinforcement Learning'
date: 2019-09-14 13:00:00 +0500
categories: RL Python
comments: true
summary: 'The start of our adventures into Reinforcement Learning'
---

Hello! My name's Jason and this is my friend Mallory.

{% include image.html url="/assets/images/1-hello.png" description="Hi" %}

Reinforcement Learning is one of the prominent forefronts of artificial intelligence, though it is much further from the spotlight as compared to the Neural Networks and Support Vector Machines that are finding many business and practical applications. Through this series, I hope to alleviate this lack of knowledge and provide you with an introduction into the world of RL.

Hold on, I'm an undergraduate student, where did I learn everything? Well, most of it stems from the textbook [Reinforcement Learning by Richard Sutton and Andrew Barto][rlbook], it's an great read and this series is based on it.

As for difficulty, this will be focused towards people who have never touched RL or even machine learning before, I will be going over mathematical equations and python code but those can be ignored if you aren't interested in the technical details.

In case you want to take a look at the code, here is a <a href="/assets/files/rl.py">python file</a> and a <a href="/assets/files/rl.ipynb">Jupyter Notebook</a>.

Now let's jump right into the basics.

## Why Reinforcement Learning?

So why use Reinforcement Learning when there are all sorts of Neural Networks, decision trees, and SVMs exist? 

1. Data: RL does not assume data is sampled from an independent and identically distribution (i.i.d.) like in other forms of machine learning, they are usually very correlated, e.g. an action chosen in a video game might have an impact several time steps later. Additionally, RL has no need for costly labeled data.

2. Adaptiveness: RL is able to adapt quickly to new unseen situations and performs well in highly complex states without requiring prior knowledge to provide useful insights.

3. Intelligence: RL has the potential to attain superhuman capabilities. Supervised machine learning generally uses data collected from humans and emulates that behaviour. Whereas reinforcement learning performs the tasks itself and can, in some cases, outperform the top humans. Because of this, some see RL as the most likely path to achieving Artificial General Intelligence (a machine as capable as a human in every aspect). 

## $k$-armed Bandits

No this is not about mutant highway robbers, this about slot machines and choices. Consider a slot machine (often referred to as a 1-armed Bandit), but in this case, it has $k$ levers to pull, each leading to a play on the machine with different, independent odds of winning. In order to maximize one's winnings over repeated plays, one would seek out the highest performing levers and keep pulling them. 

{% include image.html url="/assets/photos/bandits.PNG" description="The above figure details a 10-armed version of the $k$-armed bandit problem. Each lever has an associated reward sampling distribution, where some levers are more likely to have higher rewards." %}

Here's the setup for this environment in python:
```python
import numpy as np

q = np.random.normal(0.0, 2.0, size=10)
std = 0.5
```

Let's denote the choice we make out of the $k$ levers as the action at time $t$ as $A_t$, and the immediate reward as $R_{t+1}$ (the time is different since technically the reward occurs in the next time step but you'll see it expressed both ways in literature). We define the quality of taking $a$ as the expected reward from taking $a$:

$$q_*(a) = \mathbb{E}[R_{t+1}|A_t=a]$$

Where the $*$ represents the optimal (true) of action value function, generally we don't have access to this so we have an estimate denoted as $q(a)$. 

At any time step, there must be an action with the highest estimated value. Always choosing this action is known as *exploiting*, and to choose any non maximal choices is *exploring*. The trade-off between these 2 options is short term gain in exploitation, or long term gain in exploration. 

## Action Value Estimation

So how do we update the action value function $q(a)$ so that it better approximates $q_*(a)$? One idea is to keep an average of the returns that occurred from taking that action. The naive way to code this would be to keep an array of all past rewards and calculate the average every time step, which takes $\mathcal{O}(n)$ time and space per time step. A way to do this in $\mathcal{O}(1)$ time and space is to keep track of a count variable and update using:

$$
\begin{align}
Q_{n+1} &= \frac{1}{n} \sum^{n+1}_{i=1}R_i \\
&= \frac{1}{n} \big(R_{n+1} + (n-1)\frac{1}{n-1}\sum^{n}_{i=1}R_i \big)\\
&= \frac{1}{n} (R_{n+1} + (n-1)Q_n)\\
&= \frac{1}{n} (R_{n+1} + nQ_n - Q_n)\\
&= Q_n + \frac{1}{n} (R_{n+1} - Q_n)\\
\end{align}
$$

Here's the accompanying code:

```python
q_a = np.array([0.0] * len(q))
n_a = np.array([0] * len(q))

for _ in range(5000):
    action = np.random.randint(10)
    reward = np.random.normal(q[action], std)
    n_a[action] += 1
    q_a[action] += (reward - q_a[action]) / n_a[action]
```

## $\epsilon$-greedy Methods

Given our action value function, how do we decide on which action to take? If we only choose the greedy (highest estimated value) action, the agent might spend all its time exploiting a good action without ever even exploring the *optimal* action.

We can combat this by using $\epsilon-greedy$ methods, where we have a $1-\epsilon$ chance of taking the greedy action, and an $\epsilon$ chance to take an action at random (includes the greedy action). However, if we have a constant $\epsilon$, then even after an infinite number of iterations and convergence to the optimal value function, we will not take the optimal (which is now the greedy option) action with more than $(1-\epsilon) + (1/n)\epsilon$ chance, where $n$ is the total number of actions possible.

```python
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
```

## Optimistic Initialization

One way to improve this algorithm with a bit of knowledge about the rewards is to initialize the values of $q$ optimistically, as in, we over estimate all of the expected rewards so that each reward is explored at least once early on in the algorithm (since every initial value is eventually the greedy choice since they are higher than those of any true ones). In our example, we would just change the first line to:

```python
q_a = np.array([5.0] * len(q))
```

## Moving Rewards

Consider a situation where the rewards of the levers slowly change over the course of time. In this case, it would make sense to weigh recent rewards more compared to past rewards. We can weigh our update using a step-size $\alpha \in (0, 1]$ instead:

$$Q_{n+1} = Q_n + \alpha [R_n - Q_n]$$

We can also set $\alpha$ to be a function that changes over time (e.g. setting it to $\frac{1}{n}$ makes it the sample average from above). 

```python
def alpha(action):
  # return 1/n_a[action]
  return 0.1

```

## Upper-Confidence Bound Action Selection

In $\epsilon$-greedy methods, we do exploration randomly, with no discernment of which non-greedy action should be tried. In Upper-Confidence Bound (UCB) action selection we take into account how close the estimates are to being maximal and the uncertainty of estimates using:

$$\underset{a}{\mathrm{argmax}} \bigg[q(a) + c \sqrt{\frac{ln(t)}{n_a(a)}}\bigg]$$

Where $a$ is the action, $t$ is the current time step, $n_a(a)$ is how many times $a$ was selected, and $c$ is an adjustable parameter that controls the degree of exploration. So the $q(a)$ term measures the how maximal the estimate is, the $c\sqrt{\frac{ln(t)}{n_a(a)}}$ term measures how uncertain the estimate is. Here's an implementation of it:

```python
def ucb(c):
  for t in range(5000):
    action = np.argmax([q_a[i] + c * np.sqrt(np.log(t+1)/np.max([n_a[i], 1])) for i in range(len(q_a))])
    reward = np.random.normal(q[action], std)
    n_a[action] += 1
    q_a[action] += alpha(action) * (reward - q_a[action])

ucb(c = 2)
```

## Gradient Bandits

So far, we've been estimating the action value function $q(a)$ and then apply some method to choose an action ($\epsilon$-greedy, UCB). We can also apply a function of numerical preference $H_t(a)$ to each action. Which we then put through as soft-max function in order to obtain $\pi(a)$, a function for the probability of taking action $a$. This algorithm is based on the idea of Stochastic Gradient Descent (SGD). The preference function is updated with: 

$$
\begin{align}
H_{t+1}(A_t) &= H_t(A_t) + \alpha (R_t - \bar{R}_t)(1-\pi_t(A_t)) \\
H_{t+1}(a) &= H_t(a) + \alpha (R_t - \bar{R}_t)\pi_t(a) \\
\end{align}
$$

Where $\bar{R}_t$ is the average of all past rewards, $A_t$ is the action selected at time $t$, and all other actions update according to the second equation. The reward average acts as a baseline for how good a reward is, if the difference is very positive that action is more preferred in the future. Here's an implementation:

```python
def softmax(x):
    return np.exp(x) / np.sum(np.exp(x), axis=0)

def sgd(a):
  reward_avg = 0
  for t in range(n):
    random_num = np.random.random()
    sm = softmax(q_a)
    action = None
    for i in range(len(sm)):
      random_num -= sm[i]
      if random_num <= 0:
        action = i
        break
    
    n_a[action] += 1
    reward = np.random.normal(q[action], std)
    reward_avg += 1/(t+1) * (reward - reward_avg)

    q_a[action] += a * (reward - reward_avg) * (1 - sm[action])
    for i in range(len(sm)):
      if i != action:
        q_a[i] -= a * (reward - reward_avg) * (sm[i])

sgd(0.1)
```

## Additional Notes

For a deeper dive on Artificial Intelligence, check out this [WaitButWhy][wbw] blog post.

{% if page.comments %}
  {%- include disqus.html -%}
{% endif %}

[rlbook]: http://incompleteideas.net/book/RLbook2018.pdf
[wbw]: https://waitbutwhy.com/2015/01/artificial-intelligence-revolution-1.html

---
layout: post
title: 'Basics of Banditry 2: The Environment'
date: 2019-12-02 13:00:00 +0500
categories: RL Python
comments: true
summary: ''
---

In the previous part we outlined the basics of the value function and how to choose actions from it. Today we will more formally define a general representation of an environment and generate a policy from it.

For the previous parts:

Part 1: [Reinforcement Learning]({{ site.baseurl }}{% post_url 2019-09-14-Reinforcement-Learning %}) 


## Markov Decision Processes

A Markov Decision Process (MDP) is a sequence of decision making events where the outcome is determined partly randomly and partly controlled based on action. They also have the Markov property: all conditional probabilities of future states depend only on the current state and not past ones. MDP leads to a sequence of States ($S$), Actions ($A$), and Rewards ($R$):

$$
S_0,A_0,R_1,S_1,A_1,R_2,S_2,A_2,R_3,\ldots
$$

Given State $s$ and Action $a$, we define probabilities of Next State $s'$ and Reward $r$ with $p(s',r\|s,a)$, where:

$$
\forall s\in S, \forall a \in A(s), \sum_{s'\in S} \sum_{r\in R} p(s',r|s,a) = 1
$$

The split between environment and agent is generally defined as the limit of absolute control, even things like motors and muscles are sometimes considered part of the environment. The representations of the states and actions vary wildly between different situations and have a strong impact on the performance of the algorithm. 

<!-- There are several other forms of representing $p$ that are sometimes useful:
State Transition Probabilities
Expected rewards for State-Action Pairs
Expected Rewards for State - Action - Next-State triples:}

$$
\begin{align}
    p(s'|s,a) &= \sum_{r\in R}p(s',r|s,a) \\
    r(s,a) &= \sum_{r\in R}r \sum_{s'\in S}p(s',r|s,a) \\
    r(s,a) &= \sum_{r\in R}r \frac{p(s',r|s,a)}{p(s'|s,a)} \\
\end{align}
$$ -->

## Goals and Rewards

Our goal and associated reward function defines *what* we want the agent to accomplish, but not *how* we want it to achieve it. Generally, we define the reward function as the expected return for future rewards with respect to time $t$ as $G_t$. Since the task might never end, we discount future rewards with rate $\gamma$ between 0 and 1:

$$
\begin{align}
    G_t &= \sum_{k=0}^\infty \gamma^k R_{t+k+1}\\
    &= R_{t+1} + \gamma R_{t+2} + \gamma^2 R_{t+3} + \gamma^3 R_{t+4} + \ldots \\
    &= R_{t+1} + \gamma \big(R_{t+2} + \gamma R_{t+3} + \gamma^2 R_{t+4} + \ldots\big) \\
    &= R_{t+1} + \gamma G_{t+1}
\end{align}
$$

Although this is an infinite sum, as long as the discount is less than 1 and reward stays constant, the return stays finite as well. 

In order to combine the episodic and never-ending scenarios, we assume a constant state of reward 0 after the terminal step in episodic cases so it too is infinite without any effect to the reward function.

The design of this function is paramount to the agent's performance. Suppose we were to simulate an agent in a room where it needs to reach the door, where it receives a reward of +1, and at every time step it suffers a reward of -1. And say the simulation terminates whenever a wall, window or door is reached. In this situation, the agent would quickly learn that throwing itself on a wall or out a window is the most optimal solution, as it incurs the least negative reward. 

## Value Functions

Given our policy $\pi (a\|s)$ (probability the agent takes action $a$ in state $s$), we can define how rewarding it is for an agent to be in a certain state. 

One way to achieve this is a state value function $v_\pi(s)$, which defines the value being at state $s$ and following $\pi$ from then on:

$$v_\pi(s) = \mathbb{E}_\pi [G_t | S_t = s] = \mathbb{E}_\pi \bigg[\sum_{k=0}^\infty \gamma^k R_{t+k+1}| S_t = s\bigg]$$

Another way is a state-action value function $q_\pi(s, a)$, which defines the value of taking action $a$ in state $s$ and following $\pi$ from then on:

$$q_\pi(s,a)=\mathbb{E}[G_t | S_t = s, A_t=a] = \mathbb{E}_\pi \bigg[\sum_{k=0}^\infty \gamma^k R_{t+k+1}| S_t = s, A_t=a\bigg]$$

Relational Equations:

$$
\begin{align}
    v_\pi(s) &= \sum_{a\in A} \pi(a|s) * q_\pi(s, a) \\
    q_\pi(s, a) &= \sum_{s', r\in S, R} p(s', r | s, a) * (r + \gamma v_\pi(s'))
\end{align}
$$

Value functions satisfy a recursive relationship:

$$
\begin{align}
    v_\pi(s) &= \mathbb{E}_\pi [G_t | S_t = s] \\
    &= \mathbb{E}_\pi [R_{t+1} + \gamma G_{t+1} | S_t = s] \\
    &= \sum_a \pi(a|s) \sum_{s', r} p(s', r | s, a) \big[r + \gamma \mathbb{E}_\pi [G_{t+1} | S_{t+1} = s']\big] \\
    &= \sum_a \pi(a|s) \sum_{s', r} p(s', r | s, a) \big[r + \gamma v_\pi(s')\big] \\
\end{align}
$$

This is known as the *Bellman Equation*, essentially a weighted average of all possible actions, next states, and discounted future rewards.

## Optimization Policies

To improve our policies performance, we attempt to reach a policy greater than or equal to all others, the optimal policy $\pi_*$. We can do this by first trying to reach an optimal value function:

$$
\begin{align}
    v_*(s) &= \max_\pi v_\pi(s) \\
    q_*(s,a) &= \max_\pi q_\pi(s,a) \\
\end{align}
$$

<!-- Relationships:

$$v_*(s)=\max_a q_{\pi_*}(s,a) = \max_a \sum_{s', r} p(s', r | s, a) [r + \gamma v_*(s')]$$

$$q_*(s,a) = \sum_{s',r}p(s', r | s, a) [r + \gamma  v_*(s')]= \sum_{s',r}p(s', r | s, a) [r + \gamma \max_a' q_*(s',a')]$$ -->

Given an optimal value function, any policy that assigns nonzero probabilities only to maximal actions are optimal policies.

Although it is possible to solve these equations directly, we require knowledge of the dynamics of the environment, the Markov property, and enough computing power as the number of states and actions could be arbitrarily large. If we solve these, then we can directly compute $\pi_*$:

$$
\begin{align}
    \pi_*(a|s) =\max_a q_{*}(s,a) \\
    \pi_*(a|s) = \sum_{s', r} p(s', r | s, a) [r + \gamma v_*(s')] \\
\end{align}
$$

In general, we do not know the dynamics of the environment nor do we have the capability to compute the direct solution. Thus, we need to find a way to reach approximate states of these optimal value functions.

## Policy Evaluation

A method to find an approximately optimal value functions is Iterative Policy Evaluation. Which calculates state values using dynamic programming until the recursive formula becomes stable with less than $\Delta$ change between iterations. Let $v_k, v_{k+1}$ represent the current and next approximate value functions we have for the optimal $v_*$.

$$v_{k+1}(s) = \max_a \sum_{s',r} p(s',r|s,a)\big[r+\gamma v_k(s')\big]$$

Iterative Policy Improvement compares the values of taking each action at each state, and if they're better than their respective state values, then the policy is changed accordingly.

### Value Iteration

When we stop policy evaluation after only one sweep, and then iteratively improve the policy, we get a simple update operation called value iteration. The improvement cycle follows:

$$\pi'(s) = {\arg\max}_a q_\pi (s,a)$$

<!-- Consider an example of a Gambler's game: a gambler starts with some amount of money, their goal is to bet on weighted coin flips, winning back double his bet for each head, and losing his bet for tails. He wins when he has 100 dollars and loses upon reaching 0 dollars. Here is the python code for this:

```python
import numpy as np

# States represent how much money our gambler has
NUM_STATES = 101 # 0 - 100
V_s = np.zeros(NUM_STATES)
V_s[NUM_STATES - 1] = 1 # Only place where reward is 1

V_index = np.array(range(1, 100))

policy = np.ones(99) # 1 - 99 are not terminal states

def get_actions(s):
  return list(range(1, min(s, 100 - s) + 1))

DELTA_LIM = 0.000001
DISCOUNT = 1

p_h = 0.4

# At most there are 50 actions to take (at state 50, actions: [1,50])

prob_table = []

for s in V_index: # [1, 99]
  heads = []
  tails = []
  for a in get_actions(s): # [1, min(s, 100 - s)]
    heads.append([p_h, 0, s + a])
    tails.append([1 - p_h, 0, s - a])

  for _ in range(50 - len(heads)):
    heads.append([0,0,0])
    tails.append([0,0,0])

  prob_table.append(np.stack([np.array(heads), np.array(tails)], axis = 1))
  
prob_table = np.array(prob_table)
prob_table.shape

def value_iteration_optimized(V_s):
  delta = DELTA_LIM + 1
  
  new_policy = None
  
  while delta > DELTA_LIM:
    delta = 0
  
    v = V_s.copy()
    
    reward = prob_table[:, :, :, 1] + DISCOUNT * v[prob_table[:, :, :, 2].astype(np.intp)] # (99, 50, 2)
    reward *= prob_table[:, :, :, 0] # (99, 50, 2)
    reward = np.sum(reward, axis=2) # (99, 50)
    
    V_s = np.max(reward, axis = 1) # (99)
    V_s = np.array([0] + V_s.tolist() + [1])
    
    new_policy = np.argmax(reward, axis = 1) + 1
    
    delta = np.amax(np.abs(v - V_s))
    
    print("DELTA", np.round(delta, 6))
    
  return new_policy, V_s



def P(s, a):
  return [[p_h, s + a], [1-p_h, s - a]]

def value_iteration(V_s):
  delta = DELTA_LIM + 1
  
  new_policy = np.ones(99)
  
  while delta > DELTA_LIM:
    delta = 0
    for s in range(1, NUM_STATES - 1):
      v = V_s[s]
      rewards = []
      for a in get_actions(s):
        reward = 0
        for prob, next_state in P(s, a):
          if next_state == 101:
            reward += prob * (1 + DISCOUNT * V_s[next_state])
          else:
            reward += prob * (DISCOUNT * V_s[next_state])
        rewards.append(reward)
      V_s[s] = max(rewards)
      new_policy[s - 1] = np.argmax(rewards) + 1
      delta = max(delta, abs(v - V_s[s]))
    print("DELTA", np.round(delta, 6))
    
  return new_policy, V_s

policy, V_s = value_iteration_optimized(V_s)
print(policy)
print(V_s)

import matplotlib.pyplot as plt
plt.bar(range(1,100), policy)
plt.show()

plt.plot(V_s)
plt.show()
``` -->
### Asynchronous DP

In our Iterative DP algorithm, we update the states one by one. In reality, we have no need to update all at once, we could update one state multiple times before updating another once. However, for the guarantee of convergence, we cannot stop updating states permanently after some point in the computation. We do this to improve the rate of progress as some states need their values updated more often than others.

In practice, we can run an iterative DP algorithm at the same time that the agent is experiencing the environment. The experience gives the algorithm states to update, and simultaneously, the latest value and policy guide the agent's decisions.

### Generalized Policy Iteration

Policy iteration is a mix of making the Value Function consistent with the policy (policy evaluation), and the other is making the policy greedy with respect to the Value Function (policy improvement). These processes complete before the other runs. 

In Value iteration, we only do one sweep of policy evaluation before policy improvement. In Async DP, the evaluation and improvement process are even more closely inter-weaved.

Generalized Policy Iteration (GPI), is the general idea of letting policy evaluation and policy improvement interact, independent of their granularity. Most of RL methods fall under this. And once they processes become stable, we must have reached the optimal state and policy.
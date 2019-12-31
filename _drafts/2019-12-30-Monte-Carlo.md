---
layout: post
title: 'Basics of Banditry 3: Monte Carlo'
date: 2019-12-30 13:00:00 +0500
categories: RL Python
comments: true
summary: ''
---

In the previous part we learned how to generate an approximate value function and policy in an environment we had complete control over. Today, we will be going over a method for when we don't have complete control over the environment and can only learn from experience.

For the previous parts:

Part 1: [Reinforcement Learning]({{ site.baseurl }}{% post_url 2019-09-14-Reinforcement-Learning %})

Part 2: [The Environment]({{ site.baseurl }}{% post_url 2019-12-02-The-Environment %})

Code used:

- <a href="/assets/files/3-mc.py">Python file</a>
- <a href="/assets/files/3-mc.ipynb">Jupyter Notebook</a>

## Monte Carlo

For most situations in the real world, it is difficult to calculate the transition function for an environment as we did in previous methods. Monte Carlo is a method of learning and approximating value functions that only requires experience, e.g. an agent running around an environment until it hits a goal. It is based on averaging sample discounted returns and is only defined for episodic tasks, where all episodes eventually terminate no matter what actions are chosen. This means that the estimates for each state is independent and doesn't depend on the estimates of other states. 

## First-Visit and Every-Visit

To estimate values for State Value functions, we average the returns observed after visits to each state. There are 2 different ways to do it:
- The first-visit MC method averages returns of the first visit to each state in an episode.
- The every-visit MC method averages returns of all visits to state.

Both converge to the optimal state value function with infinite experience.

Blackjack is a good example for when it is hard to use DP (the probability of the next cards can be difficult to calculate), if you aren’t familiar with blackjack it uses these rules [https://bicyclecards.com/how-to-play/blackjack/]. A Blackjack environment has already been coded in OpenAI's Gym [https://gym.openai.com/] which we will be using:

```python
import numpy as np
import gym
env = gym.make('Blackjack-v0')

# Policy that only sticks if the sum is 20 or 21
policy = np.ones((32, 11, 2))
policy[20:22, :, :] = 0
policy[9:11, :, 1] = 0

V_s = np.zeros((32, 11, 2))
ret = np.zeros((32, 11, 2))
count = np.zeros((32, 11, 2))

DISCOUNT = 1
```

We will evaluate the above policy with First Visit MC:

```python
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
```
We need to run this for a significant number of episodes to reduce the effects of randomness and in order to have a high probability of reaching every state.

{% include image.html 
  url="/assets/images/first_no_ace.png" 
  description="Value function for when there is no usable ace." 
%}

{% include image.html 
  url="/assets/images/first_ace.png" 
  description="Value function for when there is a usable ace." 
%}

## Exploring Starts

If we do not have a model of the environment, we need more than just state values, we need action values. These are also estimated by averaging sample discounted returns. The issue with this is that many state-action pairs might never be visited, and we need to estimate the value of all pairs to guarantee optimal convergence.

This is especially difficult if we have a deterministic policy, where only one action will be taken in each state. In other words, we need to ensure exploration of all actions for each state. One way is to only consider policies that are stochastic with nonzero probability for all action state pairs. Another method is 'exploring starts', where every state action has a non zero probability of being selected as the starting state. 

Here's the accompanying code:
```python
usable = np.zeros((32, 11, 2, 2))
usable[1:22, 1:12] = 1

q = np.random.random((32, 11, 2, 2)) * usable
policy = np.argmax(q, axis=3)
ret = np.zeros((32, 11, 2, 2))
count = np.zeros((32, 11, 2, 2))
DISCOUNT = 1

for _ in range(500000):
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
```

Which gives us:

{% include image.html 
  url="/assets/images/first_ace.png" 
  description="Value function for when there is a usable ace." 
%}
{% include image.html 
  url="/assets/images/first_ace.png" 
  description="Value function for when there is a usable ace." 
%}
{% include image.html 
  url="/assets/images/first_ace.png" 
  description="Value function for when there is a usable ace." 
%}
{% include image.html 
  url="/assets/images/first_ace.png" 
  description="Value function for when there is a usable ace." 
%}

## Policy Making

In Monte Carlo methods, we still want to follow Generalized Policy Iteration, where we maintain an approximate policy and approximate value function, and repeatedly optimize them. We alternate between evaluation and improvement episode by episode. After each, the observed returns are used to update the visited states, and then the policy is improved for all these states.

There are 2 methods to go about approximating the optimal policy:
- On-policy means to evaluate or improve the policy used to generate the data. 
- Off-policy methods evaluate or improve a different policy than the data generating one. 

In any case, we still want a positive chance to select any actions in both situations to guarantee convergence.

In on-policy, the policies are generally 'soft', as in the policy always has a non zero probability of selecting every action in every state, similar to the epsilon-greedy policies. We still use first visit MC methods to estimate action values.

In off-policy, we consider 2 policies, one that is learned about and becomes the optimal policy (target policy $\pi$), and another than is more exploratory and generates behaviour (behaviour policy $b$).

Comparatively, on-policy is simpler and converges faster, but off-policy is more powerful and general (on-policy is a special case of off-policy when both target and behaviour policies are the same). 

The prediction problem is when both $\pi$ and $b$ are fixed and we want to estimate $v_\pi$ or $q_\pi$. The problem in this is that the episodes follow $b$, while we try to estimate values for $\pi$. To accomplish this, we require every action taken under $\pi$ also be taken in $b$, i.e. $\pi(a|s) \Rightarrow b(a|s)$. This is also known as the assumption of converge, which also gives us that $b$ must be stochastic where it is not identical to $\pi$, but $\pi$ can be a deterministic policy. 

Importance sampling is a technique for estimating expected values from one distribution given samples from another ($\pi$ from $b$). We apply this to off-policy learning by weighing returns according to the relative probability of their trajectories occurring under $\pi$ and $b$, called the importance-sampling ratio. The probability of the state-action trajectory $A_t, S_{t+1}, A_{t+1}, ..., S_T$ occurring under $pi$ is:

$$\pi(A_t|S_t)p(S_{t+1}|S_t, A_t)\pi(A_{t+1}|S_{t+1}) ... p(S_T|S_{T-1}, A_{T-1})$$
$$ = \prod_{k=t}^{T-1} \pi(A_k|S_k)p(S_{k+1}|S_k, A_k)$$

So the relative probability is:
$$\rho_{t:T-1} = \frac{\prod_{k=t}^{T-1} \pi(A_k|S_k)p(S_{k+1}|S_k, A_k)}{\prod_{k=t}^{T-1} b(A_k|S_k)p(S_{k+1}|S_k, A_k)} = \frac{\prod_{k=t}^{T-1} \pi(A_k|S_k)}{\prod_{k=t}^{T-1} b(A_k|S_k)} $$

So now we weight all our returns in $b$ with $\rho_{t:T-1}$ to transform it to returns for $\pi$. In doing so, let time continue to count up from episode to episode (ends at time 100, next one starts at 101) and here are some definitions:

$\mathcal{T}(s)$: a set of either all the time steps $s$ was visited (every visit method), or the first time steps of each episode that $s$ was visited (first visit method).

$T(t)$: first termination after time $t$.

$G(t)$: return after $t$ up through $T(t)$.

So we have $\{G_t\}_{t\in \mathcal{T}(s)}$, the returns that matter to state $s$, and $\{\rho_{t:T-1}\}_{t\in \mathcal{T}(s)}$, the corresponding importance-sampling ratios. 

Ordinary importance sampling:
$$V(s)=\frac{\sum_{t\in\mathcal{T}(s)}\rho_{t:T-1}G_t}{|\mathcal{T}(s)|}$$
Weighted importance sampling: (0 is denominator is zero)
$$V(s)=\frac{\sum_{t\in\mathcal{T}(s)}\rho_{t:T-1}G_t}{\sum_{t\in\mathcal{T}(s)}\rho_{t:T-1}}$$

The ordinary one has unbounded variance of the ratios, the weighted one is bounded by the value 1. But the ordinary one is unbiased where the weighted one is. But with the assumption of bounded returns, the variance of weighted converges to 0, in general the weighted one is better.

\textbf{Discounting-aware Importance Sampling}

Consider long episodes with discount values much lower than 1. After the first action, the return is just the reward, but the importance sampling ratio is a product of a string of factors equal to the length of the episode. In ordinary importance sampling, the return is scaled by the entire product, when we only need to scale by the first factor (since the return is already determined after the first one). These additional factors do nothing but add to variance.

Think of discounting as a probability of termination, as in, a degree of partial termination. For any $\gamma \in [0,1)$, think of $G_0$ as partly terminating after one step, to degree $1-\gamma$, giving the return of $R_1$. And with the second step to degree $(1-\gamma)\gamma$, i.e. degree of termination in 2 * degree of non termination in 1, and so forth. 

The partial returns here are called flat partial returns (flat means absense of discounting and partial meaning they do not extend all the way to termination $T$, but rather at $h$, the horizon):
\begin{align*}
    \Bar{G}_{t:h} &= R_{t+1} + R_{t+2} + \cdots + R_{h}, \ \ 0 \leq t < h \leq T
    \shortintertext{The full return $G_t$ can be represented as:}
    G_t &= R_{t+1} + \gamma R_{t+2} + \cdots + \gamma^{T-t-1} R_{T}\\
    &= (1-\gamma)R_{t+1} \\
    &+ (1-\gamma)\gamma (R_{t+1} + R_{t+2}) \\
    &+ (1-\gamma)\gamma^2 (R_{t+1} + R_{t+2} + R_{t+3}) \\
    &\  \vdots\\
    &+ (1-\gamma)\gamma^{T-t-2} (R_{t+1} + R_{t+2} + \cdots + R_{T-1}) \\
    &+ \gamma^{T-t-1} (R_{t+1} + R_{t+2} + \cdots + R_T) \\
    &= \gamma^{T-t-1} \Bar{G}_{t:T} + (1-\gamma)\sum_{h=t+1}^{T-1} \gamma^{h-t-1} \Bar{G}_{t:h}
\end{align*}

Now we have to scale these flat partial returns by an importance sampling ratio (also partial to fit). In doing so, we only need the ratios up to $h$, as we only involve rewards up to $h$.

Ordinary importance-sampling estimator:
$$V(s) = \frac{
    \sum_{t\in \mathcal{T}(s)} \big(
        \gamma^{T(t) - t - 1} \rho_{t:T(t)-1} \Bar{G}_{t:T(t)} + (1-\gamma)\sum_{h=t+1}^{T(t)-1} \gamma^{h-t-1} \rho_{t:h-1} \Bar{G}_{t:h}
    \big)
}{
|\mathcal{T}(s)|
}$$

Weighted importance-sampling estimator:
$$V(s) = \frac{
    \sum_{t\in \mathcal{T}(s)} \big(
        \gamma^{T(t) - t - 1} \rho_{t:T(t)-1} \Bar{G}_{t:T(t)} + (1-\gamma)\sum_{h=t+1}^{T(t)-1} \gamma^{h-t-1} \rho_{t:h-1} \Bar{G}_{t:h}
    \big)
}{
    \sum_{t\in \mathcal{T}(s)} \big(
        \gamma^{T(t) - t - 1} \rho_{t:T(t)-1} + (1-\gamma)\sum_{h=t+1}^{T(t)-1} \gamma^{h-t-1} \rho_{t:h-1}
    \big)
}$$

These 2 estimators are discounting-aware importance sampling estimators, with no affect if $\gamma = 1$.

Let's do some derivations for $V(s)$:

$$
\frac{
    \sum_{t\in \mathcal{T}(s)} \big(
        \gamma^{T(t) - t - 1} \frac{\prod_{k=t}^{T-1} \pi(A_k|S_k)}{\prod_{k=t}^{T-1} b(A_k|S_k)} \Bar{G}_{t:h} + (1-\gamma)\sum_{h=t+1}^{T(t)-1} \gamma^{h-t-1} \frac{\prod_{k=t}^{h-1} \pi(A_k|S_k)}{\prod_{k=t}^{h-1} b(A_k|S_k)}  \Bar{G}_{t:h}
    \big)
}{
    \sum_{t\in \mathcal{T}(s)} \big(
        \gamma^{T(t) - t - 1} \frac{\prod_{k=t}^{T-1} \pi(A_k|S_k)}{\prod_{k=t}^{T-1} b(A_k|S_k)}  + (1-\gamma)\sum_{h=t+1}^{T(t)-1} \gamma^{h-t-1} \frac{\prod_{k=t}^{h-1} \pi(A_k|S_k)}{\prod_{k=t}^{h-1} b(A_k|S_k)} 
    \big)
}
$$

$t = T - 1$:
\begin{align*}
&\frac{
    \gamma^{T(t) - (T-1) - 1} \frac{\prod_{k=(T-1)}^{T-1} \pi(A_k|S_k)}{\prod_{k=(T-1)}^{T-1} b(A_k|S_k)} \Bar{G}_{T-1:T} + (1-\gamma)\sum_{h=(T-1)+1}^{T(t)-1} \gamma^{h-(T-1)-1} \frac{\prod_{k=(T-1)}^{h-1} \pi(A_k|S_k)}{\prod_{k=(T-1)}^{h-1} b(A_k|S_k)}  \Bar{G}_{T-1:h}
}{
    \gamma^{T(t) - (T-1) - 1} \frac{\prod_{k=(T - 1)}^{T-1} \pi(A_k|S_k)}{\prod_{k=(T-1)}^{T-1} b(A_k|S_k)}  + (1-\gamma)\sum_{h=(T-1)+1}^{T(t)-1} \gamma^{h-(T-1)-1} \frac{\prod_{k=(T-1)}^{h-1} \pi(A_k|S_k)}{\prod_{k=(T-1)}^{h-1} b(A_k|S_k)}
}\\
&= \frac{
    \frac{\pi(A_{T-1}|S_{T-1})}{b(A_{T-1}|S_{T-1})} \Bar{G}_{T-1:T}
}{
    \frac{\pi(A_{T-1}|S_{T-1})}{b(A_{T-1}|S_{T-1})}
}\\
&= \Bar{G}_{T-1:T}
\intertext{$t = T - 2$:}
&\frac{
    \gamma \frac{\pi(A_{T-1}|S_{T-1}) \pi(A_{T-2}|S_{T-2})}{b(A_{T-1}|S_{T-1}) b(A_{T-2}|S_{T-2})} \Bar{G}_{T-2:T} 
    + 
    (1-\gamma)
    \frac{\pi(A_{T-2}|S_{T-2})}{b(A_{T-2}|S_{T-2})}  
    \Bar{G}_{T-2:T-1}
}{
    \gamma \frac{\pi(A_{T-1}|S_{T-1}) \pi(A_{T-2}|S_{T-2})}{b(A_{T-1}|S_{T-1}) b(A_{T-2}|S_{T-2})} 
    + 
    (1-\gamma)
    \frac{\pi(A_{T-2}|S_{T-2})}{b(A_{T-2}|S_{T-2})}
}\\
&= \frac{
    \gamma \frac{\pi(A_{T-1}|S_{T-1})}{b(A_{T-1}|S_{T-1})} \Bar{G}_{T-2:T} + (1-\gamma)\Bar{G}_{T-2:T-1}
}{
    \gamma \frac{\pi(A_{T-1}|S_{T-1})}{b(A_{T-1}|S_{T-1})} + (1-\gamma)
}
\intertext{$t = T - 3$:}
&\frac{
    \gamma^{T - (T-3) - 1} \frac{\prod_{k=T-3}^{T-1} \pi(A_k|S_k)}{\prod_{k=T-3}^{T-1} b(A_k|S_k)} 
    \Bar{G}_{T-3:T} + (1-\gamma)\sum_{h=(T-3)+1}^{T-1} \gamma^{h-(T-3)-1} \frac{\prod_{k=T-3}^{h-1} \pi(A_k|S_k)}{\prod_{k=T-3}^{h-1} b(A_k|S_k)}  \Bar{G}_{T-3:h}
}{
    \gamma^{T - (T-3) - 1} \frac{\prod_{k=T-3}^{T-1} \pi(A_k|S_k)}{\prod_{k=T-3}^{T-1} b(A_k|S_k)}  + (1-\gamma)\sum_{h=(T-3)+1}^{T-1} \gamma^{h-(T-3)-1} \frac{\prod_{k=T-3}^{h-1} \pi(A_k|S_k)}{\prod_{k=T-3}^{h-1} b(A_k|S_k)}
}\\\\
&=\frac{
    \gamma^{2} 
    \frac{\pi(A_{T-3}|S_{T-3})}{b(A_{T-3}|S_{T-3})}
    \frac{\pi(A_{T-2}|S_{T-2})}{b(A_{T-2}|S_{T-2})}
    \frac{\pi(A_{T-1}|S_{T-1})}{b(A_{T-1}|S_{T-1})}
    \Bar{G}_{T-3:T} + (1-\gamma)
    \big(
    \frac{\pi(A_{T-3}|S_{T-3})}{b(A_{T-3}|S_{T-3})}
    \Bar{G}_{T-3:T-2}
    + 
    \gamma 
    \frac{\pi(A_{T-3}|S_{T-3})}{b(A_{T-3}|S_{T-3})}
    \frac{\pi(A_{T-2}|S_{T-2})}{b(A_{T-2}|S_{T-2})} \Bar{G}_{T-3:T-1}
    \big)
}{
    \gamma^{2} 
    \frac{\pi(A_{T-3}|S_{T-3})}{b(A_{T-3}|S_{T-3})}
    \frac{\pi(A_{T-2}|S_{T-2})}{b(A_{T-2}|S_{T-2})}
    \frac{\pi(A_{T-1}|S_{T-1})}{b(A_{T-1}|S_{T-1})}
    + (1-\gamma)
    \big(
    \frac{\pi(A_{T-3}|S_{T-3})}{b(A_{T-3}|S_{T-3})}
    + 
    \gamma 
    \frac{\pi(A_{T-3}|S_{T-3})}{b(A_{T-3}|S_{T-3})}
    \frac{\pi(A_{T-2}|S_{T-2})}{b(A_{T-2}|S_{T-2})}
    \big)
}\\\\
&=\frac{
    \gamma^{2} 
    \frac{\pi(A_{T-2}|S_{T-2})}{b(A_{T-2}|S_{T-2})}
    \frac{\pi(A_{T-1}|S_{T-1})}{b(A_{T-1}|S_{T-1})}
    \Bar{G}_{T-3:T} + (1-\gamma)
    \big(
    \Bar{G}_{T-3:T-2}
    + 
    \gamma 
    \frac{\pi(A_{T-2}|S_{T-2})}{b(A_{T-2}|S_{T-2})} \Bar{G}_{T-3:T-1}
    \big)
}{
    \gamma^{2} 
    \frac{\pi(A_{T-2}|S_{T-2})}{b(A_{T-2}|S_{T-2})}
    \frac{\pi(A_{T-1}|S_{T-1})}{b(A_{T-1}|S_{T-1})}
    + 
    (1-\gamma)
    \big(
    1
    + 
    \gamma 
    \frac{\pi(A_{T-2}|S_{T-2})}{b(A_{T-2}|S_{T-2})}
    \big)
}\\
\end{align*}

\newpage
$t = T - 1$:
\begin{align*}
& R_{T}
\intertext{$t = T - 2$:}
& \frac{
    \gamma \frac{\pi(A_{T-1}|S_{T-1})}{b(A_{T-1}|S_{T-1})} (R_{T} + R_{T-1}) + (1-\gamma)(R_{T-1})
}{
    \gamma \frac{\pi(A_{T-1}|S_{T-1})}{b(A_{T-1}|S_{T-1})} + (1-\gamma)
}
\intertext{$t = T - 3$:}
&\frac{
    \gamma^{2} 
    \frac{\pi(A_{T-2}|S_{T-2})}{b(A_{T-2}|S_{T-2})}
    \frac{\pi(A_{T-1}|S_{T-1})}{b(A_{T-1}|S_{T-1})}
    (R_{T-2} + R_{T-1} + R_{T})
    + 
    (1-\gamma)
    \big(
    R_{T-2}
    + 
    \gamma 
    \frac{\pi(A_{T-2}|S_{T-2})}{b(A_{T-2}|S_{T-2})} (R_{T-2} + R_{T-1}
    \big)
}{
    \gamma^{2} 
    \frac{\pi(A_{T-2}|S_{T-2})}{b(A_{T-2}|S_{T-2})}
    \frac{\pi(A_{T-1}|S_{T-1})}{b(A_{T-1}|S_{T-1})}
    + 
    (1-\gamma)
    \big(
    1
    + 
    \gamma 
    \frac{\pi(A_{T-2}|S_{T-2})}{b(A_{T-2}|S_{T-2})}
    \big)
}\\
\end{align*}


\textbf{Per Decision Importance Sampling}

Here we try to reduce variance even in the absence of discounting. In the off-policy estimators, each term in the numerator is a sum.

\begin{align*}
    \rho_{t:T-1}G_t &= \rho_{t:T-1}(R_{t+1} + \gamma R_{t+2} + \cdots + \gamma^{T-t-1}R_T)\\
    &= \rho_{t:T-1} R_{t+1} + \gamma\rho_{t:T-1} R_{t+2} + \cdots + \gamma^{T-t-1}\rho_{t:T-1}R_T
\end{align*}
Each of these can be written as:
$$\rho_{t:T-1}R_{t+1} = 
\frac{\pi(A_{t}|S_{t})}{b(A_{t}|S_{t})}
\frac{\pi(A_{t+1}|S_{t+1})}{b(A_{t+1}|S_{t+1})}
\frac{\pi(A_{t+2}|S_{t+2})}{b(A_{t+2}|S_{t+2})}
\cdots
\frac{\pi(A_{T-1}|S_{T-1})}{b(A_{T-1}|S_{T-1})}
R_{t+1}
$$

Since all the terms aside from the first one occur after the reward $R_{t+1}$. And the expected value of these factors is 1:

$$\mathbb{E}\bigg[ \frac{\pi(A_{k}|S_{k})}{b(A_{k}|S_{k})}\bigg] = \sum_a b(a|S_k)\frac{\pi(a|S_{k})}{b(a|S_{k})} = \sum_a \pi (a|S_k) = 1$$

All these factors have no effect in expectation, i.e.:

$$\mathbb{E}[\rho_{t:T-1} R_{t+1}] = \mathbb{E}[\rho_{t:t} R_{t+1}]$$

We can repeat this for each of the $k$-th terms

$$\mathbb{E}[\rho_{t:T-1} R_{t+k}] = \mathbb{E}[\rho_{t:t+k-1} R_{t+k}]$$

Then our original term can be written as:
$$\mathbb{E}[\rho_{t:T-1}G_t] = \mathbb{E}[\Tilde{G}_t]$$

Where 
$$ \Tilde{G}_t = \rho_{t:t}R_{t+1} + \gamma\rho_{t:t+1}R_{t+2} + \gamma^2\rho_{t:t+2}R_{t+3} +
\cdots +
\gamma^{T-t-1}\rho_{t:T-1}R_{T}
$$

So with this, we get an ordinary importance sampling estimator with $\Tilde{G}_t$:

$$V(s) = \frac{\sum_t\in\mathcal{T}(s)\Tilde{G}_t}{|\mathcal{T}(s)|}$$

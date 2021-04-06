---
layout: post
title: 'CSC412 Communication Project'
date: 2021-04-05 13:00:00 +0500
categories: Probability Learning
comments: false
summary: ''
---

## An Overview of Importance and Rejection Sampling
This blog post was written 

## The Importance of Importance Sampling

Useful when $p$ is a complicated distribution without a simple way to sample from.

Enables us to learn something about a target distribution $p$ without ever needing to sample from $p$.


<!-- Let's use another distribution q(x) = q'(x) / Z instead
Simpler distribution q', ie. Gaussian
Must be easy to sample and to compute density
Support of q must also be the support of p

Importance Sampling
Areas where q > p and others where q < p
Overrepresented and underrepresented areas, good idea to resample with importance weighting:
For sample x_r ~ q, importance weights w'_r = p'(x_r) / q'(x_r), and if we have access to normalized weights, we have: w_r = p(x_r) / q(x_r)

int f(x) p(x) dx = int f(x) p(x) q(x) / q(x) dx
= int f(x) w(x) q(x) dx
= E_{x ~ q} [f(x) w(x)]
So we only need to sample from q, which we know how to do already
This is approximately equal to 
1/N sum f(x_r) w_r
where x_r ~ q.
This is our importance weighted estimator

Lets consider the unormalized ones
p = p' / Z_p, q = q' / Z_q
w'_r = p' / q'
w_r = (p' / Z_p) / (q' / Z_q)
int f(x) w(x) q(x) dx = Z_q/Z_p int f(x) w'(x) q(x) dx

Z_p / Z_q approx 1/R sum_r w'(x_r)

Issue: if p << q, then w -> 0, and if q << p, then w -> inf
Numerical issues, especially in high dimensions

Rejection Sampling
Consider cq', where c is a scalar multiple s.t. cq' > p over the entire support
This approximates a bounding box
1. Sample x ~ q, the proposal
2. Sample u ~ Uniform[0, cq'(x)]
3. if p'(x) < u, reject
else, accept
Therefore, x ~ p
Works only if p approx q, which becomes difficult in higher dimensions, namely since small differences in max heights in small dimensions grow exponentially as dimension increases, i.e. our c -> inf, and the ratio of area under cq' and p is 1/c, so very unlikely to accept

Metropolis idea
Local proposals based on the current sample
Let's say we start with sample x^0, can evaluate p'(x^0)
Proposal: x^{t+1} ~ q(x | x^t)
Correction: a = p'(x^{t+1}) / p'(x^t)

example
q = N(x | mean = x^{t})

If a < 1, then proposal is less likely
If a > 1, then proposal is better, so we go there

Accept bad samples sometimes, in order to not be stuck
accept x^{t+1} with probability a

Metropolis Hastings adds on the usage of other distributions, i.e. nonsymmetric distributions
So we modify our correction term
a = p'(x^{t+1}) q(x^t | x^{t+1}) / p'(x^t) q(x^{t+1} | x^t)
So this accounts for the lack of symmetry in the acceptance step

This sampling is no longer iid, we will fix this later with MC
Finding a proper MCMC proposal distribution to traverse this is difficult

Markov Chain generates a chain of points inside typical set of p
{x^0, x^1, x^2, ..., x^N} ~ p
Done by specifying "Markov Transitions" want to generate x^{t+1} ~ T(x^{t+1} | x^t)
But x^{t+1} is not independent of x^t

p(x) is invariant under T, ie. p(x) = int T(x|x') p(x') dx'
If x' ~ p and x ~ T(x|x'), then E_{x' ~ p} [T(x|x')] = p(x), i.e. x ~ p
Basically, once we get in there, we stay there

p(x) is Ergodic if p^(t) (x) -> p(x) for all p^0(x)
i.e. the distribution approaches p(x) no matter the starting distribution p^0(x)

If both these are true, then asymptotically (t -> inf), x ~ p iid

Can slide around this typical set using gradients of the likelihood function, and then go perpendicular to it, i.e. the vector field made by the normals, so we turn it into a problem from a postion, into a problem related to the velocity -->

## A Basic Example of Importance Sampling
Let's suppose that we're working in a 1 dimensional space and we want to find the average over our very simple target distribution $p = Uniform[1, 6]$. We know from elementary statistics that:

$$\mathbb{E}_{x \sim p} [x] = \frac{1 + 6}{2} = 3.5$$

In addition, almost every mathematical package out there can sample from a Uniform distribution. But for the purposes of demonstration, let's assume that we can only sample from a 1-D normal distribution defined as $q = \mathcal{N}( \mu_q, \sigma_q^2 )$. We will utilize samples $x \sim q$ and importance sampling to approximate the value: $\mathbb{E}_{x \sim p} [x]$ (which we know should be $3.5$).

{% include animation-1.html %}


## Why We Shouldn't Reject Rejection Sampling


## Monte Carlo
<!-- For most situations in the real world, it is difficult to calculate the transition function for an environment as we did in previous methods. Monte Carlo is a method of learning and approximating value functions that only requires experience, e.g. an agent running around an environment until it hits a goal. It is based on averaging sample discounted returns and is only defined for episodic tasks, where all episodes eventually terminate no matter what actions are chosen. This means that the estimates for each state is independent and doesn't depend on the estimates of other states.  -->

<!-- 

## First-Visit and Every-Visit

To estimate values for State Value functions, we average the returns observed after visits to each state. There are 2 different ways to do it:
- The first-visit MC method averages returns of the first visit to each state in an episode.
- The every-visit MC method averages returns of all visits to state.

Both converge to the optimal state value function with infinite experience.

Blackjack is a good example for when it is hard to use DP (the probability of the next cards can be difficult to calculate), if you aren’t familiar with blackjack it uses these <a href="https://bicyclecards.com/how-to-play/blackjack/" target="_blank">rules</a>. A Blackjack environment has already been coded in <a href="https://gym.openai.com/" target="_blank">OpenAI's Gym</a>:

```python
import numpy as np
import gym
env = gym.make('Blackjack-v0')

# Only stick if our hand is 20 or 21
policy = np.ones((32, 11, 2))
policy[20:22, :, :] = 0
policy[9:11, :, 1] = 0

V_s = np.zeros((32, 11, 2))
ret = np.zeros((32, 11, 2))
count = np.zeros((32, 11, 2))

DISCOUNT = 1
```

We will evaluate the above policy with First Visit MC, iterating over several hundreds of thousands of trials in order to reduce the effects of randomness and hopefully reach every state.

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

Here are the approximated State Value functions (x-axis is what card the dealer is showing and y-axis is the sum of our hand), we can see that the with an Ace is not as smooth as the one without since we encounter states without Ace much more often:

{% include image-2.html 
  url-1="/assets/images/3-mc/first_no_ace.png" 
  des-1="No Usable Ace" 
  url-2="/assets/images/3-mc/first_ace.png" 
  des-2="Usable Ace" 
%}

## Exploring Starts

If we do not have a model of the environment, we need approximations for state-action pairs, since creating a policy with state values requires the transition probabilities $p(s', r \mid  s,a)$. These actions values are also estimated by averaging sample discounted returns. The issue is that many state-action pairs might never be visited, and we need to estimate the value of all pairs to guarantee optimal convergence.

This is especially difficult if we have a deterministic policy, where only one action will be taken in each state. In other words, we need to ensure exploration of all actions for each state. One way is to only consider policies that are stochastic with nonzero probability for all state-action pairs. Another method is 'exploring starts', where every state-action pair has a non zero probability of being selected as the starting state of an episode. 

Here's the accompanying code:

```python
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
```

Which gives us:

{% include image-4.html 
  url-1="/assets/images/3-mc/es_no_ace_hit.png" 
  des-1="Value of hitting with no usable ace." 
  url-2="/assets/images/3-mc/es_ace_hit.png" 
  des-2="Value of hitting with a usable ace." 
  url-3="/assets/images/3-mc/es_no_ace_stick.png" 
  des-3="Value of sticking with no usable ace." 
  url-4="/assets/images/3-mc/es_ace_stick.png" 
  des-4="Value of sticking with no usable ace." 
%}

## Policy Making

In Monte Carlo methods, we still want to follow the idea behind Generalized Policy Iteration, where we maintain approximate value and policy functions, and optimize them episode by episode. After each episode, the observed returns are used to update the visited states, and then the policy is improved for all these states.

There are 2 methods to go about approximating the optimal policy:
- On-policy means to evaluate or improve the policy used to generate the data. 
- Off-policy methods evaluate or improve a different policy than the data generating one. 

In on-policy, the policies are generally 'soft', as in the policy always has a non zero probability of selecting every action in every state, similar to the epsilon-greedy policies. We still use first visit MC methods to estimate action values. 

Here is the code for on-policy control (learning the policy) on the Blackjack environment:

```python
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
```

Here are the resulting value functions:

{% include image-4.html 
  url-1="/assets/images/3-mc/on_no_ace_hit.png" 
  des-1="Value of hitting with no usable ace." 
  url-2="/assets/images/3-mc/on_ace_hit.png" 
  des-2="Value of hitting with usable ace." 
  url-3="/assets/images/3-mc/on_no_ace_stick.png" 
  des-3="Value of sticking with no usable ace." 
  url-4="/assets/images/3-mc/on_ace_stick.png" 
  des-4="Value of sticking with usable ace." 
%}

In off-policy, we consider 2 policies: a target policy $\pi$ that learns from the experience and becomes the optimal policy, and a behaviour policy $b$ which is more exploratory and generates the experience. Comparatively, on-policy is simpler and converges faster, but off-policy is more powerful and general (on-policy is a special case of off-policy when both target and behaviour policies are the same). 

However, there is a problem when both $\pi$ and $b$ are fixed and we want to estimate $v_\pi$ or $q_\pi$. This is that the episodes follow $b$, while we try to estimate values for $\pi$, so we need to find a way to relate them to each other. To accomplish this, we require every action taken under $\pi$ also be taken in $b$, i.e. $\pi(a\mid s) > 0 \Rightarrow b(a\mid s) > 0$. This is also known as the assumption of coverage, it follows that $b$ must be stochastic where it is not identical to $\pi$, but $\pi$ can be a deterministic policy. 

We approach this by utilizing Importance sampling, a technique for estimating expected values from one distribution given samples from another. This is applied to off-policy learning by weighting the sample returns according to the relative probability of their trajectories occurring under $\pi$ and $b$, called the importance-sampling ratio. The probability of the state-action trajectory $A_t, S_{t+1}, A_{t+1}, ..., S_T$ occurring under $\pi$ is:

$$\pi(A_t\mid S_t)p(S_{t+1}\mid S_t, A_t)\pi(A_{t+1}\mid S_{t+1}) ... p(S_T\mid S_{T-1}, A_{T-1}) = \prod_{k=t}^{T-1} \pi(A_k\mid S_k)p(S_{k+1}\mid S_k, A_k)$$

This still requires a model of the environment ($p(S_{t+1}\mid S_t, A_t)$), but if we consider the relative probability, we get:

$$\rho_{t:T-1} = \frac{\prod_{k=t}^{T-1} \pi(A_k\mid S_k)p(S_{k+1}\mid S_k, A_k)}{\prod_{k=t}^{T-1} b(A_k\mid S_k)p(S_{k+1}\mid S_k, A_k)} = \frac{\prod_{k=t}^{T-1} \pi(A_k\mid S_k)}{\prod_{k=t}^{T-1} b(A_k\mid S_k)} $$

So now we weight all our returns in $b$ with $\rho_{t:T-1}$ to transform it to returns for $\pi$. In doing so, let time continue to count up from episode to episode and we define:

- $\mathcal{T}(s)$: set of either all time steps when $s$ was visited (every visit method), or the first time steps of each episode when $s$ was visited (first visit method).
- $T(t)$: first episode termination after time $t$.
- $G_t$: return after time $t$ up until $T(t)$.

So we get:
- $\\{G_t\\}_{t \in \mathcal{T}(s)}$, the returns that matter to state $s$.
- $\\{\rho_{t:T-1}\\}_{t\in \mathcal{T}(s)}$, the corresponding importance-sampling ratios. 

There are 2 ways of sampling each with their upsides and downsides:
- Ordinary importance sampling:
$$V(s)=\frac{\sum_{t\in\mathcal{T}(s)}\rho_{t:T-1}G_t}{|\mathcal{T}(s)|}$$

- Weighted importance sampling:
$$V(s)=\frac{\sum_{t\in\mathcal{T}(s)}\rho_{t:T-1}G_t}{\sum_{t\in\mathcal{T}(s)}\rho_{t:T-1}}$$

On one hand, Weighted importance sampling has a variance between ratios of at most 1, whereas Ordinary importance sampling has unbounded variance. On the other hand, the ordinary one is unbiased where the weighted one is. But with the assumption of bounded returns (which we can assume in most cases), the variance of Weighted importance sampling converges to 0, so in general Weighted importance sampling is better.

So our update for states with importance sampling weights $W_i$ and returns $G_i$ is:

$$v_{n}(s) = \frac{\sum_{i=1}^n W_i G_i}{\sum_{i=1}^n W_i}$$

Which can be constructed incrementally by keeping count $C_i$ and using:

$$
\begin{align*}
v_{n}(s) &= v_{n-1}(s) + \frac{W_{n-1}}{C_{n-1}} [G_{n-1} - v_{n-1}(s)]\\
C_n &= C_{n-1} + W_n
\end{align*}
$$

```python
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
```

{% include image-4.html 
  url-1="/assets/images/3-mc/off_no_ace_hit.png" 
  des-1="Value of hitting with no usable ace." 
  url-2="/assets/images/3-mc/off_ace_hit.png" 
  des-2="Value of hitting with usable ace." 
  url-3="/assets/images/3-mc/off_no_ace_stick.png" 
  des-3="Value of sticking with no usable ace." 
  url-4="/assets/images/3-mc/off_ace_stick.png" 
  des-4="Value of sticking with usable ace." 
%}

For now that is all we need to know about Monte Carlo methods, but we will go much more in depth with Importance Sampling and its variants in the future.

[blackjack]: https://bicyclecards.com/how-to-play/blackjack/ -->
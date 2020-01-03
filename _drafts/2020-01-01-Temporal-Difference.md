---
layout: post
title: 'Basics of Banditry 4: Temporal Difference'
date: 2020-01-01 13:00:00 +0500
categories: RL Python
comments: true
summary: ''
---

\textbf{TD Learning}

Temporal Difference Learning is a combination of monte carlo ideas and dynamic programming. They learn directly from raw experience without a model of the environment (Monte Carlo), and update estimates based on other estimates (DP). 

We call the vanilla Monte Carlo environment with a constant-$\alpha$ MC:
$$V(S_t) = V(S_t) + \alpha(G_t - V(S_t))$$

MC methods need to complete the episode prior to updating the state values, whereas TD methods only require waiting until the next time step. The simplest TD method, TD(0), updates with:
$$V(S_t) = V(S_t) + \alpha[R_{t+1} + \gamma V(S_{t+1}) - V(S_t)]$$

TD and MC methods are both sample based methods as they learn from a singular experience rather than getting an expected value for all experiences like in DP. 

Also, the quantity in the brackets of TD(0) function is an error for the current estimated value $V(S_t)$ and the better estimate of $R_{t+1} + \gamma V(S_{t+1})$. We call this the TD error:

$$\delta_t = R_{t+1} + \gamma V(S_{t+1}) - V(S_{t})$$

The TD error is not available until the next state and reward, i.e. time $t+1$. So the if value function does not change during the episode, we can write the MC error as:
\begin{align*}
G_t - V(S_t) &= R_{t+1} + \gamma G_{t+1} - V(S_t) + \gamma V(S_{t+1}) - \gamma V(S_{t+1}) \\
&= \delta_t + \gamma(G_{t+1} - V(S_{t+1})) \\
&= \delta_t + \gamma \delta_{t+1} + \gamma^2(G_{t+2} - V(S_{t+2})) \\
&= \delta_t + \gamma \delta_{t+1} + \cdots + \gamma^{T-t-1}\delta_{T-1} + \gamma^{T-t}(G_{T} - V(S_{T})) \\
&= \delta_t + \gamma \delta_{t+1} + \cdots + \gamma^{T-t-1}\delta_{T-1} + \gamma^{T-t}(0 - 0) \\
&= \sum^{T-1}_{k=t} \gamma^{k-t} \delta_k
\end{align*}

This is not exact if $V$ is updated during the episode (as in TD(0)), but if the step size is small enough then it holds approximately. 

Let's derive the error that changing the value function during the episode would incur:
\begin{align*}
G_t - V(S_t) &= R_{t+1} + \gamma G_{t+1} - V(S_t) + \gamma (V(S_{t+1}) - \alpha \delta_t) - \gamma (V(S_{t+1}) - \alpha \delta_t) \\
&= (1 + \alpha \gamma) \delta_t + \gamma(G_{t+1} - V(S_{t+1})) \\
&= (1 + \alpha \gamma) \delta_t + \gamma (1 + \alpha \gamma) \delta_{t+1} + \gamma^2(G_{t+2} - V(S_{t+2})) \\
&= (1 + \alpha \gamma) \delta_t + \gamma (1 + \alpha \gamma)\delta_{t+1} + \cdots + \gamma^{T-t-1} (1 + \alpha \gamma) \delta_{T-1} + \gamma^{T-t}(G_{T} - V(S_{T})) \\
&= (1 + \alpha \gamma) \delta_t + \gamma (1 + \alpha \gamma) \delta_{t+1} + \cdots + \gamma^{T-t-1} (1 + \alpha \gamma) \delta_{T-1} + \gamma^{T-t}(0 - 0) \\
&= (1 + \alpha \gamma)\sum^{T-1}_{k=t} \gamma^{k-t} \delta_k
\end{align*}

The advantage TD has over DP: does not require complete model of environment with all rewards and next-state probability distributions. For MC: don't have to wait to the end of an episode to update, very useful in long episode situations. 

TD(0) is also guaranteed to converge if alpha is small enough or decreases at the usual stochastic approximation conditions.

Given a finite amount of experience, a common approach is to present the experience repeatedly until it converges. We use batch updating (calculating the TD updates at each time step but only changing V at the end of the episode) in order to converge deterministically independent of $\alpha$.

\textbf{On-Policy TD}

Now we need to calculate the action values:
$$ Q(S_t, A_t) = Q(S_t, A_t) + \alpha [R_{t+1} + \gamma Q(S_{t+1}, A_{t+1}) - Q(S_t, A_t)]$$

We perform this every transition that is non-terminal, if the next state is terminal, then the action value is 0. The term "Sarsa" is also used for this algorithm as all of $(S_t, A_t, R_{t+1}, S_{t+1}, A_{t+1})$ is used. 

As with all on-policy methods,we keep estimating $q_\pi$ for our policy $\pi$, while also changing $\pi$ to be greedy according to the state values.

Let's derive a version of MC error in terms of action values:
\begin{align*}
G_t - V(S_t) &= G_t - Q(S_t, A_t)\\
&= R_{t+1} + \gamma G_{t+1} - Q(S_t, A_t) + \gamma Q(S_{t+1}, A_{t+1}) - \gamma Q(S_{t+1}, A_{t+1}) \\
&= \delta_t + \gamma(G_{t+1} - Q(S_{t+1}, A_{t+1})) \\
&= \delta_t + \gamma \delta_{t+1} + \gamma^2(G_{t+2} - Q(S_{t+2}, A_{t+2})) \\
&= \delta_t + \gamma \delta_{t+1} + \cdots + \gamma^{T-t-1}\delta_{T-1} + \gamma^{T-t}(G_{T} - Q(S_{T}, A_{T})) \\
&= \delta_t + \gamma \delta_{t+1} + \cdots + \gamma^{T-t-1}\delta_{T-1} + \gamma^{T-t}(0 - 0) \\
&= \sum^{T-1}_{k=t} \gamma^{k-t} \delta_k
\end{align*}

\textbf{Q-learning}
One of the big breakthroughs of RL is the off policy TD control algorithm known as Q-Learning:
$$Q(S_t, A_t) = Q(S_t, A_T) + \alpha [R_{t+1} + \gamma \max_a Q(S_{t+1}, a) - Q(S_t, A_T)]$$

Where the action function $Q$ is approximating $q_*$ the optimal action value function.

\textbf{Expected Sarsa}

Just like Q-learning but it uses the expected value of the next action value pairs, rather than the maximum:
$$Q(S_t, A_t) = Q(S_t, A_T) + \alpha [R_{t+1} + \gamma \sum_a \pi(a|S_{t+1}) Q(S_{t+1}, a) - Q(S_t, A_T)]$$


\textbf{Maximization Bias}

Maximization Bias is a problem in control algorithms where the maximum of the true values is zero, but the maximum of the estimates are positives, and since our algorithms are usually greedy with respect to these estimates, this causes a positive bias. One way to solve this is to divide the experience into 2 sets and use them to learn 2 independent estimates. One estimate, $Q_1(a)$, is used to determine the maximizing action $A^* = argmax_a Q_1(a)$ while the other determines the of its value $Q_2(A^*) = Q_2(argmax_a Q_1(a))$. This is unbiased since $\mathbb{E} [Q_2(A^*)] = q(A^*)$. We can repeat this again with reversed estimates to yield another unbiased estimate. This is called Double Learning.

The update in the first case is:
$$Q_1(S_t, A_t) = Q_1(S_t, A_t) + \alpha[R_{t+1} + \gamma Q_2(S_{t+1}, argmax_a Q_1(S_{t+1}, a)) - Q_1(S_t, A_t)]$$
and with $Q_1, Q_2$ reversed in the second case.


\textbf{Afterstates and Special Cases}

In situations where we know the initial environmental dynamics, we can use an \textit{afterstate value function}, where the afterstate values refer to the values of the states after the agent makes a move, e.g. in games when we usually know the immediate results of our moves. This allows any moves from any positions that lead to the same state to update learning in every instance, rather than learning individual values for each state action pair.



\textbf{$n$-step Bootstrapping}

$n$-step TD methods combine both MC and TD methods so that we can smoothly shift in between each as needed. Additionally, they allow for the time steps between updates to be variable, somewhere between the one step and one episode of TD(0) and MC methods, respectively.

Consider the MC estimate of $V(s)$ as the target of the update (i.e. the most ideal and realistic update):

$$G_t = R_{t+1} + \gamma R_{t+2} + \gamma^2 R_{t+3} + \cdots + \gamma^{T-t-1}R_T$$

In TD(0) methods, since we truncate the actual reward at time 1 and we call it a one-step return:

$$G_{t:t+1} = R_{t+1} + \gamma V_t(S_{t+1})$$

And similarly for the $n$-step return:

$$G_{t:t+n} = R_{t+1} + \gamma R_{t+2} + \cdots + \gamma^{n-1} R_{t+n} + \gamma^n V_{t+n-1}(S_{t+n})$$

We cannot update until we have seen $R_{t+n}$ and $V_{t+n-1}$. So now the updates are:

$$V_{t+n}(S_t) = V_{t+n-1}(S_t) + \alpha[G_{t:t+n} - V_{t+n-1}(S_t)]$$

The algorithm only starts making changes at the $n$-th time step. Once we get to the end, we continue for another $n$ steps with the missing terms as 0. 

$n$-step returns expectation is guaranteed to be a better estimate than $v_\pi$ than $V_{t+n-1}$. So our worst error is less than or equal to $\gamma ^n$ times the worst error under $V_{t+n-1}$:

$$\max\limits_s \left| \mathcal{E}_\pi [G_{t:t+n} | S_t = s] - v_\pi(s) \right| \leq \gamma^n \max_s \left|V_{t+n-1} - v_\pi(s) \right|$$

This is the \textbf{error reduction property} of $n$-step. Making it potentially better than either of the extremes (MC and TD).\\

\textbf{$n$-step Sarsa}\\
In this on-policy control method, we will calculate State-Action pairs and then use a $\epsilon$-greedy policy for action choosing. We end and start with a action, so the updates are:
$$G_{t:t+n} = R_{t+1} + \gamma R_{t+2} + \cdots + \gamma^{n-1}R_{t+n} + \gamma^n Q_{t+n-1}(S_{t+n}, A_{t+n}), n \geq 1, 0 \leq t < T-n$$
Where $G_{t:t+n} = G_t$ if $t+n \geq T$. So then our update algorithm is: 
$$Q_{t+n}(S_t, A_t) = Q_{t+n-1}(S_t, A_t) + \alpha [G_{t:t+n} - Q_{t+n-1}(S_t,A_t)]$$
And the values of all other states are unchanged: $Q_{t+n}(s,a)=Q_{t+n-1}(s,a)$ for all $s,a$ such that $s\neq S_t$ or $a\neq A_t$. \\

\textbf{Off Policy Learning}\\
Off policy learning is when we learn the value function for one policy $\pi$ while following another $b$. Usually $\pi$ exploits and $b$ explores. We must compute the relative probabilities of taking actions to use the data from policy $b$ to learn in $\pi$. Since $n$-step uses returns over $n$ steps, so we need the total relative probabilities of those $n$ steps. We update with:

\begin{align*}
V_{t + n}(S_t) &= V_{t+n-1}(S_t) + \alpha p_{t:t+n-1} [G_{t;t+n} - V_{t+n-1}(S_t)] \\
Q_{t + n}(S_t, A_t) &= Q_{t+n-1}(S_t, A_t) + \alpha p_{t+1:t+n} [G_{t;t+n} - V_{t+n-1}(S_t)] \\
p_{t:h} &= \prod_{k=t}^{\min(h, T-1)} \frac{\pi(A_k | S_k)}{b(A_k | S_k)}
\end{align*}

If any of the $n$ actions have no chance of being taken by $\pi$, then the entire return is zeroed out and ignored. On the other hand, if $\pi$ is much more likely to take the action, then it would increase the weight of the return. We never run into issues with dividing by 0 as we follow policy $b$ so every action we take has a nonzero chance in $b$. 
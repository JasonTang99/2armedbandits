---
layout: post
title: 'Basics of Banditry 4: Temporal Difference'
date: 2020-01-01 13:00:00 +0500
categories: RL Python
comments: true
summary: ''
---


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



Discounting Aware Importance Sampling: for reducing the variance of off-policy estimators
In ordinary importance sampling, if episodes are long, then the multiplying y all the other samples adds a significant amount of variance to first sample, which only needs to be scaled by the first one.
Think of discounting as determining a probability of termination (a degree of partial termination). I.e. the return terminates with degree 1-gamma after 1 step, and (1-gamma)gamma^(n-1) after n steps.
\bar G_{t:h} = R_{t+1} + R_{t+2} + \cdots + R_h
G_t = R_{t+1} + \gamma R_{t+2} + \gamma^2 R_{t+3} + \cdots \gamma^{T-t-1} R_T = (1-\gamma)R_{t+1} + (1-\gamma)\gamma (R_{t+1} + R_{t+2}) + \cdots + (1-\gamma)\gamma^{T-t-1} (R_{t+1} + R_{t+2} + \cdots + R_T) = (1-\gamma) \sum_{h=t+1}^{T-1} \gamma^{h-t-1} \bar G_{t:h} + \gamma^{T-t-1} \bar G{t:T}
We can then scale this with importance sampling as usual. This accounts for discounting and does nothing for gamma=1

Per-Decision Importance Sampling
Off policy estimations rely on:
\rho_{t:T-1} G_t = \rho_{t:T-1} R_{t+1} + \rho_{t:T-1} R_{t+2} + \cdots
But it doesn't make sense for those importance sampling \frac{\pi(A_t|S_t)}{b(A_t|S_t)} to affect those rewards before them. It can be shown that there is no affect on the expected reward: \mathbb{E}[\rho_{t:T-1} R_{t+k} = \mathbb{E}[\rho_{t:t+k-1}R_{t+k}]
So we rewrite the reward as: \tilde G_t = \rho_{t:t}R_{t+1} + \gamma \rho_{t:t+1} R_{t+2} + \gamma^2 \rho_{t:t+2} R_{t+3} + \cdots + \gamma^{T-t-1} \rho_{t:T-1} R_T
This only works with ordinary importance sampling, not weighted as weighted generally does not converge to true value given infinite data.


Per Decision Methods with Control Variates
Similar to that in Monte Carlo
G_{t:h} = R_{t+1} + \gammaG_{t+1:h}
G{h:h} = V_{h-1}(S_h)
G_{t:h} = \rho_t (R_{t+1} + \gamma G_{t+1:h}) + (1-\rho_t) V_{h-1}(S_t)
This makes it so that if \rho_t goes to 0, then the target is the original estimate and not 0 (which would shrink the estimate.
The second term is called a control variate, which doesnt change the expected update. The importance sampling ratio has expected value 1 and is uncorrelated to the estimate so expected value of control variate is zero.
For action values, the first action must be taken so it has unit weight given to the state and reward that follow it. Importance sampling can only be applied to actions that follow it.
G_{t:h} = R_{t+1} + \gamma(\rho_{t+1} G_{t+1:h} + \bar V_{h-1} (S_{t+1}) - \rho_{t+1}Q_{h-1}(S_{t+1}, A_{t+1}))
G_{h:h} = Q_{h-1} (S_h, A_h)
if h >= T, G_{T-1:h} = R_T


Off-policy without Import Sampling: n-Step Tree Backup Algorithm
Take action leads to next state, of the possible actions to take we only take one due and bootstrap the rest with their estimates.The update lies in the estimated action values of leaf nodes in the tree, the internal nodes do not participate. Each leaf contributes weight equal to probability of occurence under \pi and scaled by step with discount \gamma. They are also multiplied by the weight of our chosen action in the previous tree level.
For n-step, defined as:
G_{t:t+n} = R_{t+1} + \gamma \sum_{a \neq A_{t+1}} \pi(a|S_{t+1}) Q_{t+n-1}(S_{t+1}, a) + \gamma \pi(A_{t+1} | S_{t+1}) G_{t+1: t+n}
For when t < T-1 and n>=2, n=1 handled by G_{t:t+1} = R_{t+1} + \gamma \sum_a \pi(a|S_{t+1}) Q_t(S_{t+1}, a).


n-step Q(\sigma)
Unifies all the others by giving parameter \sigma which if 1, it samples like in Sarsa, and otherwise doesn't sample like the tree-backup. Expected sarsa is sampling in all steps except the last one. We could also make each \sigma_t continuous in [0,1] and set it as a function of state and action at time t, thie is n-step Q(\sigma)
The tree backup n-step return is
G_{t:h} = R_{t+1} + \gamma \Sigma_{a\neq A_{t+1}} \pi(a|S_{t+1}) Q_{h-1}(S_{t+1},a) + \gamma \pi (A_{t+1}|S_{t+1}) G_{t+1:h} = R_{t+1} + \gamma \bar V_{h-1}(S_{t+1}) - \gamma\pi(A_{t+1}|S_{t+1})Q_{h-1}(S_{t+1}, A_{t+1}) + \gamma \pi(A_{t+1} | S_{t+1}) G_{t+1:h} = R_{t+1} + \gamma \pi (A_{t+1} | S_{t+1}) (G_{t+1:h} - Q_{h-1} (S_{t+1},A_{t+1})) + \gamma \bar V_{h-1}(S_{t+1})
This is the same for sarsa expect the \pi's are replaced with importance sampling ratios, so we slide between these with ratio \sigma
G_{t:h} = R_{t+1} + \gamma(\sigma_{t+1} \rho_{t+1} + (1-\sigma_{t+1}) \pi(A_{t+1} | S_{t+1}))(G_{t+1:h} - Q_{h-1}(S_{t+1}, A_{t+1})) + \gamma \bar V_{h-1}(S_{t+1})
Ending with G_{h:h} = Q_{h-1}(S_h, A_h) if h < T or G_{T-1:T} = R_T if h=T

Model Based methods: Dynamic Programming and Heuristic Search. Relies on planning.
Model Free methods: Monte Carlo, Temporal Difference, and n-step. Relies on learning
Both are based on learning a value function
Distribution models give all possibilities and associated probabilities. Like the MDP dynamics p(s', r|s,a). Much stronger but harder to create.
Sample models produce just one according to the probabilities. Easier to create but weaker than distribution.
We can use sample models to generate an episode of experience from a starting state and distribution models to generate all possible episodes and their probabilities.
Planning means to take the model as input into policy.
We can use simulated experience as well as real in learning (model free) methods
Indirect methods (involving learning a model of environment to plan from) make fuller use of limited experience and achieve better policy with low environmental interactions.
Direct methods are simpler and are not affected by the design of the model.
Dyna-Q includes all these processes continually, with planning as the random sample 1 step tabular Q-planning method, and the direct method being 1 step tabular Q-learning. Assume environment is deterministic(Once a model finds what s,a lead to, it never changes). Q-planning only queries from experienced states too.
Dynq-Q is just one version of the general Dyna agent.
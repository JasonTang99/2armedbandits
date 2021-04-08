---
layout: post
title: 'CSC412 Communication Project'
date: 2021-04-05 13:00:00 +0500
categories: Probability Learning
comments: false
summary: ''
---

This blog post was written for my CSC412 Computer Science Communication Project. We will be looking at a general overview of Importance/Rejection Sampling along with some simulations of situations where they might be useful. I will be summarizing information from both lectures and the textbook "Information Theory, Inference, and Learning Algorithms" by David Mackay and will be presenting it along with animations powered by the <a href="https://d3js.org/" target="_blank">D3</a> Library. To view the code used to run this blog, take a look at my <a href="https://github.com/2armedbandits/2armedbandits.github.io" target="_blank">Github Repo</a>.

## An Overview of Importance and Rejection Sampling

When working with complicated probability distributions $p$, whether it be purely theoretical or derived from real world applications, there are 2 common problems that arise:

1. Generating samples $x \sim p$. 
2. Computing expectations of functions $f(x)$ over our distribution $p$, i.e. computing 

$$\mathbb{E}_{x \sim p} [f(x)] = \int  f(x) p(x) dx$$

For sufficiently complex or high dimensional $p$, it becomes intractable to solve either of these problems exactly. Let's consider the situation in which we can evaluate the unnormalized $\widetilde{p\hspace{0.1em}}(x)$ for any given $x$, such that:

$$p(x) = \frac{\widetilde{p\hspace{0.1em}}(x)}{Z_p}$$

where we have the normalization constant $Z_p = \int \widetilde{p\hspace{0.1em}}(x) dx$. Computing this $Z_p$ quickly becomes difficult for complex $p$'s in high dimensions. But even with access to $Z_p$ (and therefore $p(x)$) it remains difficult to sample from $p$ since we need to evaluate $p(x)$ everywhere to know where we should sample from, i.e. the regions where $p(x)$ is large. 

However, if we are able to solve problem 1, we can compute an unbiased, arbitrarily close  approximation to problem 2 through a simple Monte Carlo estimator:

$$\mathbb{E}_{x \sim p} [f(x)] \approx \frac{1}{N} \sum_{i=1}^{N} f(x_i) = \hat{f}$$

where we generate $N$ samples $x_i \sim p$. Note that this estimator can get arbitrarily close to the actual expectation since its variance scales proportionally to $\frac{1}{N}$ (no dependence on dimensionality). Therefore, we can conclude that problem 1 is a harder problem than problem 2. For now, we will focus on methods of solving problem 2 without solving problem 1, which is where Importance and Rejection Sampling comes into play.

## The Importance of Importance Sampling

Since we cannot directly sample $x \sim p$ (problem 1), we will instead sample from some simpler distribution $q$. We will assume that we can easily generate samples $x \sim q$ and are able to compute an unnormalized $\widetilde{q\hspace{0.15em}}$, such that:

$$q(x) = \frac{\widetilde{q\hspace{0.15em}}(x)}{Z_q}$$

We then generate $N$ samples $\{x_i\}_{i=1}^N$ with $x_i \sim q$. Since these samples are drawn from $q$ rather than $p$, we need to adjust the weighting of each sample depending on how likely they are to be sampled by both $p$ and $q$. Namely, we compute an importance weight:

$$\widetilde{w}(x) = \frac{\widetilde{p\hspace{0.1em}}(x)}{\widetilde{q\hspace{0.15em}}(x)}$$

Assuming that $q$ and $p$ are different distributions, this adjusts the importance of samples in regions where $p$ is overrepresented by $q$ ($q > p$), and regions where $p$ is underrepresented ($p > q$). We also require that the support of $p$ is captured entirely by the support of $q$. In other words, for some point $x$, if $p(x) > 0$, then $q(x) > 0$ as well. This ensures that we have asymptotic convergence to the real mean $ \mathbb{E}_{x \sim p} [f(x)]$. Otherwise, there would exist regions where $p$ might sample from that our samples from $q$ would never visit. These restrictions make the normal distribution an ideal candidate for $q$, with support over all real numbers and being easy to sample from. It is especially useful in higher dimensions where it remains easy to sample from, unlike many other distributions.

Now we will derive how this importance weighting fits into our original goal of problem 2:

$$
\begin{align*}
\int  f(x) p(x) dx &= \int  f(x) p(x) \frac{q(x)}{q(x)} dx\\
&= \int  f(x) \frac{\widetilde{p\hspace{0.1em}}(x)}{Z_p} \frac{q(x) Z_q}{\widetilde{q\hspace{0.15em}}(x)} dx\\
&= \int  f(x) \frac{\widetilde{p\hspace{0.1em}}(x)}{\widetilde{q\hspace{0.15em}}(x)} \frac{ Z_q}{Z_p}q(x) dx\\
&=  \frac{ Z_q}{Z_p} \int  f(x) \widetilde{w}(x) q(x) dx\\
&= \frac{ Z_q}{Z_p} \mathbb{E}_{x \sim q} [f(x) \widetilde{w}(x)]\\
&\approx \frac{ Z_q}{Z_p} \frac{1}{N} \sum_{i=1}^{N} f(x_i) \widetilde{w}(x_i)\\
\end{align*}
$$

Let's consider the case where we do not have access to the normalization constants $Z_q$ or $Z_p$, we can still approximate it by first considering the expected value of an importance weight over samples drawn from $q$:

$$
\begin{align*}
\mathbb{E}_{x \sim q} [\widetilde{w}(x)] &= \int \widetilde{w}(x) q(x) dx\\
&= \int \frac{\widetilde{p\hspace{0.1em}}(x)}{\widetilde{q\hspace{0.15em}}(x)} q(x) dx\\
&= \int \frac{ q(x)}{\widetilde{q\hspace{0.15em}}(x)} \widetilde{p\hspace{0.1em}}(x) dx\\
&= \frac{1}{Z_q} \int \widetilde{p\hspace{0.1em}}(x) dx\\
&= \frac{Z_p}{Z_q} \\
\end{align*}
$$

Since we have $N$ samples from $q$, we can estimate the ratio $\frac{ Z_p}{Z_q}$ with a Monte Carlo estimator:

$$\frac{ Z_p}{Z_q} = \mathbb{E}_{x \sim q} [\widetilde{w}(x)] \approx \frac{1}{N} \sum_{i=1}^N \widetilde{w}(x_i)$$


Plugging this into our original equation, we get:

$$
\begin{align*}
\int  f(x) p(x) dx &\approx N \frac{1}{\sum_{i=1}^N \widetilde{w}(x_i)}  \frac{1}{N} \sum_{i=1}^{N} f(x_i) \widetilde{w}(x_i)\\
&\approx \frac{\sum_{i=1}^{N} f(x_i) \widetilde{w}(x_i)}{\sum_{i=1}^N \widetilde{w}(x_i)}  \\
\end{align*}
$$

Now, we have a method for utilizing samples from $q$ to learn something about our target distribution $p$.

## A Basic Example of Importance Sampling
Let's suppose that we're working in a 1 dimensional space and we want to find the average over our very simple target distribution $p = Uniform[1, 6]$. We know from elementary statistics that:

$$\mathbb{E}_{x \sim p} [x] = \frac{1 + 6}{2} = 3.5$$

In addition, almost every mathematical package out there can sample from a Uniform distribution. But for the purposes of demonstration, let's assume that we can only sample from a 1-D normal distribution defined as $q = \mathcal{N}( \mu_q, \sigma_q^2 )$. We will utilize samples $x \sim q$ and importance sampling to approximate the value $\mathbb{E}_{x \sim p} [x]$ (which we know should be $3.5$). In other words, we will be importance sampling with $f(x) = x$ for all $x$.


Use the sliders below to adjust the parameters of our $q$ distribution, then generate samples from $q$ and watch the estimate for $\mathbb{E}_{x \sim p} [x]$ asymptotically converge.

{% include animation-1.html %}


## A More Complex Example

Let's consider a more complex example brought up in chapter 29 of "Information Theory, Inference, and Learning Algorithms" by David Mackay:

$$\widetilde{p\hspace{0.1em}}(x) = 0.5 e^{0.4(x-0.4)^2 - 0.08x^4}$$

Since we don't know the normalizing constant $Z_p$ (and would have a very difficult time computing it analytically), we will use <a href="https://www.wolframalpha.com/input/?i=integral+from+-inf+to+inf+of+x+exp%280.4%28x-0.4%29%5E2+-+0.08x%5E4%29+dx+%2F+integral+from+-inf+to+inf+of+exp%280.4%28x-0.4%29%5E2+-+0.08x%5E4%29+dx" target="_blank">Wolfram Alpha</a> to get an estimated expected value:

$$\frac{\int_{-\infty}^\infty x 0.5 e^{0.4(x-0.4)^2 - 0.08x^4} dx}{\int_{-\infty}^\infty 0.5 e^{0.4(x-0.4)^2 - 0.08x^4} dx } \approx -0.682815$$

So we want to see an estimated value around $-0.682815$ from our importance sampling simulation:

{% include animation-2.html %}

From these simulations, we see that importance sampling well approximates the expected values over these relatively simple distributions. But for more complex and higher dimensional distributions, we run into issues with numerical stability. For example, we might luckily sample a point $x$ with $\widetilde{q\hspace{0.15em}}(x) \ll \widetilde{p\hspace{0.1em}}(x)$, resulting an extremely large $\widetilde{w}(x)$, which would then dominate the expected value estimate. In order to circumvent this, we look towards the method of Rejection Sampling.

<!-- Especially in higher dimensions, the <a href="https://arxiv.org/pdf/1701.02434" target="_blank">typical set</a> (the region that contributes the most to our expected value computation) few points falling  -->

## Why We Shouldn't Reject Rejection Sampling

Rejection Sampling is a slightly different method for solving problem 2, i.e. approximating $\int f(x) p(x) dx$. All the assumptions from importance sampling still persist but we also require knowledge of some constant $c$ such that for all $x$ over the support of $q$, we have:

$$c \widetilde{q\hspace{0.15em}}(x) > \widetilde{p\hspace{0.1em}}(x)$$

This essentially produces a probabilistic bounding box around $\widetilde{p\hspace{0.1em}}$. From here, we sample a point $x \sim q$ and then a point $u \sim Uniform[0, c\widetilde{q\hspace{0.15em}}(x)]$. Then, we compare $u$ with $\widetilde{p\hspace{0.1em}}(x)$, if $u > \widetilde{p\hspace{0.1em}}(x)$, then we reject that sample, otherwise we accept it. 

We can interpret this procedure as first uniformly selecting a point under the curve $c \widetilde{q\hspace{0.15em}}$. Then, keeping that point only if it falls under the curve $\widetilde{p\hspace{0.1em}}$. So our accepted samples are distributed proportionally to $\widetilde{p\hspace{0.1em}}$, i.e. they are distributed according to $p$. Then, we can use a simple Monte Carlo estimator to estimate:

$$\mathbb{E}_{x \sim p} [f(x)] \approx \frac{1}{N} \sum_{i=1}^{M} f(x_i)$$

Where we have $M$ accepted points $\{x_i\}_{i=1}^M$. In our case, we just want the vanilla expected value, so we have:

$$\mathbb{E}_{x \sim p} [x] \approx \frac{1}{N} \sum_{i=1}^{M} x_i$$

Let's see how this compares to Importance Sampling for our previous 2 example $p$'s.

{% include animation-3.html %}

Note that the red lines indicate samples that were rejected. Now for the more complex distribution:

{% include animation-4.html %}


## Rejection Sampling in Higher Dimensions

We can see that the rejection sampling approximates $\int f(x) p(x) dx$ about as well as importance sampling does, without the risk of numerical instability. However, the downside is that a proportion of our samples are rejected and have no use. This problem becomes an increasing issue in higher dimensions as the volume sandwiched between $c \widetilde{q\hspace{0.15em}}$ and $\widetilde{p\hspace{0.1em}}$ grows with increasing dimensions. Eventually, unless $c\widetilde{q\hspace{0.15em}}$ and $\widetilde{p\hspace{0.1em}}$ are the exact same distribution, the acceptance rate approaches 0%, i.e. most samples that we draw from $q$ will immediately be thrown away.

Let's consider a situation where our $p = \mathcal{N}(0, 1)$ and our $q$ is a slightly more spread out normal distribution $ \mathcal{N}(0, 1.01)$:

{% include animation-5.html %}

Recall that the normal distribution $\mathcal{N}(\mu, \sigma)$ has the density function:

$$f(x) = \frac{1}{\sigma \sqrt{2\pi}} e^{-\frac{1}{2} \left( \frac{x - \mu}{\sigma} \right)^2}$$

So we have:

$$
\begin{align*}
p(0) &= \frac{1}{\sqrt{2\pi}}\\
q(0) &= \frac{1}{1.01 \sqrt{2\pi}}\\
\frac{p(0)}{q(0)} &= \frac{1}{\sqrt{2\pi}} \cdot 1.01 \sqrt{2\pi} = 1.01
\end{align*}
$$

We need a $c > 1.01$ in 1-D to satisfy $c q > p$. However, in for an $n$-dimensional Normal Distribution, which we will denote $\mathcal{N}^n (\mu, \sigma)$, we recall that

$$\mathcal{N}^n (\mu, \sigma) = \mathcal{N}(\mu, \sigma) \times \mathcal{N}(\mu, \sigma) \times \cdots \times \mathcal{N}(\mu, \sigma)$$

Where we multiply a 1 dimensional normal distribution with another $n$ times. This means that for $p = \mathcal{N}^n (0, 1)$ and $q = \mathcal{N}^n (0, 1.01)$, we have:

$$
\begin{align*}
p(0) &= \left(\frac{1}{\sqrt{2\pi}}\right)^n\\
q(0) &= \left(\frac{1}{1.01 \sqrt{2\pi}}\right)^n\\
\frac{p(0)}{q(0)} &= 1.01^n
\end{align*}
$$

So we have $c \rightarrow \infty$ as $n \rightarrow \infty$. Let's see this empirically:

{% include animation-6.html %}

From this experiment, we see that the acceptance rate drops significantly with increasing dimension:

{% include animation-7.html %}

This values were generated by running the above simulation over $10,000$ samples with $\sigma_q = 1.01$ and seeing what percentage of samples were accepted. Sadly, there is no way for me to visually show the sampling from these higher dimensions on a 2 dimensional computer screen. However, I'll leave you with a piece of advice:

{% include image.html url="/assets/images/hinton.jpg" %}
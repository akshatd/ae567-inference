---
geometry: "margin=2cm"
---

## 3.1 1-D Bernoulli random walk

### 2.1.a

Visualization of an N-step 1-D Bernoulli random walk, with $n=1000$ steps.

![1-D Bernoulli random walk vs n](figs/3.1.svg){width=60%}

### 2.1.b

To perform a Monte Carlo estimate of $\mathbb P(S>10)$, we can simulate the random walk $10^5$ times with 100 steps in each trial and count the number of times the walk exceeds 10. The probability is then the ratio of the number of times the walk exceeds 10 to the total number of trials. This a simple indicator function that can be used with the Monte Carlo method.

Monte Carlo estimate with $10^5$ trials with 1000 steps in each walk for $\mathbb P(S>10)$ = 0.13579

$\pagebreak$

### 2.1.c

Before performing importance sampling, lets take a look at the distribution of the random walk with 100 steps.

![1-D Bernoulli random walk distribution](figs/3.2.svg){width=60%}

The distribution has

- Mean = 0.0838
- Variance = 100.63476096000001

As expected from a random walk, the distribution is normal with a mean $\approx 0$ and a standard deviation $\approx$ the number of time steps, which is the same as the number of steps in our case since we are not shrinking the time. However, we see that the probability of the walk exceeding 55 is very low, which makes it a good candidate for importance sampling.

To perform the importance sampling, we need a proposal distribution that has a high probability of exceeding 55. We can just use another random walk with a the probability $\mathbb P(X_j = 1) > \mathbb P(X_j = -1)$. If we ensure that the expected value of this new distribution is around 55, we can ensure that at least half the samples will exceed 55 so we can get a better estimate. To calculate the new probability for 100 steps, we can use a simple formula to estimate the value of a random walk given a probability $p$ :

$$
\sum_{i=1}^{100} (p)(1) + (1-p)(-1) > 55 \\
$$

$$
100(2p - 1) > 55 \\
$$

$$
p > 0.775
$$

So we can use a proposal distribution as a random walk with $\mathbb P(X_j = 1) = 0.8, \mathbb P(X_j = -1) = 0.2$ to get a better estimate for $S>55$. The Importance sampling estimate for $\mathbb P(S>55) = 7.966439961067404e-09$.

### 2.1.d

Analytical expression for the probability of the random walk exceeding 55 is a simple probability calculation for a binomial distribution:

$$
\mathbb P(S>55) = \sum_{i=56}^{100} \mathbb P(S = i) \\
$$

For $\mathbb P(S=i)$ we would have $j$ steps with +1 and $100-j$ steps with -1. Then this must hold true for $i, j$:

$$
j - (100-j) = i \\
2j - 100 = i \\
j = \frac{i+100}{2} \\
$$

Take note that we can only choose an integer from another integer, so values of j that are fractions will be ignored. This will happen when $i$ is odd, and there is no way for this random walk to generate a final step that is odd.

For n steps $\mathbb P(S=i)$ becomes:

$$
\begin{aligned}
\mathbb P(S=i) &= {100 \choose j} p^j (1-p)^{100-j}\\
p = (1-p) = 0.5, j = \frac{i+100}{2}\\
&= {100 \choose j} p^{100}\\
&= {100 \choose \frac{i+100}{2}} 0.5^{100}\\
\end{aligned}
$$

Substituting this into $\mathbb P(S>55)$:

$$
\mathbb P(S>55) = \sum_{i=56}^{100} {100 \choose \frac{i+100}{2}} 0.5^{100}
$$

Evaluating this expression gives us $\mathbb P(S>55) = 7.95266423689307e-09$, which is very close to the importance sampling estimate with a difference of only 0.17%.

Interestingly, increasing the $\mathbb P(X_j = 1)$ too high in the prposal distribution closer to 0.9 actually causes the importance sampling estimate to become worse. This could be explained by the fact that the proposal distribution gets too far from the original distribution, and there isnt really any information left in the proposal distribution to calculate probabilities from. If we set this to 1, then the importance sampling estimate becomes 0, which is not correct. So it makes sense to keep it close to the original distribution at 0.8.

### 2.1.e.i

Standard errors:

- Monte Carlo = 1.083489e-03
- Importance Sampling = 6.003709e-11

95% confidence intervals:

- Monte Carlo = [0.1337263621371712, 0.1379736378628288]
- Importance Sampling = [7.848767266908999e-09, 8.084112655225919e-09]

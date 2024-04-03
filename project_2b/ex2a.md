---
geometry: "margin=2cm"
---

## 2 Apply DRAM for Bayesian Inference

## 2.A Satellite Dynamics

After setting up the Satellite Dynamics, we can observe the true trajectory of the satellite and the noisy trajectory observed by the sensor data.

![Satellite Dynamics](figs/Satellite%20Dynamics.svg){height=73%}

## Problem 1: Parameters are the control coefficients (k1, k2)

### 1 What is the likelihood model? Please describe how you determine this.

For this exercise, the likelihood model will tell us how likely/probable the observed sensor data is given the parameters of the model $(k_1, k_2)$. We only have access to a subset of all data $(q_1, q_2, q_3)$ that the satellite generates, so we will focus on just the data that is available. Given a function $f(\theta,t)$ that propagates the trajectory of the satellite using the parameters $\theta = (k_1,k_2)$ and outputs the quaternions $(q_1^{(t)}, q_2^{(t)}, q_3^{(t)})$ at any given time $t$,

Let

$$
\begin{aligned}
\text{params: }\theta &= (k_1,k_2) \\
\text{time instances: } t &= 1,2,...,50 \\
\text{trajectory function: } f(\theta, t) &= \begin{bmatrix} q_1^{(t)} \\ q_2^{(t)} \\ q_3^{(t)} \end{bmatrix} \\
\end{aligned}
$$

We have already observed the sensor data $y^{(t)}$ for the true satellite trajectory at each time instance $t$.

$$
\text{actual sensor data at time t: } y^{(t)} = \begin{bmatrix} y_1^{(t)} \\ y_2^{(t)} \\ y_3^{(t)} \end{bmatrix} \\
$$

Hence, the likelihood can be written as:

$$
\text{Likelihood: } \mathcal{L}(\theta) = \mathbb{P}(y|\theta)
$$

The sensor captures quaternion data ($q_1,q_2,q_3$) corrupted by independent Gaussian noise with 0 mean and standard deviation of 0.05. This means the data captured by the sensor of the actual satellite trajectory is not deterministic, but stochastic and follows a probability distribution. The meaning of $\mathbb{P}(y|\theta)$ is that we will be comparing the sensor data $y^{(t)}$ with the trajectory output $q^{(t)}$ by the function $f(\theta,t)$ at each time instance $t$.

$$
\begin{aligned}
\text{sensor noise: } \xi &\sim \mathcal{N}(0, 0.05^2) \\
\text{distribution of sensor data at time t: } s^{(t)} &= \begin{bmatrix} q_1^{(t)} + \xi \\ q_2^{(t)} + \xi \\ q_3^{(t)} + \xi \end{bmatrix}
\end{aligned}
$$

$s^{(t)}$ is a multivariate gaussian distribution with mean $q^{(t)}$ and covariance matrix $0.05^2 I$. This is because the noise is independent and identically distributed for each quaternion component.

$$
\begin{aligned}
\text{dimensions: } d &= 3 \\
\text{covariance matrix: } \Sigma &= 0.05^2 I = \begin{bmatrix} 0.05^2 & 0 & 0 \\ 0 & 0.05^2 & 0 \\ 0 & 0 & 0.05^2 \end{bmatrix} \\
s^{(t)} &= \mathcal{N}(q^{(t)}, \Sigma) \\
\end{aligned}
$$

Now we know that the likelihood for a certain sensor measurement of the quaternions is determined by how far away it is from the quaternions of the trajectory propagated by certain $\theta$, taking into account the noise of the sensor. Different parameters will result in different trajectories, and the sensor data might be closer to some trajectories than others. The likelihood then is the probability of the observed sensor data given a trajectory mapped by the chosen parameters. In other words, we are trying to see how probable/likely the data captured by the sensor is given its noise, if a certain trajectory was mapped by the chosen parameters. The trajectory determined by the parameters is in continuous time, while the sensor data is in discrete time. We need to select the quaternions of the trajectory at the same time instances as the sensor data. The probability of $y^{(t)}$ given $\theta$ is then the probability of a multivariate gaussian distribution with mean $q^{(t)}$ (output of the function $f(\theta,t)$) and covariance matrix $0.05^2 I$ at $y^{(t)}$. The likelihood is then the product of the probabilities of the $y^{(t)}$ given $s^{(t)}$ at each time instance $t$.

$$
\text{Likelihood: } \mathcal{L}(\theta) = \prod_{t=1}^{50} \mathbb{P}(y^{(t)} | s^{(t)})
$$

For ease of computation, we will actually use the log-likelihood.

$$
\text{Log-Likelihood: } \log \mathcal{L}(\theta) = \sum_{t=1}^{50} \log \mathbb{P}(y^{(t)} | s^{(t)}) \\
$$

Since the noise is a multivariate gaussian, we can easily use log probability of a multivariate gaussian to compute this value.

Let

$$
\begin{aligned}
\text{determinant of covariance matrix: } & |\Sigma| \\
\end{aligned}
$$

Then

$$
\begin{aligned}
\log \mathbb{P}(y^{(t)} | s^{(t)}) &= \log \mathcal{N}(y^{(t)} | q^{(t)}, \Sigma) \\
&= \log \left( \frac{\exp \left( -\frac{1}{2} (y^{(t)} - q^{(t)})^T \Sigma^{-1} (y^{(t)} - q^{(t)}) \right)}{\sqrt{(2\pi)^d |\Sigma|}} \right) \\
&= -\frac{d}{2} \log(2\pi) - \frac{1}{2} \log(|\Sigma|) - \frac{1}{2} (y^{(t)} - q^{(t)})^T \Sigma^{-1} (y^{(t)} - q^{(t)}) \\
&= -\frac{1}{2} \left( d \log(2\pi) + \log(|\Sigma|) + (y^{(t)} - q^{(t)})^T \Sigma^{-1} (y^{(t)} - q^{(t)}) \right) \\
\end{aligned}
$$

**Note:** If the matrices $y$ and $q$ contain all the quaternions for all time instances,

$$
\begin{aligned}
y &= \begin{bmatrix} y_1^{(1)} & y_1^{(2)} & ... & y_1^{(50)} \\ y_2^{(1)} & y_2^{(2)} & ... & y_2^{(50)} \\ y_3^{(1)} & y_3^{(2)} & ... & y_3^{(50)} \end{bmatrix} \\
q &= \begin{bmatrix} q_1^{(1)} & q_1^{(2)} & ... & q_1^{(50)} \\ q_2^{(1)} & q_2^{(2)} & ... & q_2^{(50)} \\ q_3^{(1)} & q_3^{(2)} & ... & q_3^{(50)} \end{bmatrix} \\
\end{aligned}
$$

And the difference $y - q$ is computed element-wise, the log likelihood can be computed in a vectorized form which is explored in the `multigauss_logpdf` function in the code giving a ~1000x speedup over `scipy.stats.multivariate_normal.logpdf` and 10x over `lognormpdf`.

### 2 What is the form of the posterior?

In Bayesian inference, the posterior is the probability of the parameters($\theta$) given the observed data($y$). It is proportional to the likelihood of the data given the parameters and the prior probability of the parameters.

$$
\mathbb{P}(\theta | y) = \frac{\mathbb{P}(\theta) \mathbb{P}(y | \theta)}{\mathbb{P}(y)} \\
$$

Where

$$
\begin{aligned}
\mathbb{P}(\theta) &= \text{Prior probability of the parameters} \\
\mathbb{P}(y | \theta) &= \text{Likelihood of the data given the parameters} \\
\mathbb{P}(y) &= \text{Marginal likelihood of the data} \\
\mathbb{P}(\theta | y) &= \text{Posterior probability of the parameters given the data} \\
\end{aligned}
$$

The posterior is basically the prediction of our parameters $\theta = (k_1,k_2)$ using our prior belief about $k_1,k_2$ (Prior), how probable the data $y$ is given a trajectory propagated by the parameters $k_1,k_2$ (Likelihood) and the probability of the data $y$ itself(Marginal Likelihood). The Prior for each $\theta_i$ is given in the problem statement to be $\theta_i \sim \mathcal{N}(0, 1)$, so we can both sample from it and determine the probability density. The Likelihood can be computed as described in the previous section. The Marginal Likelihood is the probability of the data $y$ given the model of the satellite, which is the integral of their probability over all possible parameters $k_1,k_2$. Since the space of parameters is unbounded unsolvable analytically, this integral is intractable. This is why we use MCMC methods to sample from the posterior distribution directly, because it is able to sample from a distribution that cannot be be evaluated up to their normalizing constant, which is the marginal likelihood of the data in this case.

In this way, we can compute the posterior and log posterior distribution by:

$$
\begin{aligned}
\text{Posterior: } \mathbb{P}(\theta | y) &: \mathbb{P}(\theta) \mathbb{P}(y | \theta) \\
\text{Log Posterior: } \log \mathbb{P}(\theta | y) &: \log \mathbb{P}(\theta) + \log \mathbb{P}(y | \theta) \\
\end{aligned}
$$

Here both the probabilities are multivariate gaussian and can be calculated in the same way as described in the previous section.

### 3 How did you tune your proposal? Think carefully about what a good initial point and initial covariance could be?

### 4 Analyze your results using the same deliverables as you used in Section 1.

### 5 Plot the true parameters on your plots of the marginals for reference.

### 6 Plot the prior and posterior predictives of the dynamics (separately)

## Problem 2: Parameters are the control coefficients and a product of inertia (k1, k2, J12)

### 1 What is the likelihood model? Please describe how you determine this.

### 2 What is the form of the posterior?

### 3 How did you tune your proposal? Think carefully about what a good initial point and initial covariance could be?

### 4 Analyze your results using the same deliverables as you used in Section 1.

### 5 Plot the true parameters on your plots of the marginals for reference.

### 6 Plot the prior and posterior predictives of the dynamics (separately)

for some prior/posterior samples, run the dynamics and plot them in a transparent light gray on top of your “truth” dynamics and data. These are essentially the probabilistic predictions of your model before and after you have accounted for the data. How do they look compared to the “truth.” How does this plot differ between the prior and posteriors?

### Please comment on the following:

- What is the difference between the two parameter inference problems?
- How does the posterior predictive change?
- Are there any notable differences?
  $$

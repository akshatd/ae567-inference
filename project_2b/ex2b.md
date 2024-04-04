---
geometry: "margin=2cm"
---

$\pagebreak$

## 2.B SIR Model

After setting up the SIR Model, we can see the observed evolution of the disease and the data points in the plot below. This is done using the true params of the identifiable model since we cannot generate the true evolution using the non-identifiable model.

![SIR Model Evolution](figs/SIR%20Model.svg){height=73%}

## Problem 1 (Identifiable): Parameters are $\theta$ = $(\beta, r, \delta)$

### 1 What is the likelihood model? Please describe how you determine this.

For this exercise, the likelihood model will tell us how likely/probable the observed data is given the parameters of the model $(\beta, r, \delta)$. We only have access to a subset of all data (only I from S, I and R) that the evolution trajectory generates, so we will focus on just the data that is available. Given a function $f(\theta,t)$ that evolves the trajectory of the disease model using the parameters $\theta = (\beta, r, \delta)$ and outputs the Infected $(I^{(t)})$ at any given time $t$,

Let

$$
\begin{aligned}
\text{params: }\theta &= (\beta, r, \delta) \\
\text{time instances: } t &= 1,2,...,61 \\
\text{trajectory function: } f(\theta, t) &= I^{(t)}
\end{aligned}
$$

We have already observed the data $y^{(t)}$ for the true trajectory at each time instance $t$.

$$
\text{actual observed data at time t: } y^{(t)}
$$

Hence, the likelihood can be written as:

$$
\text{Likelihood: } \mathcal{L}(\theta) = \mathbb{P}(y|\theta)
$$

We observe data ($I$) corrupted by independent Gaussian noise with 0 mean and standard deviation of 50. This means the data observed of the actual disease trajectory is not deterministic, but stochastic and follows a probability distribution. The meaning of $\mathbb{P}(y|\theta)$ is that we will be comparing the observed data $y^{(t)}$ with the trajectory output $I^{(t)}$ by the function $f(\theta,t)$ at each time instance $t$.

$$
\begin{aligned}
\text{data noise: } \xi &\sim \mathcal{N}(0, 50^2) \\
\text{distribution of observed data at time t: } o^{(t)} &= I^{(t)} + \xi
\end{aligned}
$$

$o^{(t)}$ is a gaussian distribution with mean $I^{(t)}$ and variance $50^2$.

$$
\begin{aligned}
o^{(t)} &= \mathcal{N}(I^{(t)}, \Sigma) \\
\end{aligned}
$$

Now we know that the likelihood for a certain observation of the infected population is determined by how far away it is from the infected population of the trajectory evolved by certain $\theta$, taking into account the noise of in our observations. Different parameters will result in different evolutions, and the observed data might be closer to some evolutions than others. The likelihood then is the probability of the observed data given a trajectory evolved by the chosen parameters. In other words, we are trying to see how probable/likely the data of the infected population is given its noise, if a certain trajectory was evolved by the chosen parameters. The trajectory determined by the parameters is in continuous time, while the observed data is in discrete time. We need to select the infected population of the evolution at the same time instances as the observed data. The probability of $y^{(t)}$ given $\theta$ is then the probability of the gaussian distribution with mean $I^{(t)}$ (output of the function $f(\theta,t)$) and variance $50^2$ at $y^{(t)}$. The likelihood is then the product of the probabilities of the $y^{(t)}$ given $o^{(t)}$ at each time instance $t$.

$$
\text{Likelihood: } \mathcal{L}(\theta) = \prod_{t=1}^{61} \mathbb{P}(y^{(t)} | o^{(t)})
$$

For ease of computation, we will actually use the log-likelihood.

$$
\text{Log-Likelihood: } \log \mathcal{L}(\theta) = \sum_{t=1}^{61} \log \mathbb{P}(y^{(t)} | o^{(t)}) \\
$$

Since the noise is gaussian, we can easily use log probability of a gaussian to compute this value.

Let

$$
\sigma = 50
$$

Then

$$
\begin{aligned}
\log \mathbb{P}(y^{(t)} | o^{(t)}) &= \log \mathcal{N}(y^{(t)} | I^{(t)}, \Sigma) \\
&= \log \left( \frac{\exp \left( -\frac{1}{2} (\frac{y^{(t)} - I^{(t)}}{\sigma})^2 \right)}{\sigma\sqrt{2\pi}} \right) \\
&= -\frac{1}{2} \log(2\pi) - \log(\sigma) - \frac{1}{2} \left( \frac{y^{(t)} - I^{(t)}}{\sigma} \right)^2 \\
\end{aligned}
$$

We can then plug this into the log-likelihood function to compute the log-likelihood of the data given the parameters for all time instances.

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

The posterior is basically the prediction of our parameters $\theta = (\beta,r,\delta)$ using our prior belief about $(\beta, r, \delta)$ (Prior), how probable the data $y$ is given a trajectory propagated by the parameters $(\beta, r, \delta)$ (Likelihood) and the probability of the data $y$ itself(Marginal Likelihood). The Prior for each $\theta_i$ is given in the problem statement to be $\theta_i \sim \mathcal{N}(0, 1)$, so we can both sample from it and determine the probability density. The Likelihood can be computed as described in the previous section. The Marginal Likelihood is the probability of the data $y$ given the SIR model, which is the integral of its probability over all possible parameters $(\beta, r, \delta)$. Since the space of parameters is unbounded unsolvable analytically, this integral is intractable. This is why we use MCMC methods to sample from the posterior distribution directly, because it is able to sample from a distribution that cannot be be evaluated up to their normalizing constant, which is the marginal likelihood of the data in this case.

In this way, we can compute the posterior and log posterior distribution by:

$$
\begin{aligned}
\text{Posterior: } \mathbb{P}(\theta | y) &: \mathbb{P}(\theta) \mathbb{P}(y | \theta) \\
\text{Log Posterior: } \log \mathbb{P}(\theta | y) &: \log \mathbb{P}(\theta) + \log \mathbb{P}(y | \theta) \\
\end{aligned}
$$

Here the prior is a multivariate gaussian but with an identity covariance, which means we can treat all the parameters are independent and calculate the logPDF for each parameter independently.The likelihood is a gaussian as well, as described in the previous section and can be calculated as described in the previous section.

### 3 How did you tune your proposal? Think carefully about what a good initial point and initial covariance could be?

The proposal used for the DRAM is a gaussian random walk proposal. For the initial point we can start off from the Maximum Aposteriori(MAP) estimate of the posterior distribution. This will be a certain value of $\theta=(\beta, r, \delta)$. The MAP estimate can be found by maximizing the logPDF of the posterior distribution. If we simplify the problem and approximate the posterior as a 3D gaussian, the scaled negative inverse of the hessian at the MAP can be used as the covariance matrix for a random walk proposal. This is also called the Laplace Approximation. The scaling factor of the negative inverse hessian is important for determining the initial covariance of the the random walk proposal and has been taken from the suggestion in the notes (ref: eq 16.6). Specifically:

Let

$$
\text{Log Posterior: } \log \mathbb{P}(\theta | y) = f_{Xpost}
$$

Then

$$
\begin{aligned}
\theta_{MAP} &= \arg\max_x f_{Xpost} \\
\theta^{(0)} &= \theta_{MAP} \\
\Sigma_{proposal}^{(0)} &= -\frac{2.4^2}{d}\left(\nabla^2 f_{Xpost}(\theta_{MAP})\right)^{-1} \\
\end{aligned}
$$

Using `scipy.optimize.minimize` with $-f_{Xpost}$ as the objective function, we find the MAP and initial proposal covariance matrix:

- $\theta^{(0)} = (\beta, r, \delta)$: (0.02 0.61 0.14)
- $$
  \begin{aligned}
  & \Sigma_{proposal}^{(0)} :
  \begin{bmatrix}
  4.82564094e-06 & -6.58406798e-06 & -4.59524017e-06 \\
  -6.58406798e-06 & 1.23289295e-03 & 6.73081629e-04 \\
  -4.59524017e-06 & 6.73081629e-04 & 5.56126832e-04 \\
  \end{bmatrix}
  \end{aligned}
  $$

After the initial $\theta^{(0)}=(\beta^{(0)},r^{(0)},\delta^{(0)})$ and $\Sigma_{proposal}^{(0)}$ are determined, the proposal is tuned by adjusting $\gamma_{DRAM}$ and $s_d$ in the DRAM algorithm as described in part 1 of this project. Here i use the notation $\gamma_{DRAM}$ for the tuning knob inside DRAM, and $\gamma$ for the parameter of the SIR model we are trying to infer The covariance of the accepted samples is updated at each iteration and scaled by $s_d$ to get the covariance matrix of the first proposal, and if the first proposal is rejected, the covariance matrix is scaled further by $\gamma_{DRAM}$ for the second proposal.

Since the ideal acceptance ratio is 20-30%, I try to change $s_d$ and $\gamma_{DRAM}$ to get an acceptance ratio in this range. While doing this, I also monitor the Marginals, Autocorrelation vs lag plot, the Integrated Autocorrelation(IAC) and the visual inspection of the samples to ensure that the samples are mixing well and the proposal is tuned correctly. Specifically

- Acceptance ratio:
  - if low, decrease $s_d$ and $\gamma_{DRAM}$
  - if high, increase $s_d$ and $\gamma_{DRAM}$
- Marginals:
  - if clustering around a few points and not exploring enough, increase $s_d$ and $\gamma_{DRAM}$
- Autocorrelation:
  - if doesnt decay quickly, increase $s_d$ and $\gamma_{DRAM}$
- IAC:
  - if high, increase $s_d$ and $\gamma_{DRAM}$

For the sake of speed, to increase the acceptance ratio I preferred to decrease $s_d$ before I increased $\gamma_{DRAM}$, because if the first proposal is rejected, the trajectory has to be propagated again for the second proposal, which is computationally expensive. Hence I prefered the first proposal to be accepted more often rather than it rejecting and the 2nd proposal being accepted. $\gamma_{DRAM}$ is meant to be used when the first proposal rejected for being too exploratory, so it should ideally be $\leq$ 1 so the second proposal is more conservative.

After tuning, I ended up with $s_d=9$ and $\gamma_{DRAM}=0.9$ for the final proposal. I started off with a high acceptance ratio, so I increased $s_d$ to get the acceptance ratio in the desired range. I also increased $\gamma_{DRAM}$ slightly. I made sure that all other metrics remained good while this happened.

$\pagebreak$

### 4/5 Analyze your results using the same deliverables as you used in Section 1.

![Marginals for Problem 1](figs/2.B%20Problem%201_marginal.svg){width=100%}

![Visual Inspection for Problem 1](figs/2.B%20Problem%201_inspection.svg){width=75%}

![Autocorrelation for Problem 1](figs/2.B%20Problem%201_autocorrelation.svg){width=75%}

- Integrated autocorrelation values:

  - $\beta$: 2.60
  - $r$: 4.61
  - $\delta$: 6.47

- Acceptance ratio: 30%

The acceptance ratio started off very high, but was tuned to 30% while making sure that the samples are mixing well. We can see that the marginals look good, and the Autocorrelation plot decays quickly leading the low values of IAC. Finally, the visual inspection of the samples shows that the samples do not get stuck in a local area and the MC is exploring the parameter space well.

When we compare the value of the true $(\beta, r, \delta)$ with the marginals, we can see that the true values are actually quite close, but not exactly the peak of the gaussian marginals except for $\beta$. This could be due to there being noise in our data. The fact that our prior was not very far of from the true values should have helped them get this close, closer than for problem 2A. Given these limitation, we cannot expect the posterior to converge exactly to the true values, but we can see that true values are not that far off from the peak of the marginals, which is a good sign. Also given that all the covariance plots look like gaussians, it makes sense that we have converged close to the true values using a gaussian proposal for our MCMC.

$\pagebreak$

### 6 Plot the prior and posterior predictives of the dynamics (separately)

The true data is in blue and the predicted data is in grey

![Prior Predictive for Problem 1](figs/2.B%20Problem%201%20Prior_predictive.svg){width=80%}

![Posterior Predictive for Problem 1](figs/2.B%20Problem%201%20Posterior_predictive.svg){width=80%}

The prior predictive looks like it grows/decays exponentially, only being somewhere near the true trajectory in the beginning while the posterior predictive actually matches the true trajectory of the evolution quite closely. This is a good sign that our model is working well and the parameters are being inferred correctly. The difference between the prior and posterior predictives is quite stark, showing the power of Bayesian inference given that after learning the parameters we can make much better predictions.

$\pagebreak$

## Problem 2 (Not-identifiable): Parameters are $\theta = (\gamma, \kappa, r, \delta)$

### 1 What is the likelihood model? Please describe how you determine this.

For this exercise, the likelihood model will tell us how likely/probable the observed data is given the parameters of the model $(\gamma, \kappa, r, \delta)$. We only have access to a subset of all data (only I from S, I and R) that the evolution trajectory generates, so we will focus on just the data that is available. Given a function $f(\theta,t)$ that evolves the trajectory of the disease model using the parameters $\theta = (\gamma, \kappa, r, \delta)$ and outputs the Infected $(I^{(t)})$ at any given time $t$,

Let

$$
\begin{aligned}
\text{params: }\theta &= (\gamma, \kappa, r, \delta) \\
\text{time instances: } t &= 1,2,...,61 \\
\text{trajectory function: } f(\theta, t) &= I^{(t)}
\end{aligned}
$$

We have already observed the data $y^{(t)}$ for the true trajectory at each time instance $t$.

$$
\text{actual observed data at time t: } y^{(t)}
$$

Hence, the likelihood can be written as:

$$
\text{Likelihood: } \mathcal{L}(\theta) = \mathbb{P}(y|\theta)
$$

We observe data ($I$) corrupted by independent Gaussian noise with 0 mean and standard deviation of 50. This means the data observed of the actual disease trajectory is not deterministic, but stochastic and follows a probability distribution. The meaning of $\mathbb{P}(y|\theta)$ is that we will be comparing the observed data $y^{(t)}$ with the trajectory output $I^{(t)}$ by the function $f(\theta,t)$ at each time instance $t$.

$$
\begin{aligned}
\text{data noise: } \xi &\sim \mathcal{N}(0, 50^2) \\
\text{distribution of observed data at time t: } o^{(t)} &= I^{(t)} + \xi
\end{aligned}
$$

$o^{(t)}$ is a gaussian distribution with mean $I^{(t)}$ and variance $50^2$.

$$
\begin{aligned}
o^{(t)} &= \mathcal{N}(I^{(t)}, \Sigma) \\
\end{aligned}
$$

Now we know that the likelihood for a certain observation of the infected population is determined by how far away it is from the infected population of the trajectory evolved by certain $\theta$, taking into account the noise of in our observations. Different parameters will result in different evolutions, and the observed data might be closer to some evolutions than others. The likelihood then is the probability of the observed data given a trajectory evolved by the chosen parameters. In other words, we are trying to see how probable/likely the data of the infected population is given its noise, if a certain trajectory was evolved by the chosen parameters. The trajectory determined by the parameters is in continuous time, while the observed data is in discrete time. We need to select the infected population of the evolution at the same time instances as the observed data. The probability of $y^{(t)}$ given $\theta$ is then the probability of the gaussian distribution with mean $I^{(t)}$ (output of the function $f(\theta,t)$) and variance $50^2$ at $y^{(t)}$. The likelihood is then the product of the probabilities of the $y^{(t)}$ given $o^{(t)}$ at each time instance $t$.

$$
\text{Likelihood: } \mathcal{L}(\theta) = \prod_{t=1}^{61} \mathbb{P}(y^{(t)} | o^{(t)})
$$

For ease of computation, we will actually use the log-likelihood.

$$
\text{Log-Likelihood: } \log \mathcal{L}(\theta) = \sum_{t=1}^{61} \log \mathbb{P}(y^{(t)} | o^{(t)}) \\
$$

Since the noise is gaussian, we can easily use log probability of a gaussian to compute this value.

Let

$$
\sigma = 50
$$

Then

$$
\begin{aligned}
\log \mathbb{P}(y^{(t)} | o^{(t)}) &= \log \mathcal{N}(y^{(t)} | I^{(t)}, \Sigma) \\
&= \log \left( \frac{\exp \left( -\frac{1}{2} (\frac{y^{(t)} - I^{(t)}}{\sigma})^2 \right)}{\sigma\sqrt{2\pi}} \right) \\
&= -\frac{1}{2} \log(2\pi) - \log(\sigma) - \frac{1}{2} \left( \frac{y^{(t)} - I^{(t)}}{\sigma} \right)^2 \\
\end{aligned}
$$

We can then plug this into the log-likelihood function to compute the log-likelihood of the data given the parameters for all time instances.

### 2 What is the form of the posterior?

In Bayesian inference, the posterior is the probability of the parameters($\theta = \gamma, \kappa, r, \delta$) given the observed data($y$). It is proportional to the likelihood of the data given the parameters and the prior probability of the parameters.

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

The posterior is basically the prediction of our parameters $\theta = (\gamma, \kappa, r, \delta)$ using our prior belief about $(\gamma, \kappa, r, \delta)$ (Prior), how probable the data $y$ is given a trajectory propagated by the parameters $(\gamma, \kappa, r, \delta)$ (Likelihood) and the probability of the data $y$ itself(Marginal Likelihood). The Prior for each $\theta_i$ is given in the problem statement to be $\theta_i \sim \mathcal{N}(0, 1)$, so we can both sample from it and determine the probability density. The Likelihood can be computed as described in the previous section. The Marginal Likelihood is the probability of the data $y$ given the SIR model, which is the integral of its probability over all possible parameters $(\gamma, \kappa, r, \delta)$. Since the space of parameters is unbounded unsolvable analytically, this integral is intractable. This is why we use MCMC methods to sample from the posterior distribution directly, because it is able to sample from a distribution that cannot be be evaluated up to their normalizing constant, which is the marginal likelihood of the data in this case.

In this way, we can compute the posterior and log posterior distribution by:

$$
\begin{aligned}
\text{Posterior: } \mathbb{P}(\theta | y) &: \mathbb{P}(\theta) \mathbb{P}(y | \theta) \\
\text{Log Posterior: } \log \mathbb{P}(\theta | y) &: \log \mathbb{P}(\theta) + \log \mathbb{P}(y | \theta) \\
\end{aligned}
$$

Here the prior is a 4-D multivariate gaussian but with an identity covariance, which means we can treat all the parameters are independent and calculate the logPDF for each parameter independently.The likelihood is a 1-D gaussian as well, as described in the previous section and can be calculated as described in the previous section.

### 3 How did you tune your proposal? Think carefully about what a good initial point and initial covariance could be?

The proposal used for the DRAM is a gaussian random walk proposal. For the initial point we can start off from the Maximum Aposteriori(MAP) estimate of the posterior distribution. This will be a certain value of $\theta=(\gamma, \kappa, r, \delta)$. The MAP estimate can be found by maximizing the logPDF of the posterior distribution. If we simplify the problem and approximate the posterior as a 4D gaussian, the scaled negative inverse of the hessian at the MAP can be used as the covariance matrix for a random walk proposal. This is also called the Laplace Approximation. The scaling factor of the negative inverse hessian is important for determining the initial covariance of the the random walk proposal and has been taken from the suggestion in the notes (ref: eq 16.6). Specifically:

Let

$$
\text{Log Posterior: } \log \mathbb{P}(\theta | y) = f_{Xpost}
$$

Then

$$
\begin{aligned}
\theta_{MAP} &= \arg\max_x f_{Xpost} \\
\theta^{(0)} &= \theta_{MAP} \\
\Sigma_{proposal}^{(0)} &= -\frac{2.4^2}{d}\left(\nabla^2 f_{Xpost}(\theta_{MAP})\right)^{-1} \\
\end{aligned}
$$

Using `scipy.optimize.minimize` with $-f_{Xpost}$ as the objective function, we find the MAP and initial proposal covariance matrix:

- $\theta^{(0)} = (\gamma, \kappa, r, \delta)$: (0.14170413, 0.14170406, 0.61444686, 0.14098853)
- $$
  \begin{aligned}
  & \Sigma_{proposal}^{(0)} :
  \begin{bmatrix}
  0.00076125 & -0.00074788 & 0.00018916 & 0.00017149 \\
  -0.00074788 & 0.00097256 & -0.00019211 & -0.00017223 \\
  0.00018916 & -0.00019211 & 0.00070551 & 0.00037287 \\
  0.00017149 & -0.00017223 * 0.00037287 & 0.00033792
  \end{bmatrix}
  \end{aligned}
  $$

After the initial $\theta^{(0)}=(\gamma^{(0)},\kappa^{(0)},r^{(0)},\delta^{(0)})$ and $\Sigma_{proposal}^{(0)}$ are determined, the proposal is tuned by adjusting $\gamma_{DRAM}$ and $s_d$ in the DRAM algorithm as described in part 1 of this project. Here i use the notation $\gamma_{DRAM}$ for the tuning knob inside DRAM, and $\gamma$ for the parameter of the SIR model we are trying to infer. The covariance of the accepted samples is updated at each iteration and scaled by $s_d$ to get the covariance matrix of the first proposal, and if the first proposal is rejected, the covariance matrix is scaled further by $\gamma_{DRAM}$ for the second proposal.

Since the ideal acceptance ratio is 20-30%, I try to change $s_d$ and $\gamma_{DRAM}$ to get an acceptance ratio in this range. While doing this, I also monitor the Marginals, Autocorrelation vs lag plot, the Integrated Autocorrelation(IAC) and the visual inspection of the samples to ensure that the samples are mixing well and the proposal is tuned correctly. Specifically

- Acceptance ratio:
  - if low, increase $s_d$ and $\gamma_{DRAM}$
  - if high, decrease $s_d$ and $\gamma_{DRAM}$
- Marginals:
  - if clustering around a few points and not exploring enough, increase $s_d$ and $\gamma_{DRAM}$
- Autocorrelation:
  - if doesnt decay quickly, increase $s_d$ and $\gamma_{DRAM}$
- IAC:
  - if high, increase $s_d$ and $\gamma_{DRAM}$

For the sake of speed, to increase the acceptance ratio I preferred to decrease $s_d$ before I increased $\gamma_{DRAM}$, because if the first proposal is rejected, the trajectory has to be propagated again for the second proposal, which is computationally expensive. Hence I prefered the first proposal to be accepted more often rather than it rejecting and the 2nd proposal being accepted.

After tuning, I ended up with $s_d=2$ and $\gamma_{DRAM}=0.5$ for the final proposal.

For this specific problem, I started off with a really low acceptance ratio, but extremely strongly clustered points in the visual inspection along with slowly decaying Autocorrelation and high IAC values. Thus, I was in a dilemma, whether to increase acceptance ratio by reducing $s_d$ or if i should try and get the DRAM algorithm to explore more and lead to less clustered points. In the end, I decided to look at the posterior predictives and select the tuning parameters that gave me answers that got close to the true evolution of the disease. This makes sense to me because ultimately we are interested in seeing if our inferred $\theta$ can accurately reproduce the original evolution of the disease.

$\pagebreak$

### 4/5 Analyze your results using the same deliverables as you used in Section 1.

![Marginals for Problem 1](figs/2.B%20Problem%202_marginal.svg){width=70%}

![Visual Inspection for Problem 1](figs/2.B%20Problem%202_inspection.svg){width=75%}

$\pagebreak$

![Autocorrelation for Problem 1](figs/2.B%20Problem%202_autocorrelation.svg){width=75%}

- Integrated autocorrelation values:

  - $\gamma$: 621
  - $\kappa$: 333
  - $r$: 121
  - $\delta$: 78

- Acceptance ratio: 4%

The acceptance ratio for this problem started off very low, and I tried to increase it by decreasing $s_d$, but that caused the visual inspection plot to look even more clustered. Here we can see that all the metrics are pretty bad, but that is because I prioritized getting the posterior predictive to match the true evolution of the disease as closely as possible

When we compare the value of the true parameters with the marginals, we can see that the marginals are not exactly gaussian, especially when we look at the one for $\gamma$ and $\kappa$, As discussed before, it looks like the graph of $\frac{1}{x}$, which makes sense since both of them multiplied together should be similar to $\beta$. This problem does not just stem from the fact that we have noise in the data, but also the fact that our proposal would have been wildly different from the actual distribution of the data as we can see in the marginals.

$\pagebreak$

### 6 Plot the prior and posterior predictives of the dynamics (separately)

The true data is in blue and the predicted data is in grey

![Prior Predictive for Problem 2](figs/2.B%20Problem%202%20Prior_predictive.svg){width=80%}

![Posterior Predictive for Problem 2](figs/2.B%20Problem%202%20Posterior_predictive.svg){width=80%}

The prior predictive looks totally random, only being close to the true trajectory in the beginning while the posterior predictive actually matches the true trajectory of the evolution quite closely. This is a good sign that our model is working well and the parameters are being inferred correctly. The difference between the prior and posterior predictives is quite stark, showing the power of Bayesian inference given that after learning the parameters we can make much better predictions.

$\pagebreak$

## Please comment on the following:

- How does non-identifiability affect the Bayesian approach?

The non-identifiability makes it extremely difficult to infer the parameters, especially because the parameters $\gamma$ and $\kappa$ related to $\beta$ like $\beta = \gamma\kappa$, which makes the posterior distribution look like $\frac{1}{x}$. We have extremely low acceptance ratio, with seemingly no way to tune it without losing out in other metrics to analyse the DRAM MCMC

- In many models it may not be clear by inspection that certain parameters are non-identifiable. How can you use the Bayesian approach to probe whether this might be the case?

In such models, we can see if our analysis metrics like the ones discussed in this report go against each other in terms of tuning the MCMC. If they do, it is highly likely that the parameters are non-identifiable. We can also try to have an approach where we fix one of the parameters and only search for the others, and doing this sequentially might help us understand the non-identifiability of the parameters.

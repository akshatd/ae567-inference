---
geometry: "margin=2cm"
---

# AE567 Project 3: Filtering

# Akshat Dubey

# 1 Model

Discrete-time dynamics of the pendulum:

$$
\begin{aligned}
\text{Time steps: } & k = 0, 1, 2, \ldots, N \\
\text{State at time $k$: } & x^k = \begin{bmatrix} x_1^k \\ x_2^k \end{bmatrix} \\
\text{Noise at time $k$: } & {\bf q}^k \sim \mathcal{N}(0, {\bf Q}) \\
\text{where } & {\bf Q} = \begin{bmatrix}
\frac{q^c \Delta t^3}{3} & \frac{q^c \Delta t^2}{2} \\
\frac{q^c \Delta t^2}{2} & q^c \Delta t
\end{bmatrix} \\
\text{Dynamics: } & x^{k+1} =
\begin{bmatrix}
x_1^k + x_2^k \Delta t \\
x_2^k - g \sin(x_1^k) \Delta t
\end{bmatrix}
+ {\bf q}^k \\
\text{Measurement interval: } & \delta \\
\text{Measurement noise at time $\delta k$: } & r^{\delta k} \sim \mathcal{N}(0, R) \\
\text{Measurement at time $\delta k$: } & y^{\delta k} = \sin(x_1^{\delta k}) + r^{\delta k}
\end{aligned}
$$

For this problem,

$$
\begin{aligned}
N &= 500 \\
x^0 &= \begin{bmatrix} 1.5 \\ 0 \end{bmatrix} \\
q^c &= 0.1 \\
\Delta t &= 0.01 \\
\text{Prior: } P(x^0) &= \mathcal{N}\left(\begin{bmatrix} 1.5 \\ 0 \end{bmatrix}, \begin{bmatrix} 1 & 0 \\ 0 & 1 \end{bmatrix}\right) \\
\text{Assume } g &= 9.81 \\
\end{aligned}
$$

## 1.1 Write down the linearized extended Kalman filtering equations

Let

$$
\begin{aligned}
\Phi(x^k, \Delta t) &= \begin{bmatrix}
x_1^k + x_2^k \Delta t \\
x_2^k - g \sin(x_1^k) \Delta t
\end{bmatrix} \\
h(x^{\delta k}) &= \sin(x_1^{\delta k}) \\
x^{k+1} &= \Phi(x^k; \Delta t) + {\bf q}^k \\
y^{\delta k} &= h(x^{\delta k}) + r^{\delta k} \\
\end{aligned}
$$

In the Kalman filter, we first predict using the prior to get a prediction mean $\bar m^k$ and prediction covariance $\bar C^k$ and then update using the measurement to get a posterior mean $m^k$ and posterior covariance $C^k$. The posterior for time $k$ then becomes the prior for time $k+1$. In this problem, we only get measurements every $\delta$ timesteps, so we will keep predicting the dynamics using the prediction at the previous timestep and use that as the update till we get a new measurement. Since we have a nonlinear model, we need to linearize the dynamics and measurement to implement the extended Kalman filter. Propagating the dynamics forward (prediction) requires the update at the previous time step, so we linearize it around $m^{k-1}$. Updating our measurement needs the prediction at that time step, so we linearize it around $\bar m^{\delta k}$.

Let $A^k$ and $H^{\delta k}$ be the derivatives of the dynamics and measurements respectively

$$
\begin{aligned}
A^k &= \nabla_x \Phi(x^k; \Delta t) | _{x^k = m_{k-1}} \\
&= \begin{bmatrix}
1 & \Delta t \\
-g \cos(m_1^{k-1}) \Delta t & 1
\end{bmatrix} \\
H^{\delta k} &= \nabla_x h(x^{\delta k}) | _{x^{\delta k} = \bar m^{\delta k}} \\
&= \begin{bmatrix}
\cos(\bar m_1^{\delta k}) & 0
\end{bmatrix}
\end{aligned}
$$

### Predict

$$
\begin{aligned}
\bar m^k &= \Phi(m^{k-1}; \Delta t) \\
\bar C^k &= A^k C^{k-1} (A^k)^T + {\bf Q} \\
\end{aligned}
$$

Note that here $m^{k-1} = \bar m^{k-1}$ and $C^{k-1} = \bar C^{k-1}$ till we get a new measurement.

### Update

$$
\begin{aligned}
\mu &= h(\bar m^{\delta k}) \\
U &= \bar C^{\delta k} (H^{\delta k})^T \\
S &= H^{\delta k} \bar C^{\delta k} (H^{\delta k})^T + R \\
m^{\delta k} &= \bar m^{\delta k} + U S^{-1} (y^{\delta k} - \mu) \\
C^{\delta k} &= \bar C^{\delta k} - U S^{-1} U^T
\end{aligned}
$$

## 1.2 Write down all the integrals required for Gaussian filtering, plug-in the equations given above and simplify any integrals that are analytically tractable

Let all measurements from time $a$ to $b$, at intervals of $\delta$ = $y^{a:b}$

Then, Gaussian filtering equations are the following:

### Predict

Here, we are trying to compute the distribution of the predicted state of the pendulum at time $k$ using the measurements till time $k-1$. This distribution can be approximated as a normal distribution with a mean and covariance.

$$
P(x^k|y^{0:k-1}) = \mathcal{N}(\bar m^k, \bar C^k)
$$

The prediction mean $\bar m^k$ here is just the expectation of the dynamics at time $k$ given the measurements till time $k-1$. Hence, we can just integrate(or "sum") over all possibilities of the dynamics using $\Phi(x^{k-1}; \Delta t)$ as the value and $P(x^{k-1}|y^{0:k-1})$ as the probability of that value.
The prediction covariance $\bar C^k$ is the covariance of the dynamics at time $k$ given the measurements till time $k-1$. Hence, we can just find the covariance over all possibilities of the difference of the dynamics from the mean using $(\Phi(x^{k-1}; \Delta t) - \bar m^k)$ and use its "square" as the value and $P(x^{k-1}|y^{0:k-1})$ as the probability of that value.

$$
\begin{aligned}
\bar m^k &= \mathbb{E}[x^k|y^{0:k-1}] \\
 & \text{rewriting $x^k$ using the dynamics and the noise} \\
 &= \mathbb{E}[\Phi(x^{k-1}; \Delta t) + {\bf q}^k|y^{0:k-1}] \\
 & \text{representing the expectation as an integral} \\
 &= \int \Phi(x^{k-1}; \Delta t) P(x^{k-1}|y^{0:k-1}) dx^{k-1} \\
 & \text{substituting the dynamics and distribution} \\
 &= \int
\begin{bmatrix}
x_1^{k-1} + x_2^{k-1} \Delta t \\
x_2^{k-1} - g \sin(x_1^{k-1}) \Delta t
\end{bmatrix}
\mathcal{N}(x^{k-1}; m^{k-1}, C^{k-1}) dx^{k-1} \\
 & \text{using linear gaussian multiplication wherever possible} \\
 &= \begin{bmatrix}
m_1^{k-1} + m_2^{k-1} \Delta t \\
m_2^{k-1} - g \Delta t \int \sin(x_1^{k-1}) \mathcal{N}(x^{k-1}; m^{k-1}, C^{k-1}) dx_1^{k-1} \\
\end{bmatrix} \\
\bar C^k &= \mathbb{C}ov[x^k, x^k|y^{0:k-1}] \\
 & \text{rewriting $x^k$ using the dynamics and the noise} \\
 &= \mathbb{C}ov[\Phi(x^{k-1}; \Delta t) + {\bf q}^k, \Phi(x^{k-1}; \Delta t) + {\bf q}^k|y^{0:k-1}] \\
 & \text{pulling out the covariance of ${\bf q}^k$} \\
 &= \mathbb{C}ov[\Phi(x^{k-1}; \Delta t), \Phi(x^{k-1}; \Delta t)] + {\bf Q} \\
 & \text{representing the expectation of the covariance as an integral} \\
 &= \int (\Phi(x^{k-1}; \Delta t) - \bar m^k) (\Phi(x^{k-1}; \Delta t) - \bar m^k)^T P(x^{k-1}|y^{0:k-1}) dx^{k-1} + {\bf Q} \\
 & \text{substituting the dynamics and distribution} \\
 &= \int
\begin{bmatrix}
x_1^{k-1} + x_2^{k-1} \Delta t - \bar m_1^k \\
x_2^{k-1} - g \sin(x_1^{k-1}) \Delta t - \bar m_2^k
\end{bmatrix}
\begin{bmatrix}
x_1^{k-1} + x_2^{k-1} \Delta t - \bar m_1^k & \\
x_2^{k-1} - g \sin(x_1^{k-1}) \Delta t - \bar m_2^k
\end{bmatrix} ^T
\mathcal{N}(x^{k-1}; m^{k-1}, C^{k-1}) dx^{k-1} + {\bf Q} \\
\end{aligned}
$$

### Update

In this step, we will use the new measurement to update the probability distribution of the state of the pendulum at time $\delta k$. Since we have another variable now, $y^{\delta k}$, we need to use the joint distribution of $x^{\delta k}$ and $y^{\delta k}$ to find the updated mean and covariance. If we assume this joint distribution is gaussian, we get

$$
\begin{aligned}
P(x^{\delta k}, y^{\delta k}|y^{0:k-1}) &= \mathcal{N}\left(\begin{bmatrix} \bar m^{\delta k} \\ \mu \end{bmatrix}, \begin{bmatrix} \bar C^{\delta k} & U \\ U^T & S \end{bmatrix}\right) \\
\end{aligned}
$$

where we have the following new elements in addition to the prediction mean and covariance:

$$
\begin{aligned}
\text{expected value of the new measurement: } & \mu \\
\text{cross-covariance between the state and the new measurement: } & U \\
\text{covariance of the new measurement: } & S \\
\end{aligned}
$$

which can be computed as follows:

$$
\begin{aligned}
\mu &= \mathbb{E}[y^{\delta k}] \\
 & \text{rewriting $y^{\delta k}$ using the measurement model} \\
 &= \mathbb{E}[h(x^{\delta k}) + r^{\delta k}] \\
 & \text{given that $r^{\delta k}$ has zero mean} \\
 &= \mathbb{E}[h(x^{\delta k})] \\
 & \text{representing the expectation as an integral} \\
 &= \int h(x^{\delta k}) P(x^{\delta k} | y^{0:k-1}) dx^{\delta k} \\
 & \text{substituting the measurement model and distibution} \\
 &= \int \sin(x_1^{\delta k}) \mathcal{N}(x^{\delta k}_1; \bar m^{\delta k}_1, \bar C^{\delta k}_{11}) dx^{\delta k}_1 \\
U &= \mathbb{C}ov[x^{\delta k}, y^{\delta k}] \\
 & \text{rewriting $y^{\delta k}$ using the measurement model} \\
 &= \mathbb{C}ov[x^{\delta k}, h(x^{\delta k}) + r^{\delta k}] \\
 & \text{representing the expectation of the covariance as an integral} \\
 &= \int (x^{\delta k} - \bar m^{\delta k}) (h(x^{\delta k}) - \mu)^T P(x^{\delta k} | y^{0:k-1}) dx^{\delta k} \\
 & \text{substituting the measurement model and distibution} \\
 &= \int (x^{\delta k} - \bar m^{\delta k}) (\sin(x_1^{\delta k}) - \mu)^T \mathcal{N}(x^{\delta k}; \bar m^{\delta k}, \bar C^{\delta k}) dx^{\delta k} \\
S &= \mathbb{C}ov[y^{\delta k}, y^{\delta k}] \\
 & \text{rewriting $y^{\delta k}$ using the measurement model} \\
 &= \mathbb{C}ov[h(x^{\delta k}) + r^{\delta k}, h(x^{\delta k}) + r^{\delta k}] \\
 & \text{pulling out the covariance of $r^{\delta k}$} \\
 &= \mathbb{C}ov[h(x^{\delta k}), h(x^{\delta k})] + R \\
 & \text{representing the expectation of the covariance as an integral} \\
 &= \int (h(x^{\delta k}) - \mu) (h(x^{\delta k}) - \mu)^T P(x^{\delta k} | y^{0:k-1}) dx^{\delta k} + R \\
 & \text{substituting the measurement model and distibution} \\
 &= \int (\sin(x_1^{\delta k}) - \mu) (\sin(x_1^{\delta k}) - \mu)^T \mathcal{N}(x^{\delta k}_1; \bar m^{\delta k}_1, \bar C^{\delta k}_{11}) dx^{\delta k}_1 + R \\
\end{aligned}
$$

To get the updated mean and covariance of our estimate back from the joint distribution, we can use the conditioning formula for gaussians:

$$
\begin{aligned}
m^{\delta k} &= \bar m^{\delta k} + U S^{-1} (y^{\delta k} - \mu) \\
C^{\delta k} &= \bar C^{\delta k} - U S^{-1} U^T
\end{aligned}
$$

$\pagebreak$

# 2 Gaussian Filtering

The state trajectories and data measured from the dynamics is plotted below:

![States and data](figs/Pendulum%20trajectories%20and%20data.svg)

$\pagebreak$

## 2.1 Extended Kalman Filter

The state estimates and their $\pm 2\sigma$ bounds are plotted below, with the measurement interval($\delta$), noise(R) and mean-squared errors(MSE) in the titles:

![EKF estimates](figs/Pendulum%20trajectories%20with%20EKF.svg)

$\pagebreak$

## 2.2 Unscented Kalman Filter

The Unscented Kalman Filter uses Gaussian integration to evaluate the integrals in the prediction and update steps. To make things easier, we consider integration with respect to a standard Gaussian measure, $u \sim \mathcal{N}(0, I)$, which we can transform back to the state by

$$
\begin{aligned}
x &= m + \sqrt{q} u \\
\text{where }& \\
m &: \text{mean of the state} \\
q &: \text{covariance of the state} \\
\end{aligned}
$$

So, in any of the equations where you have an integral over some function $f$ of the state $x$ multiplied by a probability, you could rewrite it as

$$
\begin{aligned}
\int f(x) P(x) dx &= \int f(m + \sqrt{q} u) \mathcal{N}(u; 0, I) du \\
\end{aligned}
$$

For prediction, this integration involves integrating over the probabilities of the state in the previous update step,

$$
P(x^{k-1}|y^{0:k-1}) = \mathcal{N}(m^{k-1}, C^{k-1})
$$

For the update step, this integration involves integrating over the probabilities of the state in the current prediction step,

$$
P(x^{\delta k}|y^{0:k-1}) = \mathcal{N}(\bar m^{\delta k}, \bar C^{\delta k})
$$

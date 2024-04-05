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
H^{\delta k} &= \nabla_x h(x^{\delta k}) | _{x^{\delta k} = \bar m^{\delta k}} \\
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

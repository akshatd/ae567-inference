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
\bar m^k &\approx \Phi(m^{k-1}; \Delta t) \\
\bar C^k &\approx A^k C^{k-1} (A^k)^T + {\bf Q} \\
\end{aligned}
$$

Note that here $m^{k-1} = \bar m^{k-1}$ and $C^{k-1} = \bar C^{k-1}$ till we get a new measurement.

### Update

$$
\begin{aligned}
\mu &\approx h(\bar m^{\delta k}) \\
U &\approx \bar C^{\delta k} (H^{\delta k})^T \\
S &\approx H^{\delta k} \bar C^{\delta k} (H^{\delta k})^T + R \\
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

The prediction mean $\bar m^k$ here is just the expectation of the state at time $k$ given the measurements till time $k-1$. Hence, we can just marginalize/integrate(or "sum") over all possibilities of the states using $\Phi(x^{k-1}; \Delta t)$ as the state and $P(x^{k-1}|y^{0:k-1})$ as the probability of that state.
The prediction covariance $\bar C^k$ is the covariance of the state at time $k$ given the measurements till time $k-1$. Hence, we can just find the covariance over all possibilities of the difference of the state from the prediction mean using $(\Phi(x^{k-1}; \Delta t) - \bar m^k)$ and use its "square" as the value and $P(x^{k-1}|y^{0:k-1})$ as the probability of that value.

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

In this step, we will use the new measurement with bayes rule to update the probability distribution of the state of the pendulum at time $\delta k$. Since we have the measurement now, $y^{\delta k}$, we need to use the joint distribution of $x^{\delta k}$ and $y^{\delta k}$ to find the updated mean and covariance. If we assume this joint distribution is gaussian, we get

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

To get the updated mean and covariance of our estimate back from the joint distribution, we can use the conditioning formula for multivariate Gaussians:

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

For nonlinear Gaussian filtering, we are trying to solve the equations given in part 1.2 assuming both the prediction and update distributions are Gaussian. If the dynamics of the system were linear, we could have simply used standard linear Gaussian operations to solve them, and then used the conditioning equations to get our estimate update, but in this case they are non-linear so the problem is not straightforward. Hence, we have to use one of these strategies to solve them:

### 1. Pretend that the system is linear

In this case, we approximate the nonlinear system as a linear system or a 1st order Taylor Series approximation. For the prediction step, we linearize the dynamics around the previous posterior mean $m_{k-1}$ since that is the last available estimate of the system. For the update step, we linearize the measurement model around the current predicted mean $\bar m_k$ since that is the last estimate of the state of the system. After this, applying marginalization for prediction and bayes rule for update becomes a simple linear Gaussian operation. This approach is used in the Extended Kalman Filter (2.1), and the equations and explanation has already been covered in 1.1.

### 2. Use Gaussian integration to marginalize

In this case, we keep the non-linearity of the system, but this means we cannot use linear Gaussian operations to marginalize and compute the expectation and covariance, but we actually need to integrate the probability distribution of the state. Here, we approximate the distributions to be Gaussian, and then use Gaussian integration to compute the integrals required for the prediction and update steps. This approach is used in the Unscented Kalman Filter (2.2) and Gauss-Hermite Kalman Filter (2.3).

To start with Gaussian integration, we first consider integration with respect to a standard Gaussian measure, $u \sim \mathcal{N}(0, I)$, which we can transform back to the state by

$$
x = \mu + \sqrt{\Sigma} u , x \sim \mathcal{N}(\mu, \Sigma)
$$

So, in any of the equations where you have an integral over some function $f$ of the state $x$ multiplied by a probability, you could rewrite it as

$$
\begin{aligned}
\int f(x) P(x) dx &= \int f(\mu + \sqrt{\Sigma} u) \mathcal{N}(u; 0, I) du \\
\end{aligned}
$$

For prediction, this integration involves integrating over the probabilities of the state in the previous update step, with the functions being the prediction mean and covariance:

$$
\begin{aligned}
f(x) &: & \Phi(x^{k-1}; \Delta t) \\
P(x) &: P(x^{k-1}|y^{0:k-1}) =& \mathcal{N}(m^{k-1}, C^{k-1}) \\
\end{aligned}
$$

For update, this integration involves integrating over the probabilities of the state in the current prediction step, and the functions are the various means and covariances of the update step:

$$
\begin{aligned}
f(x) &: \text{one of} \\
&& h(x^{\delta k}) \\
&& (x^{\delta k} - \bar m^{\delta k}) (h(x^{\delta k}) - \mu)^T \\
&& (h(x^{\delta k}) - \mu) (h(x^{\delta k}) - \mu)^T \\
P(x) &: P(x^{\delta k}|y^{0:k-1}) =& \mathcal{N}(\bar m^{\delta k}, \bar C^{\delta k})
\end{aligned}
$$

To compute these integrals, we use a sum of the transformed points $u$ multiplied by their weights $w$, where the weights are chosen such that the sum approximates the integral. The Unscented Kalman Filter and Gauss-Hermite Kalman Filter use different sets of points and weights, defined by their own "quadrature" rules to approximate the integrals, and the equations and explanation for both will be covered in 2.2 and 2.3.

Therefore, assuming we have $U$ points and weights, the integral can be approximated as

$$
\int f(x) P(x) dx \approx \sum_i^{U} f(\mu + \sqrt{\Sigma} u^i) w^i
$$

$\pagebreak$

## 2.1 Extended Kalman Filter

The state estimates and their $\pm 2\sigma$ bounds are plotted below, with the measurement interval($\delta$), noise(R) and mean-squared errors(MSE) in the titles:

![EKF estimates](<figs/Pendulum%20trajectories%20(EKF).svg>)

We can see that the Extended Kalman filter manages to estimate the state of the pendulum quite well, with the estimates following the true states closely when the noise is low and the measurements are taken frequently. The MSE is also low compared to the range of values of the states. We can see that increasing noise worsens the performance of the filter a lot more than decreasing the number of measurements. In the case that the noise is very high, the state estimate veers off the true state by quite a bit. In some instances when the realization of the noise was very large, the EKF can give quite bad estimates compared to the real states, with the trajectory no longer following the sinusoid that we expect from the pendulum.

## 2.2 Unscented Kalman Filter (order 3)

The 3rd order Unscented Kalman Filter is accurate up to third order integrals and uses a $2d+1$ quadrature rule, where $d$ is the number of dimensions in the state. It starts with the mean of the state and assigning 2 points per dimension in the state, such that they cover both the positive and negative sides of the mean in the dimension. We start with a set of points be $\bf{M}$, and in 2 dimensions, it would look like

$$
\begin{aligned}
\bf{M} &= \begin{Bmatrix}
\begin{pmatrix} 0 \\ 0 \end{pmatrix},
\begin{pmatrix} 1 \\ 0 \end{pmatrix},
\begin{pmatrix} 0 \\ 1 \end{pmatrix},
\begin{pmatrix} -1 \\ 0 \end{pmatrix},
\begin{pmatrix} 0 \\ -1 \end{pmatrix}
\end{Bmatrix}
\end{aligned}
$$

The Unscented Kalman Filter then scales these points to enable a better approximation of the integrals yielding us quadrature points $u^i$, and then finally associates weights to calculate the mean $w^i_m$ and covariance $w^i_C$ in our prediction and update steps. The total sum of all the function values at these transformed points multiplied with the weights gives the integral.

The Unscented Kalman Filter defines the points and weights as follows:

$$
\begin{aligned}
\text{Tuning parameters: } & \alpha, \beta, \kappa \\
\text{Dimension of the state: } & d = 2 \\
\text{Scaling factor: } & \lambda = \alpha^2 (d + \kappa) - d \\
\text{Quadrature Points: } & u^i =  \sqrt{d + \lambda} \hat u^i, \hat u^i \in \bf{M} \\
\text{Mean Weights: } & w^i_m = \left\{ \begin{array}{ll}
\frac{\lambda}{d + \lambda} & \mbox{$i = 0$} \\
\frac{1}{2(d + \lambda)} & \mbox{otherwise}
\end{array} \right. \\
\text{Covariance Weights: } & w^i_C = \left\{ \begin{array}{ll}
\frac{\lambda}{d + \lambda} + (1 - \alpha^2 + \beta) & \mbox{$i = 0$} \\
\frac{1}{2(d + \lambda)} & \mbox{otherwise}
\end{array} \right. \\
\end{aligned}
$$

Then the Unscented Kalman Filter approximation is as follows:

$$
\begin{aligned}
\int f(x) P(x) dx &\approx \sum_{i=1}^{2d+1} f(\mu + \sqrt{\Sigma} u^i)w^i \\
\end{aligned}
$$

Where we use $w^i = w^i_m$ when calculating the mean and $w^i = w^i_C$ when calculating the covariance. These steps for calculating the sigma points and weights has been implemented in the function `unscented_points`, which returns all the sigma points and weights that can be used directly in the relevant integrals for the prediction and update steps, where the functions and distributions are as described in section 2.

The results for the Unscented Kalman Filter are plotted below, and we can see that it performs slightly better than the Extended Kalman Filter in the worst case, with the estimates following the true states more closely. The MSE is also slightly lower than the EKF in the worst case when the noise is high. However we can see that it takes a lot longer to converge to the true state compared to the EKF, and the estimate uncertainty in the beginning fifth of the time period is quite bad, even in the case of low noise and frequent measurements.

![UKF estimates](<figs/Pendulum%20trajectories%20(UKF).svg>)

$\pagebreak$

## 2.3 Gauss-Hermite Kalman Filter (order 3 and 5)

The Gauss-Hermite Kalman Filter uses Gaussian Quadratures with a $O^d$ point quadrature rule and is exact for polynomials up to order $2O-1$ where $O$ is the order and $d$ is the dimension of the state. We can use the Galub-Welsch algorithm that finds the weights and sigma points for the quadrature through the solution of an eigenvalue problem associated with orthogonal polynomials.

A set of polynomials ${p_i (x)}^m_{i=0}$ is orthogonal with respect to $w(x)$ if

$$
\int p_i(x) p_j(x) w(x) dx = 0, \text{when } i \neq j
$$

and all orthogonal polynomials obey some three-term recurrence relation

$$
\begin{aligned}
p_n(x) &= (a_n x + b_n) p_{n-1}(x) - c_n p_{n-2}(x) \\
\text{where }& p_{-1}(x) = 0, p_0(x) = 1
\end{aligned}
$$

There are many such families of polynomials, but here we use the Hermite polynomials, can be found using a recursive formula

$$
\begin{aligned}
H_{n+1}(x) &= xH_n(x) - nH_{n-1}(x) \\
H_n{x} &= x H_{n-1}(x) - (n-1)H_{n-2}(x) \\
\text{giving us: }& \\
a_n &= 1 \\
b_n &= 0 \\
c_n &= n-1 \\
\end{aligned}
$$

The Hermite polynomials are orthogonal with respect to the standard normal distribution, which is what we need since we are integrating with respect to a standard normal measure for our marginalization in the update and prediction steps.

$$
\begin{aligned}
\int H_i(u) H_j(u) \mathcal{N}(u; 0, 1) du &= i! \delta_{ij} \\
\text{where } \delta_{ij} &= \left\{ \begin{array}{ll}
1 & \mbox{if $i = j$} \\
0 & \mbox{if $i \neq j$}
\end{array} \right.
\end{aligned}
$$

So now that we have our orthogonal polynomials, we can move towards our eigenvalue problem, tackled in `gh_oned`. We setup a symmetric tridiagonal matrix $T_{O \times O}$ according to

$$
\begin{aligned}
T_{O \times O} &= \begin{bmatrix}
\alpha_1 & \beta_1 & 0 & 0 & \ldots & 0 \\
\beta_1 & \alpha_2 & \beta_2 & 0 & \ldots & 0 \\
0 & \beta_2 & \alpha_3 & \beta_3 & \ldots & 0 \\
. & . & . & . & . & . \\
\ldots & \ldots & \ldots & \ldots & \ldots & \beta_{O-1} \\
. & . & . & 0 & \beta_{O-1} & \alpha_O \\
\end{bmatrix} \\
\text{where } & \\
\alpha_i &= -\frac{b_i}{a_i} \\
\beta_i &= \sqrt{\frac{c_{i+1}}{a_i a_{i+1}}} \\
\end{aligned}
$$

Next we take the do an eigenvalue decomposition of this matrix to get the eigenvalues and eigenvectors. The eigenvalues are the points $u^i$. The weights $w^i$ are calculated using a constant times the first element of each orthonormal eigenvector squared. The constant is chosen such that the sum of the weights is 1.

At this point we have our $O$ points and weights with respect to a standard gaussian in 1 dimension. We extend this to $d$ dimensions by duplicating the points in each dimension such that we get $O^d$ points. The weights are then calculated by multiplying the weights of each dimension together. This is done in the function `tensorize`

To put it more concretely, if we have 2 dimensions with an order of 3, we would have $3^2 = 9$ points and weights. The points $u^{ij}$ would be the cartesian product of the 1D points $u^i$, and the weights would be the product of the 1D weights.

We finally have our set of points and weights, so we just need to transform these points (done in `rotate_points` function) to our target distribution and do a weighted sum. Following the notation in section 2, we should be able to approximate an integral in 2 dimensions for our problem as follows

$$
\begin{aligned}
\int f(x) P(x) dx & \approx \sum_{o_1=1}^O \sum_{o_2=1}^O f(\mu + \sqrt{\Sigma} u^{o_1o_2}) w^{o_1o_2} \\
\text{where } & \text{there are $O$ total points in the first dimension} \\
o_d &=  \text{: the index of the point in the $d^{th}$ dimension} \\
u^{o_1o_2} &= \begin{pmatrix} u_1^{o_1} \\ u_2^{o_2} \end{pmatrix} \\
w^{o_1o_2} &= w^{o_1} w^{o_2}
\end{aligned}
$$

Now that we are back in the familiar notation of approximating integrals, we can substitute the functions and distributions from the prediction and update steps as described in section 2 to get the Gauss-Hermite Kalman filter approximation for both cases when $O = 3$ and $O = 5$.

The results for both the 3rd and 5th order Gauss-Hermite Kalman Filter are plotted below. I chose to do both because the effort to make them both work was trivial and it would give me better insights into what happens when we increase the order of approximation for such a low-dimensional problem. We can see that it performs the best out of all the filters in the worst case, with the estimates that follow the true states closer than the rest. The MSE is also the lowest among all the filters in the worst case when the noise is high. We can see that it takes a lot longer to converge to the true state compared to the EKF, and the estimate uncertainty in the beginning fifth of the time period is quite bad, even in the case of low noise and frequent measurements. We can also see that the difference between the 3rd and 5th order GHKF is not that much, with the 5th order GHKF having slightly better estimates and lower MSE.

![3rd order GHKF estimates](<figs/Pendulum%20trajectories%20(3rd%20order%20GHKF).svg>)

![5th order GHKF estimates](<figs/Pendulum%20trajectories%20(5th%20order%20GHKF).svg>)

$\pagebreak$

## 2.4 Comparison

### Robustness

While

### Computational complexity and timing

### Accuracy

$\pagebreak$

# 3 Particle Filtering

In Particle filtering, we are still trying to solve the same problem of estimating the state of the pendulum by assimilating data and forming a posterior, but here the big difference compared to the Gaussian Integration based Filters is that we do not assume that the filtering distribution is gaussian. Particle Filtering uses importance sampling to sample from an arbitrary distribution, which makes it quite powerful since it can approximate any distribution. The basic idea is to represent the filtering distribution as a set of particles, each with a weight, and then propagate these particles through the dynamics and measurement model to get the new state and weight of the particles. The weights are then normalized to get the posterior distribution. The particles are resampled based on their weights to avoid degeneracy, and the process is repeated for every new measurement. This sequential sampling of the posterior is why Particle Filtering is also called Sequential Importance Sampling.

## Dynamics as the proposal

TODO: explain

## EKF strategy for approximating the proposal

TODO: explain

## 3.1 Running the particle filters

The 3 combinations of $\delta$ and $R$ chosen are $(\delta, R) = {(5, 0.001), (40, 0.1), (40, 1)}$. These were chosen because they include both the extremes plus a middle ground in terms of the $\pm 2 \sigma$ bounds of the state estimates for the various filters in section 2. This is to make sure that we have a good idea of how the particle filter performs in the best case scenario and also when it breaks in an unrealistically high noise scenario.

### Dynamics as the proposal

![Dynamics proposal: Particle filter filtering distribution](figs/Filtering%20Distr:%20PF%20with%20dynamics%20proposal.svg)

We can see that here the particle filter tracks the true state for both the low and medium noise case pretty well. If you look at the estimation performance in the worst case, it is actually not that bad considering the data points are nowhere close to the actual dynamics of the system. It is performing similar to the Gaussian Integration based Kalman Filters, which is a good sign but is is better in the sense that it is able to narrow down the uncertainty a lot faster than the Kalman Filters, which have large uncertainty for almost the first quarter of the run even in low noise scenarios. One curious thing I noticed was that for a ver low value of measurement noise $r$, I would get a lot of particle degeneracy and resampling, while for high noise that did not seem to happen. This actually makes sense since for a very low noise, the distribution will be pretty tight compared to the noise in the dynamics, and since the proposal is only based on the dynamics, only a very small number of particles will have a high weight based on the likelihood from the data, leading to a lot of resampling.

![Dynamics proposal: Effective sample size](figs/ess_PF%20with%20dynamics%20proposal.svg)

### EKF strategy for approximating the proposal

TODO: make this plot and fill this up

## 3.2

Here we create a reference simulation using the dynamics proposal for the 3 chosen combinations of $\delta$ and $R$ to compare the performance of the Particle Filter with the Gaussian Integration based Kalman Filters. The joint posterior of the states is drawn at t = [0, 1.25, 2.5, 3.75, 5] for the 3 cases, and the gaussian distribution approximated by EKF is overlaid on top as well to compare how good the Gaussian approximation is. I have used a million samples for each time step in the particle filter to get a good approximation of the posterior, but I have only plotted 10000 samples so that we can still see the contours of the non-gaussian and gaussian posterior clearly.

<!-- ![Joint Posterior for dynamics proposal with UKF approximation](figs/Posterior:%20PF%20with%20dynamics%20proposal.svg) -->

We can see that as the noise increases, the posterior becomes more and more non-gaussian, and the Gaussian approximation by the UKF becomes worse and worse. In the presence of low noise, the state estimates are very close to each other, and will behave similarly as time progresses, retaining their gaussian shape from the gaussian prior pretty well. However, as we increase the measurement noise, the samples stray further away from the prior mean and start getting affected a lot by the nonlinear dynamics, which is why we see that the posterior propagates to become more and more non-gaussian. Therefore, the UKF approximation is really good when there is low noise, but is totally wrong when there is high noise, which is expected since the posterior is extremely non-gaussian. In the case of medium noise, we can see that the EKF approximation is not that bad, staying close to the mean and mostly maintaining the spread of the posterior. However, in some occasions like ($\delta$=40, R=0.1, t = 2.5) the covariance is clearly wrong since the gaussian is stretched in the opposite direction compared to the true posterior.

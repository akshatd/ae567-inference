---
geometry: "margin=2cm"
---

# AE567 Project 2 Part 1: Bayesian Inference and Decisions

# Akshat Dubey

## 1 Find the source!

## 1.1

### What is uncertain in this problem?

We want the location(x, y coordinates) of the source of the scent in the 2D plane, which would require us to be able to predict how the scent dissipates in order to trace it back to the source. We are uncertain how the scent dissipates in the 2D plane, which in turn makes us uncertain about the strength of the scent at any location besides at the data points. Here is a visualization of the field of play:

![Initial set of data](figs/initial_data.svg){width=60%}

### What are you trying to learn?

How the scent dissipates with respect to any point on the 2D plane. Specifically, a distribution of the strength of the scent at any given location in the field of play. This can be defined by a covariance kernel that relates the scent at any point to the scent at any other point with a covariance matrix.

### What quantities can you use to summarize your uncertainty?

The random variable that we want to predict is the strength of the scent across the field of play. Here, I use $2\sigma$ or $\sigma^2$ of the posterior distribution(predicted distribution of scent strength) at every point in the playing field to represent its uncertainty. $2\sigma$ is used for plotting the uncertainty to be consistent with the in-class notebooks. However $\sigma^2$ is used for the acquisition function because it is the variance of the posterior distribution which is given directly by the `gpr` function, making it less computationally expensive compared to $2\sigma$ which requires a square root of the variance matrix.

## 1.2

How did you mathematically formulate this problem?

### What is your search space?

The search space is the 2D plane of the field of play where the scent is being sensed. It is normalized to $[-1, 1] \times [-1, 1]$ and discretized into a grid of $50 \times 50$ points for a total of 2500 points total in the search space.

### How is the decision with regards to the next sensor request made?

- How is it related to the uncertainty you described in the previous question?
- In general, what is the algorithm you use to determine where next to request sensor readings?
- Can you provide pseudocode of the algorithms you considered and why you considered them?

To make a decision regards to the next sensor request, specifically which new points to acquire next, I use two acquisition functions that return points that:

1. Maximize reduction in variance: using posterior variance
2. Maximize expected improvement: using posterior mean and variance

Posterior variance is the uncertainty in the predicted value of the scent at every point in the field of play as described in **1.1**. Posterior mean is the mean of all the possible scent strengths at a certain point in the field of play.
Based on the stage of data acquisition, I use one or both of these acquisition functions to decide the next sensor request.
These algorithms will be discussed in more detail in section **1.3**.

## 1.4

**NOTE: Placing section 1.4 before 1.3 because section 1.3 needs to refer to the choice of the kernel.**

### How did you choose the GP kernel (nonparametric) or basis functions/features (parametric)?

I chose nonparametric GP because for this problem, the goal is not to learn the parameters of the model which might be useful later, but to make predictions about the scent at any point so that we can arrive at the source of the scent. A nonparametric GP is more suited to this problem because we can ignore any underlying model and just go ahead and make predictions based on a covariance kernel. Specifically, we want to predict

$$
f((x, y)^{*}) \mid \text{data}
$$

Where $f((x, y)^{*})$ gives a distribution of the strength of scent at an arbitrary point $(x, y)^{*}$ in our field of play. The GP is then a distribution over functions $f$ such that for the collection of 2500 discretized points in the $50 \times 50$ field of play, we have 2500 random variables, where each random variable is defined by the distribution we get when we evaluate the distribution of functions at each of those 2500 points. This collection of 2500 random variables has a multivariate Gaussian distribution, and a GP on this collection can be completely specified with a mean $m(x,y)$ where $x,y$ is one of the 2500 points in the field of play with coordinates $x$ and $y$ and a covariance function $k((x,y), (x,y)^{\prime})$ where $(x,y)$ and $(x,y)^{\prime}$ are any two of the 2500 points in the field of play.

$$
\begin{aligned}
  m(x,y) &\equiv \mathbb{E}[f(x,y)],\\
  k((x,y), (x,y)^{\prime}) &\equiv \mathbb{C}\text{ov}[f(x,y), f((x,y)^{\prime})] \equiv \mathbb{E}[(f(x,y) - m(x,y))(f((x,y)^{\prime}) -
  m((x,y)^{\prime}))]
\end{aligned}
$$

The notation for the GP is then:

$$
  f(x,y) \sim \mathcal{GP}(m(x,y), k((x,y), (x,y)^{\prime})).
$$

The posterior predictions about the scent in any of the 2500 points on the field of play are then distributed according to $f(x,y) \mid data \sim \mathcal{N}(m_p(x,y), k_p((x,y), (x,y)^{\prime}))$ where

$$
\begin{aligned}
m_p(x,y) &= m(x,y) + k((x,y), \bar{(x,y)})\left(k(\bar{(x,y)}, \bar{(x,y)}) + \sigma^2I\right)^{-1}(\bar{s} - m(\bar{(x,y)})) \\
k_p((x,y), (x,y)^{\prime}) &= k((x,y), (x,y)^{\prime}) - k((x,y), \bar{(x,y)})\left(k(\bar{(x,y)}, \bar{(x,y)}) + \sigma^2I\right)^{-1}k(\bar{(x,y)}, (x,y)^{\prime})
\end{aligned}
$$

Where $\bar{(x,y)}$ is the collection of all the points where we have already acquired data, $\bar{s}$ is the collection of all the scent strengths at the points where we have already acquired data, and $\sigma^2$ is the variance of the noise in the data. This has been set to $1e-2$ in the code.

For this GP, we use a Gaussian prior with a mean equal to the mean of the scent data.

![Prior distribution of scent](figs/Prior_.svg){width=90%}

The kernels for nonparametric GP are chosen based on how we expect the scent to dissipate in the field of play. Three kernels were considered:

1. Polynomial
2. Squared Exponential
3. Periodic
4. Combination of Squared Exponential and Periodic

The code for the kernels needed to be modified to account for the 2 dimensional nature of the field of play. This was achieved by simply multiplying the kernels for each axis with each other. As an example, here is what the kernel for the Squared Exponential kernel looks like:

$$
k( (x, y), (x, y)^{\prime}) = \tau_1 \exp(-\frac{1}{2} (x - x^{\prime})^2 / l_1^2) \tau_2 \exp(-\frac{1}{2} (y - y^{\prime})^2 / l_2^2)
$$

### What did you do about any hyperparameters?

The hyperparameters were tuned by maximizing the log marginal likelihood of the data. The objective function is:

$$
log p(\hat s \mid \bar{(x, y)}, \theta) = -\frac{1}{2} \hat s^T(k(\bar{(x,y)}, \bar{(x,y)}) + \sigma^2I)^{-1}\hat s - \frac{1}{2} \log \mid k(\bar{(x,y)}, \bar{(x,y)}) + \sigma^2I \mid - \frac{n}{2} \log 2\pi
$$

Where $\hat s = \bar s - m(\bar{(x,y)})$. For all kernels except for the polynomial kernel, the negative of this objective function was minimized using `scipy.optimize.minimize`. For the polynomial kernel, the objective function was maximized with a global search since the hyperparameter _d_ needs to be an integer. This technique is good when we have very little data, and want to ensure that our posterior predictions are correct at the few points we actually have data for. This is the case for this problem, hence this tuning technique was used.

$\pagebreak$

#### 1. Polynomial Kernel

:

![Polynomial kernel](figs/Posterior:%20Tuned%20Poly%20Kernel_.svg){width=90%}

_Likelihood: -7.02; Tuned Hyperparameters: c: 0.22, d: 2.00_

The Polynomial kernel does not seem to be fitting very well. We can see that the posterior mean is mostly constant, with the data points having seemingly no effect on the posterior mean. It also has extremely high values in the right corners where are are no data points.

#### 2. Squared Exponential Kernel

:

![Squared Exponential kernel](figs/Posterior:%20Tuned%20Sq.Exp%20Kernel_.svg){width=90%}

_Likelihood: -4.07; Tuned Hyperparameters: $\tau$: 0.60, l: 0.27_

The Squared Exponential kernel seems to be fitting the data well. The posterior mean is close to the data points and the posterior variance is low at the data points and high at the points where there are no data points, which is expected.

$\pagebreak$

#### 3. Periodic Kernel

:

![Periodic kernel](figs/Posterior:%20Tuned%20Periodic%20Kernel_.svg){width=90%}

_Likelihood: -4.07; Tuned Hyperparameters: $\tau$: 0.60, l: 0.02, p: 82.28_

The Periodic kernel seems to be fitting the data well. The posterior mean is close to the data points and the posterior variance is low at the data points and high at the points where there are no data points, which is expected. This is very close to the fit using Squared Exponential kernel.

#### 4. Combination of Squared Exponential and Periodic Kernels

:

![Combination of Squared Exponential and Periodic Kernels](figs/Posterior:%20Tuned%20Sq.Exp%20and%20Periodic%20combined%20Kernel_.svg){width=90%}

_Likelihood: -4.25; Tuned Hyperparameters: Exponential($\tau$: 0.77, l: 0.75) Periodic($\tau$: 0.77, l: 0.82, p: 0.94)_

This combination of kernels was chosen since they had the highest log likelihood. However, the fit is not as good as the fit using either of the kernels individually. The posterior mean is high between low points which is not a good sign, and the likelihood is also lower than the likelihood of the individual kernels.

### How did you choose the parameters?

I chose nonparametric GP, hence no decision needed to be made on the parameters, only the kernels which has already been discussed earlier.

### Choice of kernel

I chose the Squared Exponential kernel because it fit the data well and the posterior mean and variance were close to the data points. The Periodic kernel also fit the data well, but the Squared Exponential kernel was chosen because there is no reason for the data to be periodic, so it did not make sense to choose it. In addition, intuitively I would expect the scent to dissipate exponentially and only have a local "effect", unlike the periodic and polynomial kernels which have a global "effect". The "effect" of a kernel here can be thought of as the decay of covariance as the distance between two points increases. The Squared Exponential kernel has a very fast decay, which is what I would expect for the scent.

## 1.3

Relatedly, list at least three possible candidates for the strategy (algorithms for requesting next points) that you considered prior to making a choice? Which one did you choose? Why? Provide details on algorithm performance such as

- (a) How did you use the probabilistic description of the map to inform your decisions?
- (b) How sensitive did you find your predictions to be to the choice of models (kernels,
  parameters, hyperparameters)? How did this impact your approach?

I considered the following strategies for requesting the next points:

1. Maximum reduction in total variance
2. Maximum Expected Improvement
3. Upper Confidence Bound

### Description of the Acquisition Functions/Algorithms/Strategies

#### 1. Maximum reduction in total variance

:

This algorithm selects the point that maximizes the reduction in the total variance of the posterior distribution. The total variance is the sum of the variances at all the points in the map. The probabilistic description used here is the posterior variance. The algorithm is as follows:

1. Takes in inputs: a kernel function, existing x and y data, the number of prediction points, and the number of new points to acquire.

2. Initializes a noise covariance matrix and a prediction points matrix.

3. Calculates the mean and covariance of the predictions using the `gpr` function.

4. Loops to acquire new points. For each prediction point required:

   - Calculates the variance of the predictions and sets a threshold at 90% of the maximum variance.

   - Identifies the viable points whose variance is above the threshold.

   - Initializes a second loop to iterate through the viable points, skipping every other point to reduce computation cost. For each viable point:

     - Adds the point back to the dataset with the predicted mean as the value at that point.

     - Computes the new posterior variance after adding the point using the `gpr` function.

     - Calculates the reduction in total posterior variance after adding the point.

     - If the reduction in variance is greater than the current maximum reduction, it updates the maximum reduction

   - Adds the point that resulted in the maximum reduction in variance to the augmented dataset.

5. After acquiring all the new points, it returns the new points

While this approach works, it is greedy and not globally optimal since it is possible to have an arrangement of points that reduces the total variance more if they are acquired together and their variance reduction was considered together. However, due to limited time, the greedy approach was chosen. The optimal approach could be implemented recursively, but this would be computationally expensive.

<!-- It could also be implemented using Voronoi decomposition to split up the space between the available points so we get a globally optimized solution for minimizing variance -->

#### 2. Maximum Expected Improvement

:

Expected Improvement is a known technique in Bayesian optimization that selects the point that maximizes the expected improvement in the posterior mean. The algorithm is as follows:

1. Takes in inputs: a kernel function, existing x and y data, the number of prediction points, the number of new points to acquire, and an exploration factor.

2. Initializes a noise covariance matrix and a prediction points matrix.

3. Calculates the mean and covariance of the predictions using the `gpr` function.

4. Loops to acquire new points. For each prediction point required:

   - Calculates the Expected Improvement (EI) for each prediction point with the exploration factor.

   - Identifies the point with the maximum EI.

   - Adds this point to the existing data.

   - Computes the new posterior mean and variance after adding the point using the `gpr` function.

5. After acquiring all the new points, it returns the new points

The function to calculate the Expected Improvement(EI) does this for each point:

1. Let

   $$
   \begin{aligned}
   \mu_p &= \text{Posterior mean at a point} \\
   \sigma_p &= \text{Posterior variance at a point} \\
   s_{best} &= \text{Max posterior mean scent} \\
   \xi &= \text{Exploration factor} \\
   \Phi &= \text{CDF of the standard normal distribution} \\
   \phi &= \text{PDF of the standard normal distribution} \\
   \end{aligned}
   $$

2. Normalize the max posterior mean with the current point
   $$
   z = \frac{\mu_p - s_{best} - \xi}{\sigma_p} \\
   $$
3. Calculate the expected improvement
   $$
   EI = \sigma_p . (z . \Phi(z) + \phi(z))
   $$

When $\xi$ is high, the algorithm is more exploratory and when $\xi$ is low, the algorithm is more exploitative. This is because when $\xi$ is high, the algorithm is more likely to select points with high variance, and when $\xi$ is low, the algorithm is more likely to select points with high posterior mean.

#### 3. Upper Confidence Bound

This algorithm selects the point that maximizes the upper confidence bound of the posterior distribution. The upper confidence bound is the sum of the mean and the standard deviation of the posterior distribution with a weight on the standard deviation to balance exploration vs exploitation. The algorithm is as follows:

1. Takes in inputs: a kernel function, existing x and y data, the number of prediction points, the number of new points to acquire, and an exploration factor.

2. Initializes a noise covariance matrix and a prediction points matrix.

3. Calculates the mean and covariance of the predictions using the `gpr` function.

4. Loops to acquire new points. For each prediction point required:

   - Calculates the Upper Confidence Bound (UCB) for each prediction point with the exploration factor.

   - Identifies the point with the maximum UCB.

   - Adds this point to the existing data.

   - Computes the new posterior mean and variance after adding the point using the `gpr` function.

5. After acquiring all the new points, it returns the new points

I did not proceed with using this algorithm because it introduced a new hyperparameter that needed to be tuned, and I was already using the exploration factor for the Expected Improvement algorithm which was working well.

### Choice of Acquisition Function/Algorithm/Strategy

My requirements for an Acquisition function changed as I got more data, so I used different Acquisition functions at different stages of data acquisition.

#### Initial Data

:

For the sake of clarity, I have repeated the plot of the initial posterior distribution with the tuned Squared Exponential kernel here.

Hyperparameters: $\tau$: 0.60, l: 0.27

Likelihood: -4.07.

![Initial Posterior Distribution](figs/Posterior:%20Tuned%20Sq.Exp%20Kernel_.svg){width=90%}

Both strategies were used to explore their usefulness, here the Expected improvement algorithm was run with an exploration factor of 2 to prefer maximum exploration. However, I decided to go with the Maximum reduction in total variance algorithm because it suggested points that were not on the edges of the field of play. This would ensure that the variance reduction due to those points would be more significant.

We can also observe both the strategies in action, as they discover new points, the viable area for the next point to be acquired reduces.

| Maximum reduction in total variance                                                           | Maximum Expected Improvement                                                                      |
| --------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------- |
| ![Variance reduction points](figs/Variance%20reduction%20for%20initial%20data.svg){width=40%} | ![Expected improvement points](figs/Expected%20improvement%20for%20initial%20data.svg){width=40%} |

![Variance reduction points on posterior](figs/Posterior:%20optimized%20Sq.Exp_Max%20reduced%20variance.svg){width=90%}

![Expected improvement points on posterior](figs/Posterior:%20optimized%20Sq.Exp_Max%20expected%20improvement.svg){width=90%}

$\pagebreak$

#### After 1st set of data

:

First, the Squared Exponential covariance kernel was re-tuned using the new data, which resulted in

Hyperparameters: $\tau$: 0.69, l: 0.28

Likelihood: -9.24.

The Hyperparameters did not change much, but the likelihood dropped quite a bit. This produced the following posterior distribution:

![Posterior distribution after 1st set of data](figs/Posterior:%20optimized%20Sq.Exp%20with%201st%20new%20data_.svg){width=90%}

We can see that due to receiving one point that was really high and another that was really low, the maximum variance has increased, and the overall likelihood of the model after tuning has reduced. This is expected since the model is now less certain about the scent at the points where we have data. It is not all bad, because now the overall variance in the posterior has reduced.

I again use both strategies to explore the new points to acquire for the 2nd set of data to get 3 points each. This time I reduced the exploration factor for the Expected improvement to 1 I noticed that both strategies proposed a point in the bottom left corner, so I decided to use points from both the strategies for a total of 5 points, expecting there to be a balance of exploration and exploitation this way.

| Maximum reduction in total variance                                                               | Maximum Expected Improvement                                                                          |
| ------------------------------------------------------------------------------------------------- | ----------------------------------------------------------------------------------------------------- |
| ![Variance reduction points](figs/Variance%20reduction%20after%201st%20new%20data.svg){width=40%} | ![Expected improvement points](figs/Expected%20improvement%20after%201st%20new%20data.svg){width=40%} |

![Variance reduction points on posterior after 1st data](figs/Posterior:%20optimized%20Sq.Exp%20with%201st%20new%20data_Max%20reduced%20variance.svg){width=90%}

![Expected improvement points on posterior after 1st data](figs/Posterior:%20optimized%20Sq.Exp%20with%201st%20new%20data_Max%20expected%20improvement.svg){width=90%}

$\pagebreak$

#### After 2nd set of data

:

Again the Squared Exponential covariance kernel was re-tuned using the new data, which resulted in

Hyperparameters: $\tau$: 0.73, l: 0.29

Likelihood: -14.32.

The Hyperparameters did not change much, but the likelihood dropped just like the previous time. This produced the following posterior distribution:

![Posterior distribution after 2nd set of data](figs/Posterior:%20optimized%20Sq.Exp%20with%202nd%20new%20data_.svg){width=90%}

We can see that with this new set of data, the overall variance in the posterior has reduced even further, but the maximum variance has actually increased. Again, we see 2 new points with a really high scent strength in the top left and bottom right corner, which makes it harder for the covariance kernel to fir the data. This is reflected in the likelihood of the model after tuning, which has reduced even further.

I again use both strategies to explore the new points to acquire for the 3rd set of data to get 3 points each. This time I reduced the exploration factor for the Expected improvement to 0.1, since this was the final set of data and I wanted to ensure that I was exploiting the data as much as possible. Which is also why I only took 1 point from the Maximum reduction in total variance algorithm and 2 points from the Maximum Expected Improvement algorithm. I also manually adjusted the point from the Maximum reduction in total variance algorithm since it proposed a point in the top right corner which would not be very useful since the variance there was still high, additionally a point in the corner would not be very helpful to predict the maximum scent location. The other 2 points from the Maximum Expected Improvement algorithm were chosen because they were close to the points where the posterior mean was high, which would help to confirm the location of the maximum scent.

| Maximum reduction in total variance                                                               | Maximum Expected Improvement                                                                          |
| ------------------------------------------------------------------------------------------------- | ----------------------------------------------------------------------------------------------------- |
| ![Variance reduction points](figs/Variance%20reduction%20after%202nd%20new%20data.svg){width=40%} | ![Expected improvement points](figs/Expected%20improvement%20after%202nd%20new%20data.svg){width=40%} |

![Variance reduction points on posterior after 2nd data](figs/Posterior:%20optimized%20Sq.Exp%20with%202nd%20new%20data_Max%20reduced%20variance.svg){width=90%}

![Expected improvement points on posterior after 2nd data](figs/Posterior:%20optimized%20Sq.Exp%20with%202nd%20new%20data_Max%20expected%20improvement.svg){width=90%}

$\pagebreak$

#### After 3rd set of data

:

The Squared Exponential covariance kernel was re-tuned using the new data, which resulted in

Hyperparameters: $\tau$: 0.76, l: -0.25

Likelihood: -17.15

The Hyperparameters did not change much, but the likelihood dropped just like the previous times. Interestingly, the l Hyperparameter was negative this time. This would not affect the covariance calculation since it is squared, so I kept it as it was. This produced the following posterior distribution:

![Posterior distribution after 3rd set of data](figs/Posterior:%20optimized%20Sq.Exp%20with%203rd%20new%20data_.svg){width=90%}

### Sensitivity of the Algorithm to the Choice of Models

TODO: Add graphs for these algos as they change with param tuning for sq exp kernel for the first set of data

## 1.5, 1.6

What is your final prediction about the scent location?
TODO:

- plot predictions for each stage with a box around the posterioir mean
- graph of probability of max scent location being within the box of local max
- also graph 95%ci for each stage
- plot of the final 2 locations using the posterior mean max
- graph of how this probability changes with each stage
- give answers using post the mean and the maximum probability that u calculate in the 95% ci graph

## References

- https://www.cs.cornell.edu/courses/cs4787/2019sp/notes/lecture16.pdf
- https://ekamperi.github.io/machine%20learning/2021/06/11/acquisition-functions.html

---
geometry: "margin=2cm"
---

# AE567 Project 2 Part 1: Bayesian Inference and Decisions

# Akshat Dubey

## 1 Find the source!

## 1.1

- TODO:
  - Intentionally chosen few contours to make the differences more obvious.
  - didnt choose periodic exponential kernel because no reason to believe the scent would be periodic, it should actually be dissipating exponentially. so exponential kernel is more appropriate.
  - tune hyperparams by maximizing likelihood because useful in low data regime

_You must be as specific as possible, you cant just say “my uncertainty is represented as variance”. Variance of what? What is the random variable you are talking about?_

- What is uncertain in this problem?

  - We want the location(x, y coordinates) of the source of the scent in the 2D plane, which would require us to be able to predict how the scent dissipates in order to trace it back to the source. We are uncertain how the scent dissipates in the 2D plane, which in turn makes us uncertain about the strength of the scent at any location besides at the data points.

- What are you trying to learn?

  - The model of dissipation of the scent in the 2D plane. Specifically, a model that consists of a covariance matrix defined by a covariance kernel that relates the scent at any point to the scent at any other point with a covariance matrix.

- What quantities can you use to summarize your uncertainty?

  - $2\sigma or \sigma^2$ of the posterior distribution(predicted distribution of scent strength) at every point in the playing field. $2\sigma$ is used for plotting the uncertainty to be consistent with the in-class notebooks. However $\sigma^2$ is used for the acquisition function because it is the variance of the posterior distribution which is given directly by the `gpr` function, making it less computationally expensive compared to $2\sigma$ which requires a square root of the variance matrix.

## 1.2

- How did you mathematically formulate this problem? Please be as specific as possible, to the extent that your results would be reproducible. For example: what is your search space? How is the decision with regards to the next sensor request made? How is it related to the uncertainty you described in the previous question? In general, what is the algorithm you use to determine where next to request sensor readings?

  - TODO

- Can you provide pseudocode of the algorithms you considered and why you considered them?

  - TODO

## 1.3

- Relatedly, list at least three possible candidates for the strategy (algorithms for requesting next points) that you considered prior to making a choice? Which one did you choose? Why? A decision making strategy is an algorithm which takes in available information and outputs a sensor request. Provide details on algorithm performance such as
  - (a) How did you use the probabilistic description of the map to inform your decisions?
  - (b) How sensitive did you find your predictions to be to the choice of models (kernels,
    parameters, hyperparameters)? How did this impact your approach?

## 1.4

- When discussing your regression choice, be clear on the folloiwng points. How did you choose the GP kernel (nonparametric) or basis functions/features (parametric)? How did you choose the parameters? What did you do about any hyperparameters?

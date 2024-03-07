---
geometry: "margin=2cm"
---

# AE567 Project 2 Part 1: Bayesian Inference and Decisions

# Akshat Dubey

## 1 Find the source!

## 1.1

_You must be as specific as possible, you cant just say “my uncertainty is represented as variance”. Variance of what? What is the random variable you are talking about?_

- What is uncertain in this problem?

  - We want the location(x, y coordinates) of the source of the scent in the 2D plane, which would require us to be able to predict how the scent dissipates in order to trace it back to the source. We are uncertain how the scent dissipates in the 2D plane.

- What are you trying to learn?

  - The model of dissipation of the scent in the 2D plane. Specifically, a model $M$ that consists of a set of parameters $\theta$ that can be used with an input set of coordinates (x, y) to predict the scent concentration at those coordinates in terms of a sensor reading.

- What quantities can you use to summarize your uncertainty?

  - Variance of the posterior distribution of the parameters $\theta$ of the model $M$ given the sensor readings.

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

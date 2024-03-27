---
geometry: "margin=2cm"
---

$\pagebreak$

## 2 Apply DRAM for Bayesian Inference

## 2.A Satellite Dynamics

After setting up the Satellite Dynamics, we can observe the true trajectory of the satellite vs the noisy trajectory needed for the inference problem starting at the initial state $x_0 = [-0.600, \ 0.400, \ -0.200, \ 0.663, \ 1.200, \ -1.500, \ 0.200]$

![True Satellite Dynamics](figs/True%20Satellite%20Dynamics.svg){height=80%}
![Noisy Satellite Dynamics](figs/Noisy%20Satellite%20Dynamics.svg){height=80%}

1. Problem 1: Parameters are the control coefficients (k1, k2)
2. Problem 2: Parameters are the control coefficients and a product of inertia (k1, k2, J12)

### 2.A.1 What is the likelihood model? Please describe how you determine this.

### 2.A.2 What is the form of the posterior?

### 2.A.3 How did you tune your proposal? Think carefully about what a good initial point and initial covariance could be?

### 2.A.4 Analyze your results using the same deliverables as you used in Section 1.

### 2.A.5 Plot the true parameters on your plots of the marginals for reference.

### 2.A.6 Plot the prior and posterior predictives of the dynamics (separately):

for some prior/posterior samples, run the dynamics and plot them in a transparent light gray on top of your “truth” dynamics and data. These are essentially the probabilistic predictions of your model before and after you have accounted for the data. How do they look compared to the “truth.” How does this plot differ between the prior and posteriors?

### Please comment on the following:

- What is the difference between the two parameter inference problems?
- How does the posterior predictive change?
- Are there any notable differences?

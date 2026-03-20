Proximal Path-Specific Inference:

This repository provides implementation codes for Proximal Path-Specific Effects (PSE) estimation. 

We address the challenge of estimating the effect along a specific pathway (e.g., $A \to M \to Y$) in the presence of recanting witnesses D (treatment-induced mediator-outcome confounders) and general unmeasured confounders U (directly affect A, D, M and Y).

By leveraging the proximal path-specific inference framework, we use observed covariates as proxy variables ($W$ and $Z$) to identify effects that are otherwise unidentifiable under standard sequential ignorability assumptions.

🚀 Overview

The project is divided into two main components based on the estimation strategy and programming language:

1. Semiparametric estimation (R language):
   specifies parametric models for bridge functions, to illustrate the quadruple robustness of the proximal estimator based on the semiparametric efficient influence function.

2. Nonparametric estimation (Python):

   implements the proximal debiased machine learning in the cross-fitting procedure, using regularized minimax-based optimization to solve for bridge functions without model specification.


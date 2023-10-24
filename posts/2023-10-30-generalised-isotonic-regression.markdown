---
title: Generalised Isotonic Regression - In Depth
summary: Isotonic Regression tries to fit a line/plane/hyperplane to a sequence of observations that lies as "close" as possible to the observations, while maintaining monotonicity.
library: github.com/ewal31/GeneralisedIsotonicRegression
---

Summary

* what is isotonic regression
* implement the multivariate isotonic regression algorithm from paper
* how does it work (linear programming with slack surplus)
* look at ways to improve the input by reducing constraints

Isotonic Regression tries to fit a line/plane/hyperplane to a sequence of observations that lies as "close" as possible to the observations, while maintaining monotonicity.

[@LussRossetGeneralizedIsotonicRegression]

## What is Isotonic Regression?



## How is it done historically?

- Pooled adjacent violators algorithm (PAVA)
- these implementations can be more general (i.e. not necessarily differentiable), but restricted to 1-d

## How does this paper differ?

The approach in the paper
- support convex differentiable functions

## Linear Programming

* basic form with and without constraints?
* solving

## Karush-Kuhn-Tucker (KKT) Conditions

Page 220 and finished on 226 of https://web.stanford.edu/~boyd/cvxbook/bv_cvxbook.pdf might be good to include
Could be good moving to the paper, and showing the lagrange blah blah stuff first

Page 244 of the above shows the proof as to why this sort of problem is optimal if the functions are convex
and lists the KKT conditions

## Implementing the Algorithm

## Can also be done as Quadratic Program but slower

## Handling Duplicate Points

## Many Dimensions

## Regularisation

## More generalisation?

---

### References
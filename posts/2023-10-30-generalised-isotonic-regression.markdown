---
title: Generalised Isotonic Regression - In Depth
summary: Isotonic Regression tries to fit a line/plane/hyperplane to a sequence of observations that lies as "close" as possible to the observations, while maintaining monotonicity.
library: github.com/ewal31/GeneralisedIsotonicRegression
plotly_js_file: /js/GeneralisedIsotonicRegressionPlots.js
---

Summary

* what is isotonic regression
* implement the multivariate isotonic regression algorithm from paper
* how does it work (linear programming with slack surplus)
* look at ways to improve the input by reducing constraints

Isotonic Regression tries to fit a line/plane/hyperplane to a sequence of observations that lies as "close" as possible to the observations, while maintaining monotonicity.

## What is Isotonic Regression?

* fits a piecewise-constant non-decreasing (or -increasing) function to data
* maintains monotonicity

In the classical formulation (with $L_2$ loss) where $\mathfrak{I}$ is a partial order defined on our dataset, we want to solve

$$\begin{aligned}
&\text{minimize}   &\quad \sum_i (\hat{y}_i - y_i)^2 \\
&\text{subject to} &\quad \hat{y}_i \leq \hat{y}_j, \forall \left ( i, j \right ) \in \mathfrak{I}
\end{aligned}
$$

```{=html}
<div id="UniIsoPlot">
</div>
```

## How is it done historically?

- Pooled adjacent violators algorithm (PAVA)
- these implementations can be more general (i.e. not necessarily differentiable), but restricted to 1-d

https://www.stat.umn.edu/geyer/8054/notes/isotonic.pdf

## Linear Programming

* basic form with and without constraints?
* solving

## Karush-Kuhn-Tucker (KKT) Conditions

Page 220 and finished on 226 of https://web.stanford.edu/~boyd/cvxbook/bv_cvxbook.pdf might be good to include
Could be good moving to the paper, and showing the lagrange blah blah stuff first

Page 244 of the above shows the proof as to why this sort of problem is optimal if the functions are convex
and lists the KKT conditions

Want to change to standard form

$$\begin{aligned}
&\text{minimize}   &\quad f_0(x) \\
&\text{subject to} &\quad f_i(x) \leq 0, \quad \forall i \\
&                  &\quad h_j(x) = 0,    \quad \forall j
\end{aligned}
$$

The Lagrangian of this problem incorporates the constraints into the objective function
as weighted sums over the constraint functions.

$$
L(x, \lambda, v) = f_0(x) + \sum_i \lambda_i f_i(x) + \sum_j v_i h_i(x)
$$

[@BoydVandenbergheConvexOptimization{}, pages 215]

**Two-way Partitioning Problem** (non-convex)

$$\begin{aligned}
&\text{minimize}   &\quad x^{T} W x \\
&\text{subject to} &\quad x_i^2 = 1, \forall i
\end{aligned}
$$

Constraint forces $x_i = \plusmn 1$

In words, we want to find the vector $x$ filled with $\plusmn 1$ that minimises $x^{T} W x$.

[@BoydVandenbergheConvexOptimization{}, pages 219-221]

KKT

consider optimal $x^*$, $\lambda^*$ and $v^*$ points that permit a primal and dual solution with zero duality gap.
the lagragian must be minimised at this point and therefore the gradient vanishes

$$
\Delta f_0(x^*) + \sum_i \lambda_i^* \Delta f_i (x^*) + \sum_j v_j \Delta h_j(x^*) = 0
$$

Consequently,

$$\begin{aligned}
                                                               f_i(x^*)             &\leq 0, &\quad \forall i \\
                                                               h_j(x^*)             &= 0   , &\quad \forall j \\
                                                               \lambda_i^*          &\geq 0, &\quad \forall i \\
                                                               \lambda_i^* f_i(x^*) &= 0,    &\quad \forall i \\
\Delta f_0(x^*) + \sum_i \lambda_i^* \Delta f_i(x^*) + \sum_j v_j^* \Delta h_j(x^*) &= 0
\end{aligned}
$$

out KKT conditions

[@BoydVandenbergheConvexOptimization{}, pages 244]

## How does this paper differ?

The approach in the paper
- support convex differentiable functions

The linear program that we want to solve is defined

$$
\text{min} \left\{ z^{T} x \colon x_i \leq x_j \forall (i, j) \in \mathfrak{I}, -1 \leq x_i \leq 1 \forall i \in V \right\}
$$

where

$$
z_i = \left. \frac{\partial f_i (\hat{y}_i)}{\partial \hat{y}_i} \right\vert_{w_V}
$$

[@LussRossetGeneralizedIsotonicRegression]

## Implementing the Algorithm

First build matrix in the correct form for the linear program solver.

```cpp
Eigen::SparseMatrix<int>
adjacency_to_LP_standard_form(
    const Eigen::SparseMatrix<bool>& adjacency_matrix,
    const VectorXu& considered_idxs
) {
    const uint64_t total_observations = considered_idxs.rows();
    const uint64_t total_constraints = constraints_count(
        adjacency_matrix, considered_idxs);
    const uint64_t columns = 2 * total_observations + total_constraints;

    Eigen::SparseMatrix<int, Eigen::ColMajor> standard_form(
        total_observations, columns);

    int idx = 0;
    for (Eigen::Index j = 0; j < total_observations; ++j) {
        for (
            Eigen::SparseMatrix<bool>::InnerIterator it(
                adjacency_matrix, considered_idxs(j));
            it;
            ++it
        ) {
            // add graph edges
            auto row_idx = std::find(
                considered_idxs.begin(), considered_idxs.end(), it.row());

            if (row_idx != considered_idxs.end()) {
                // row of constraint
                standard_form.insert(
                        std::distance(considered_idxs.begin(), row_idx),
                        2 * total_observations + idx) = 1;

                // col of constraint
                standard_form.insert(
                        j,
                        2 * total_observations + idx) = -1;

                ++idx;
            }
        }

        // Add slack/surplus variables (source and sink)
        standard_form.insert(j, j) = 1;
        standard_form.insert(j, j + total_observations) = -1;
    }

    standard_form.makeCompressed();

    return standard_form;
}
```

Altogether.

```cpp
Eigen::VectorX<bool>
minimum_cut(
    const Eigen::SparseMatrix<bool>& adjacency_matrix,
    const Eigen::VectorXd loss_gradient, // z in the paper
    const VectorXu considered_idxs
) {
    /*
     * min b^T x
     * A x >= c
     * x >= 0
     *
     */
    const uint64_t total_observations = considered_idxs.rows();
    const auto A = adjacency_to_LP_standard_form(adjacency_matrix, considered_idxs);
    const uint64_t total_constraints = A.cols() - 2 * total_observations;

    std::vector<double> b(A.cols());
    for (size_t i = 0; i < total_observations * 2; ++i) b[i] = 1;
    for (size_t i = 0; i < total_constraints; ++i) b[2 * total_observations + i] = 0;

    std::vector<double> c(loss_gradient.begin(), loss_gradient.end());

    const double infinity = 1.0e30; // Highs treats large numbers as infinity
    HighsModel model;
    model.lp_.num_col_ = A.cols();
    model.lp_.num_row_ = A.rows();
    model.lp_.sense_ = ObjSense::kMinimize;
    model.lp_.offset_ = 0;
    model.lp_.col_cost_ = std::move(b);
    model.lp_.col_lower_ = std::move(std::vector<double>(A.cols(), 0));
    model.lp_.col_upper_ = std::move(std::vector<double>(A.cols(), infinity));
    model.lp_.row_lower_ = c;
    model.lp_.row_upper_ = std::move(c);

    std::vector<int64_t> column_start_positions(A.cols() + 1);
    std::vector<int64_t> nonzero_row_index(A.nonZeros());
    std::vector<double> nonzero_values(A.nonZeros());
    uint64_t idx = 0;
    for (Eigen::Index j = 0; j < A.outerSize(); ++j) {
        column_start_positions[j] = idx;
        for (Eigen::SparseMatrix<int>::InnerIterator it(A, j); it; ++it) {
            nonzero_row_index[idx] = it.row();
            nonzero_values[idx] = it.value();
            ++idx;
        }
    }
    column_start_positions[column_start_positions.size()-1] = idx;

    model.lp_.a_matrix_.format_ = MatrixFormat::kColwise;
    model.lp_.a_matrix_.start_ = column_start_positions;
    model.lp_.a_matrix_.index_ = nonzero_row_index;
    model.lp_.a_matrix_.value_ = nonzero_values;

    Highs highs;
    highs.setOptionValue("solver", "simplex");
    highs.setOptionValue("simplex_strategy", 4); // Primal

    highs.passModel(model);
    highs.run();

    return Eigen::VectorXd::Map(
        &highs.getSolution().row_dual[0],
        highs.getSolution().row_dual.size()).array() > 0; // 0 left = 1 right
}
```

## Can also be done as Quadratic Program but slower

## How do we get an adjacency matrix? Handling Duplicate Points?

One option in multiple dimensions is to define a partial order, whereby a point $\bm{x}$ is less than another $\bm{y}$ if,
along each axis, it is less than or equal, i.e.

$$
\bm{x}_i \leq \bm{y}, \quad \forall i
$$

## Many Dimensions

## Regularisation

## More generalisation?

---

### References

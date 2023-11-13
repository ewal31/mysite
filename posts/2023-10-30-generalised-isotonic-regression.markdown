---
title: Generalised Isotonic Regression - In Depth
summary: Isotonic Regression tries to fit a line/plane/hyperplane to a sequence of observations that lies as "close" as possible to the observations, while maintaining monotonicity.
library: github.com/ewal31/GeneralisedIsotonicRegression
include_plotly: true
include_d3: true
js_file: /js/GeneralisedIsotonicRegressionPlots.min.js
---

```{=html}
<div id="MultiIsoPlot"></div>
```

## What is Isotonic Regression?

Isotonic Regression, similar to other regression models, tries to find a
predictive relationship between a set of features and their corresponding
observations. A method such as Linear Regression, constrains the space
of possible relationships, such that the predictions vary linearly
with changes in the feature space. Isotonic Regression, conversely,
has no restrictions on its shape apart from requiring a strict
non-decreasing (alt. non-increasing) relationship;
typically a monotonic piecewise-constant function. It can, therefore,
be a useful alternative, when it is known that a monotonic relationship
between variables exists.

In the classical formulation (with $L_2$ loss) where $\mathfrak{I}$ is
a partial order defined over our independent variables, Isotonic
Regression can be defined as the following Linear Program.

$$\begin{aligned}
&\text{minimize}   &\quad \sum_i (\hat{\bm{y}}_i - \bm{y}_i)^2 \\
&\text{subject to} &\quad \hat{\bm{y}}_i \leq \hat{\bm{y}}_j, \forall \left ( i, j \right ) \in \mathfrak{I}
\end{aligned}
$$

Throughout this post, we will discuss what a Linear Program is, and how it leads
to the algorithm discussed in [@LussRossetGeneralizedIsotonicRegression] for
solving the Isotonic Regression problem in an arbitrary number of dimensions
while supporting any convex differentiable loss function.

## Linear Programming

Linear Programming is a generic approach to solving constraint problems with a
linear objective function. The standard form of such a problem asks us to find
a vector $\bm{x} \in \mathbb{R}^p$ that takes the form

$$\begin{aligned}
&\text{minimize}   &\quad f_0(\bm{x}) \\
&\text{subject to} &\quad f_i(\bm{x}) \leq 0, \quad \forall i \\
&                  &\quad h_j(\bm{x}) = 0,    \quad \forall j
\end{aligned}
$$

Where our inequalities $f_i$ and equalities $h_i$ express a linear
relationship.

The advantage of prescribing such a form, is that generic sotware can be
written to solve any number of problems that can be expressed in such a
manner. [^GeneralKernelForm]
With $f_i$ being any linear function, we aren't even restricted
to minimisation problems and inequalities of the form $\leq 0$. Multiplying
by negative one in $f_0$ lets us morph a maximisation problem into the above
form. Similarly, a greater than inequality can be change to a less than and
add an offset can be added to allow for equalities and inequalities with values
other than zero. For example, the linear relation $x \geq 5$ could be changed to
$5 - x \leq 0$ giving us a function $f_i(x) = 5 - x$ in order to fit the above
form.

[^GeneralKernelForm]: For example, the commercial software, [MOSEK](https://www.mosek.com/),
or the the open source [HiGHS](https://highs.dev/#top), which I have used
in this implementation.

Consider, for example, a hobby photographer, that also enjoys making sourdough
bread. He might want to maximise the amount of time he spends his two favourite
activies. Unfortunately, he needs a job in order to finance his hobbies -
photography equipment can in particular be quite expensive. This leads us
to a constrained optimisation problem. As we can be either working or
enjoying one of our hobbies and we need to work some minimum amount of time
in order to have sufficient funds for our hobbies and to survive and save $s$ a bit
so we have geq value larger than 0.

Photography $p$ is 5 times more expensive, but want to do it 2 times more
than baking $b$. Will allow for 10 units of money a month. Can't spend negative
time doing something. Can only eat so much bread each weak. So maybe
limit total bread making to 5 times a week.

[^OnlineSolver]

[^OnlineSolver]: Can play around with setting up and solving some of these problems using an
online solver such as [this online optimizer](https://online-optimizer.appspot.com/?model=builtin:default.mod)

$$\begin{aligned}
&\text{maximize}   &\quad 2 p + b \\
&\text{subject to} &\quad 5 p + b \leq s \\
&                  &\quad b \leq 5 \\
&                  &\quad p \geq 0 \\
&                  &\quad b \geq 0
\end{aligned}
$$

or in standard form

$$\begin{aligned}
&\text{minimize}   &\quad - 2p - b \\
&\text{subject to} &\quad 5 p + b - s \leq 0 \\
&                  &\quad b - 5 \leq 0 \\
&                  &\quad -p \leq 0 \\
&                  &\quad -b \leq 0
\end{aligned}
$$

With such a small number of constraints and dimensions it is easy to solve this
Linear Program. We just need to plot the constraints and choose the largest
value possible.

```{=html}
<div id="LinearProgram"></div>
```

This leads to the result, that our hobby photographer should spend an hour
a week taking pictures, and 5 hours baking bread.

[^TutorialVideo]

[^TutorialVideo]: For a more in-depth introduction see this video [The Art of Linear Programming](https://www.youtube.com/watch?v=E72DWgKP_1Y)

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

$$\begin{aligned}
&\text{minimize}   &\quad z^T x \\
&\text{subject to} &\quad x_i \leq x_j, \forall (i, j) \in \mathfrak{I} \\
&                  &\quad -1 \leq x_i \leq 1, \forall i \in \mathfrak{V}
\end{aligned}
$$

where $x_i$ is a relaxed variable that selects which cluster a given sample belongs to,
$\mathfrak{V}$ is the currently considered subset and

$$
z_i = \left. \frac{\partial f_i (\hat{y}_i)}{\partial \hat{y}_i} \right\vert_{w_V}
$$

[@LussRossetGeneralizedIsotonicRegression]

```{=html}
<div id="UniIsoPlot"></div>
```

so this zTc loss value that weights what to do next
is somehow a measure of how well separated the points are either side of the most recent split
loss could be really large, but there might be little potentially gained
but this derivative measure says more something like
the points on the right of split are on mostly larger than the estimator
and the points on the left are mostly smaller.
it might be worth normalising this by the number of points though
which isn't done in the post or code

## Implementing the Algorithm

First build matrix in the correct form for the linear program solver.

```{=html}
<div id="AdjacencyMatrix"></div>
```

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


# TOOD

* what is isotonic regression
* implement the multivariate isotonic regression algorithm from paper
* how does it work (linear programming with slack surplus)
* look at ways to improve the input by reducing constraints

Isotonic Regression tries to fit a line/plane/hyperplane to a sequence of observations that lies as "close" as possible to the observations, while maintaining monotonicity.

Should probably mention some concrete applications
https://dl.acm.org/doi/abs/10.1145/1102351.1102430

Do I want to discuss other solutions at all?

- Pooled adjacent violators algorithm (PAVA)
- these implementations can be more general (i.e. not necessarily differentiable), but restricted to 1-d

https://www.stat.umn.edu/geyer/8054/notes/isotonic.pdf

---

### References

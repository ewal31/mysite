---
title: Generalised Isotonic Regression - In Depth
summary: Isotonic Regression tries to fit a line/plane/hyperplane to a sequence of observations that lies as "close" as possible to the observations while maintaining monotonicity. This post takes an in-depth look at how the paper Generalized Isotonic Regression solves this problem in an arbitrary number of dimensions while supporting any convex differentiable loss function. First, Linear Programming is introduced, and then it is shown how the partitioning-based algorithm follows from the KKT conditions.
library: github.com/ewal31/GeneralisedIsotonicRegression
include_plotly: true
include_d3: true
js_file: /js/GeneralisedIsotonicRegressionPlots.min.js
---

```{=html}
<div id="MultiIsoPlot"></div>
```

## What is Isotonic Regression?

Like other regression models, Isotonic Regression tries to find a predictive
relationship between a set of features and corresponding observations. A method
such as Linear Regression constrains the space of possible relationships such
that the predictions vary linearly with changes in the feature space; in other
words, it has a constant rate of change. Isotonic Regression, conversely, has
no restrictions on its shape apart from requiring a strict non-decreasing (alt.
non-increasing) relationship - typically a monotonic piecewise-constant
function - meaning there can be large jumps between subsequent feature values.
It can, therefore, be a helpful alternative when a monotonic relationship
between variables exists, especially when this relationship does not follow an
obvious Linear or Polynomial form typical in regression.

In the classical formulation (with $L_2$ loss) where $\mathfrak{I}$ is a
partial order defined over our independent variables, *Isotonic Regression* can
be defined as the following optimisation problem.

$$\begin{aligned}
&\text{minimise}   &\quad \sum_i (\hat{y}_i - y_i)^2 & \\
&\text{subject to} &\quad \hat{y}_i \preceq \hat{y}_j, &\quad \forall \left ( i, j \right ) \in \mathfrak{I}
\end{aligned}
$$

Throughout this post, we will discuss what a Linear Program is and how it leads
to the algorithm discussed in [@LussRossetGeneralizedIsotonicRegression] for
solving the Isotonic Regression problem in an arbitrary number of dimensions
while supporting any convex differentiable loss function.

## Linear Programming

Linear Programming is a generic approach to solving constrained problems with a
linear objective function and linear constraints. The standard form of such a
problem asks us to find a vector $\bm{x} \in \mathbb{R}^p$ minimising our
objective $o(\bm{x})$ and takes the form:

$$\begin{aligned}
&\text{minimise}         &o(\bm{x}) \\
&\text{subject to} \quad &b_i(\bm{x}) &\leq 0, &\quad \forall i \\
&                        &c_j(\bm{x}) &=    0, &\quad \forall j
\end{aligned}
$$

where our inequalities $b_i$ and equalities $c_j$ express linear relationships.

The advantage of prescribing such a form is that generic software can be
written to solve any problem that fits the template. [^AvailableSoftware] With
$o$ being any linear function, we are not even restricted to minimisation
problems and inequalities of the form $\leq 0$. Multiplying our objective by negative
one lets us morph a maximisation problem into the above form. In the same
way, a greater than inequality can be changed to a less than. Furthermore, an
offset can be included to support equalities and inequalities with values other
than zero. For example, the linear relation $x \geq 5$ changed to $5 - x \leq
0$ produces the function $b_i(x) = 5 - x$, which also fits the above form. We
can then use generic interior point or simplex solvers to minimise the objective.

[^AvailableSoftware]: For example, the commercial software,
    [MOSEK](https://www.mosek.com/), or the the open source
    [HiGHS](https://highs.dev/#top), which I have used in this implementation.

### Example Linear Program

Consider, for example, a hobbyist photographer and amateur baker. He wants to
maximise the time he spends on his two favourite activities but, unfortunately,
cannot finance them without limits. From these wishes, we can create a
constrained optimisation problem.

[^OnlineSolver]

[^OnlineSolver]: You may choose to play around with setting up and solving some
    of these problems using an
    [online solver](https://online-optimizer.appspot.com/?model=builtin:default.mod).

Our hobbyist wants to *maximise* his time doing his favourite activities;
perhaps he prefers photography, $p$ over baking, $b$ ($2p +b$). Unfortunately,
photography is considerably more expensive at 5€ per hour than baking at 1€,
and he has a limited budget of 10€ constraining the activities ($5p + b \leq 10$).
Finally, he does not want to make more baked goods than he can eat, so he
limits himself to 5 hours of baking ($b \leq 5$). Along with the natural
assumption that a negative amount of time can not be spent on an activity, this
produces the Linear Program:

$$\begin{aligned}
&\text{maximize}         & 2 p + b \\
&\text{subject to} \quad & 5 p + b &\leq 10 \\
&                        & b       &\leq 5 \\
&                        & p       &\geq 0 \\
&                        & b       &\geq 0
\end{aligned}
$$

With such few constraints and dimensions, this Linear Program is easily solved.
We plot the constraints and choose the largest value possible; one of the
vertices. [^Technically] However, once we have more than three dimensions, it
is more challenging to visualise such a problem. Regardless, a similar approach
can be taken, checking each vertex of the feasible region of the corresponding
multidimensional polytope.

[^Technically]: Technically, it will be one of the vertices if there is only a
    single solution and otherwise, any point along a line between vertices if
    there are multiple solutions.

```{=html}
<div id="LinearProgram"></div>
```

Plotting all of the constraints, we see our maximal vertex in the top right
corner, which suggests that our hobbyist spend an hour a week taking pictures
and five hours baking bread.

[^TutorialVideo]

[^TutorialVideo]: For a more in-depth introduction, see this video
    [The Art of Linear Programming](https://www.youtube.com/watch?v=E72DWgKP_1Y)

We can also write this problem in the standard form we mentioned above.

$$\begin{aligned}
&\text{minimise}         & - 2p - b \\
&\text{subject to} \quad & 5 p + b - s &\leq 0 \\
&                        & b - 5       &\leq 0 \\
&                        & -p          &\leq 0 \\
&                        & -b          &\leq 0
\end{aligned}
$$


### Lagrangian Dual Function

Given the standard form:

$$\begin{aligned}
&\text{minimise}         &o(\bm{x}) \\
&\text{subject to} \quad &b_i(\bm{x}) &\leq 0, &\quad \forall i \\
&                        &c_j(\bm{x}) &=    0, &\quad \forall j
\end{aligned}
$$

we also have the option to define a Lagrangian dual function. Essentially, this
new definition relaxes the constraints by multiplying them with a
$\lambda_i \in \mathbb{R}_0^+$ or $v_i \in \mathbb{R}_0^+$ term, for
inequalities and equalities respectively, removing discontinuities in the
search space. When violating constraints, we no longer move from a finite to an
undefined objective value. Furthermore, by directly including the weighted sum
of these constraints in the objective, the Lagrangian acts as a lower bound on
the original problem.

$$
L(\bm{x}, \bm{\lambda}, \bm{v}) = o(\bm{x}) + \sum_i \lambda_i b_i(\bm{x}) + \sum_j v_j c_j(\bm{x})
$$

It is natural to ask what the optimal lower bound is for a given $\bm{\lambda}$
and $\bm{v}$. This is the infimum of the above objective.

$$
g(\bm{\lambda}, \bm{v}) = \text{inf}_{\bm{x}} L(\bm{x}, \bm{\lambda}, \bm{v})
$$

Taking this objective, we then define the dual problem of our standard form
above; the following convex maximisation problem:

$$\begin{aligned}
&\text{maximize}         & g(\bm{\lambda}, \bm{v}) \\
&\text{subject to} \quad & \bm{\lambda} \succeq 0
\end{aligned}
$$

[@BoydVandenbergheConvexOptimization{}, pages 215-223]

### Example Lagrangian Dual Problem

Using the form of this new dual Lagrangian problem, we can lower-bound our
example from above. Our Lagrangian is:

$$\begin{aligned}
L(p, b, \bm{\lambda}) &= -2p - b + \lambda_1 \left ( 5p + b - 10 \right ) + \lambda_2 \left ( b - 5 \right ) - \lambda_3 p - \lambda_4 b \\
                      &= p \left ( 5 \lambda_1 - \lambda_3 - 2 \right ) + b \left ( \lambda_1 + \lambda_2 - \lambda_4 - 1 \right ) + \left ( - 10 \lambda_1 - 5 \lambda_2 \right ) \\
\end{aligned}
$$

We have no $\bm{v}$ term in this case due to a lack of equality constraints,
and we have $(p, b)$ in place of $\bm{x}$. Our dual problem is then formulated
as follows:

$$\begin{aligned}
&\text{maximize}         &\text{inf}_{p, b} \text{ } & p \left ( 5 \lambda_1 - \lambda_3 - 2 \right ) + b \left ( \lambda_1 + \lambda_2 - \lambda_4 - 1 \right ) + \left ( - 10 \lambda_1 - 5 \lambda_2 \right ) \\
&\text{subject to} \quad &\bm{\lambda} & \succeq 0
\end{aligned}
$$

What we notice is that this program is unbounded for all values of
$\bm{\lambda}$ except for those where $(5 \lambda_1 - \lambda_3 - 2) = 0$ and
$(\lambda_1 + \lambda_2 - \lambda_4 - 1) = 0$. If either of these terms is not
zero, the corresponding $p$ or $b$ term can be arbitrarily small, and the
infimum returns negative infinity. [^PropertyLinearProg] Using this, we can derive an analytic
solution by first rearranging our two equalities

[^PropertyLinearProg]: Linear programs are always unbounded like this, but this
    is not generally true of optimisation problems.

$$\begin{aligned}
\lambda_1 &= 1/5 \lambda_3 + 2/5 \\
\lambda_2 &= \lambda_4 + 1 - \lambda_1 \\
          &= \lambda_4 + 3/5 - 1/5 \lambda_3 \\
\end{aligned}
$$

and then substituting these into our objective function $L(p, b, \bm{\lambda})$ and simplifying

$$\begin{aligned}
L(p, b, \bm{\lambda}) &= -\left ( 10 \lambda_1 + 5 \lambda_2 \right ) \\
                      &= -\left ( 2 \lambda_3 + 4 + 5 \lambda_4 + 3 - \lambda_3 \right ) \\
                      &= -\left ( \lambda_3 + 5 \lambda_4 + 7 \right )
\end{aligned}
$$

and realising that if any $\lambda_i > 0$ we move away from our maximum, we reach a maximum of

$$
\lambda_3 = 0, \lambda_4 = 0: \quad -7
$$

Finally, recalling that we negated the objective in the formulation of our original
problem to fit the standard form, we get $7$. In other words, our lower bound is
**identical** to the optimal value from our original problem! [^LPExamples] Activating the curve
$2p +b = 7$ in our feasible region graph above also shows that this new
constraint resulting from the Lagrangian dual program intersects the feasible
at precisely one point, our optimal vertex from before.

[^LPExamples]: For many, many more examples, see this excellent and free textbook
    [Convex Optimisation](https://web.stanford.edu/~boyd/cvxbook/bv_cvxbook.pdf)

### Karush-Kuhn-Tucker (KKT) Conditions

If we assume, for a moment, that we know the optimal value $\bm{x}^*$ and
optimal dual values $\bm{\lambda}^*$ and $\bm{v}^*$ of our Lagrangian and that
our lower bound is equal to the standard form solution, as in the preceding
example, we can conclude that the Lagrangian,
$L(\bm{x}, \bm{\lambda}^*, \bm{v}^*)$ has attained its minimum at $\bm{x}^*$
and consequently has a gradient of zero.

$$
\Delta o(\bm{x}^*) + \sum_i \lambda_i^* \Delta b_i (\bm{x}^*) + \sum_j v_j^* \Delta c_j(\bm{x}^*) = 0
$$

We can conclude several additional conditions that must hold at this optimal
set of values. Together, these are the Karush-Kuhn-Tucker (KKT) Conditions:

$$\begin{aligned}
b_i(\bm{x}^*)             &\leq 0, &\quad \forall i \\
c_j(\bm{x}^*)             &= 0   , &\quad \forall j \\
\lambda_i^*               &\geq 0, &\quad \forall i \\
\lambda_i^* b_i(\bm{x}^*) &= 0,    &\quad \forall i
\end{aligned}
$$

The first two must hold as an optimal $\bm{x}^*$ must satisfy the constraints
of our linear program's standard form. The third is true as the optimal
$\bm{\lambda}^*$ and $\bm{v}^*$ must satisfy the constraint of the Lagrangian
dual program. To reach the final condition, we build on the others. From the
first, we know that $b_i(\bm{x}^*) \leq 0$. This means that any $\lambda_i > 0$
will push our Lagrangian dual problem away from its maximum as one of its
summations multiplies the positive $\lambda_i$ with a negative $b_i(\bm{x})$.
Consequently, attaining the optimal solution requires that either $\lambda_i$
or $b_i(\bm{x})$ equal zero producing the last condition.

[^Feasible]

[^Feasible]: This all assumes there is a feasible solution and strong duality,
    i.e. that the optimal solution to both the dual and primal problem is
    identical.

It is not always the case that the primal and dual programs produce the same
solution. If, however, our problem is convex, as in our example above, then we
always have a zero duality gap when together values $\hat{\bm{x}}$,
$\hat{\bm{\lambda}}$ and $\hat{\bm{v}}$ satisfy the KKT conditions.
[^PapersConvexity] Without this convexity guarantee, the gradient of the
Lagrangian equating to zero does not imply it has been minimised. When
convexity does hold, though, we can use the KKT conditions to show that the
following equality holds:

[^PapersConvexity]: Luckily, the constraints in the linear program from the
    paper we are looking at here are convex and the objective is restricted to
    convex differentiable functions.

$$\begin{aligned}
g(\hat{\bm{\lambda}}, \hat{\bm{v}}) &= L(\hat{\bm{x}}, \hat{\bm{\lambda}}, \hat{\bm{v}}) \\
                                    &= o(\hat{\bm{x}}) + \sum_i \hat{\lambda}_i b_i(\hat{\bm{x}}) + \sum_j \hat{v}_i c_i(\hat{\bm{x}}) \\
                                    &= o(\hat{\bm{x}})
\end{aligned}
$$

[@BoydVandenbergheConvexOptimization{}, pages 244-245]

## Generalised Isotonic Regression

Having covered the necessary background material, we can move to the paper. We
aim to solve the following general version of the isotonic regression problem:

$$\begin{aligned}
&\text{minimise}           &\sum_i f_i(\hat{y}_i) & \\
&\text{subject to}_i \quad &\hat{y}_i  - \hat{y}_j \leq 0, &\quad \forall \left ( i, j \right ) \in \mathfrak{I}
\end{aligned}
$$

where each $f_i : \mathbb{R} \rightarrow \mathbb{R}$ is a differentiable convex
function. Should we wish to use an $L_2$ loss function, we would substitute
$(\hat{y}_i - y_i)^2$ in place of $f_i(\hat{y})$.

As a starting point, we consider the corresponding dual Lagrangian objective
function and the KKT conditions. For a single point, $i$, the Lagrangian is:

$$
L_i(\hat{\bm{y}}, \bm{\lambda}) = f_i(\hat{y}_i)  + \sum_j \lambda_{ij} \left ( \hat{y}_i  - \hat{y}_j \right ) + \sum_k \lambda_{ki} \left ( \hat{y}_k  - \hat{y}_i \right )
$$

In the first summation, we include the constraints for all points larger than
$i$ and in the second for all points smaller. The KKT for a single point $i$
are:

$$\begin{aligned}
\frac{\partial f_i(\hat{y}_i^*)}{\partial \hat{y}_i^*} + \sum_j \lambda_{ij}^* - \sum_k \lambda_{ki}^* &= 0 \\
\hat{y}_i^* - \hat{y}_j^*                                 &\leq 0, &\quad \forall (i, j) \in \mathfrak{I} \\
\text{no equality constraints} \\
\lambda_{ij}^*                                            &\geq 0, &\quad \forall (i, j) \in \mathfrak{I} \\
\lambda_{ij}^* \left ( \hat{y}_i^* - \hat{y}_j^* \right ) &= 0,    &\quad \forall (i, j) \in \mathfrak{I}
\end{aligned}
$$

These conditions reveal several properties of the optimal solution:

1. From conditions two and four, we conclude that optimal estimates
   $\hat{y}_i^*$ and $\hat{y}_j^*$ for isotonically constrained points $i
   \preceq j$ are either identical, as would be the case when the unconstrained
   estimate otherwise produces $\hat{y}_i > \hat{y}_j$, or the two estimates
   trivially satisfy the constraint and $\lambda_{ij} = 0$.
2. Consequently, the optimal solution partitions the space into blocks $V$,
   where every point within a block has the identical estimate $\hat{y}_V^*$
3. The second condition, furthermore, implies isotonicity at the optimum.

Therefore, the paper's authors suggest an algorithm which iteratively selects
the block $V$ with the largest between-group variance and seeks to partition
this group or conclude that it is already optimal and move on to the next. An
example can be seen in the plot below. As with many statistical and machine
learning algorithms, with too many iterations, it appears to overfit the
solution.

```{=html}
<div id="UniIsoPlot"></div>
```

### Optimal Cut

From the first KKT condition above, we know that a block $V$ that is already
optimally partitioned will satisfy:

$$
\sum_{i \in V} \frac{\partial f_i(\hat{y}_V)}{\partial \hat{y}_V} = 0
$$

[^TelescopingEquivalence]

[^TelescopingEquivalence]: Our first KKT condition is a telescopic series that
    sums to zero. Consider three points $a$, $b$ and $c$; all terms cancel out
    when summed.
    $$
    \begin{aligned}
    a&: 0 - \lambda_{ab} - \lambda_{ac} \\
    b&: \lambda_{ab} - \lambda_{bc} \\
    c&: \lambda_{ac} + \lambda_{bc} \\
    \end{aligned}
    $$

If, however, this is not the case, we can find a more optimal solution by
partitioning our block into two subblocks $V_-$ and $V_+$, where all points
$i\in V_-$ are smaller than those $j \in V_+$ according to our isotonicity
constraints.

$$
\sum_{j \in V_+} \frac{\partial f_j(\hat{y}_V)}{\partial \hat{y}_V} - \sum_{i \in V_-} \frac{\partial f_i(\hat{y}_V)}{\partial \hat{y}_V} < 0
$$

However, this only happens if the first summand has a net negative gradient and
the second a net positive. In other words, a better partitioning implies that
our estimates $\hat{y}_j$ want to increase in magnitude, while $\hat{y}_i$
wants to decrease, maintaining isotonicity. After partitioning a block, this
difference also measures how well separated the two new blocks are and is used
to guide the iterative splitting. At each iteration, therefore, we select the
block with the maximal difference and search for a new optimal cut or to verify
that the block is already optimal.

[^ConsiderL2]: Consider, for example, the $L_2$ loss. In this case
    $\frac{\partial f_i(\hat{y}_V)}{\partial \hat{y}_V} = 2 (\hat{y}_V - y_i)$.
    Where $y_i$ is on average larger than $\hat{y}_V$, as should be the case
    with the left summand for a non-optimal block, we have a net negative
    total.

$$
\text{minimise}_{\left ( V^-, V^+ \right ) \in V} \left ( \sum_{j \in V^+} \frac{\partial f_j(\hat{y}_V)}{\partial \hat{y}_V} - \sum_{i \in V^-} \frac{\partial f_i(\hat{y}_V)}{\partial \hat{y}_V} \right )
$$

[^EquivalentMaxVariance]

[^EquivalentMaxVariance]: This is equivalent to maximising the between-group
    variance at each step. For details see
    [@LussRossetEfficientRegularizedIsotonicRegression]

Introducing a new variable $x_i$, which can be either $1$ or $-1$, this
minimisation problem is equivalent to the following binary program:

$$\begin{aligned}
&\text{minimise}   &\quad \sum_{i \in V} x_i \frac{\partial f_i(\hat{y}_V)}{\partial \hat{y}_V} \\
&\text{subject to} &\quad x_i \leq x_j                      &\quad \forall (i,j) \in \mathfrak{I}_V \\
&                  &\quad x_i \in \left \{ -1, +1 \right \} &\quad \forall i \in V
\end{aligned}
$$

Finally, relaxing the integer constraint, allowing $x_i$ to vary continuously
between $-1$ and $1$, and converting it to our standard form, we reach the
linear program from the paper:

$$\begin{aligned}
&\text{minimise}    &\quad \sum_{i \in V} x_i \frac{\partial f_i(\hat{y}_V)}{\partial \hat{y}_V} \\
&\text{subject to}  &\quad  x_i - x_j \leq 0 &\quad \forall (i,j) \in \mathfrak{I}_V \\
&                   &\quad -x_i - 1   \leq 0 &\quad \forall i \in V \\
&                   &\quad  x_i - 1   \leq 0 &\quad \forall i \in V
\end{aligned}
$$

[^MinimisationVsMaximisation]

[^MinimisationVsMaximisation]: Previous work from the author of the discussed
    paper typically considers a maximisation problem instead of the
    minimisation problem we discuss here. See for example,
    [@LussRossetDecomposingIsotonicRegression{}, equation 3] and
    [@LussRossetEfficientRegularizedIsotonicRegression{}, equation 3]. These
    two variants are equivalent, but the order of comparison is reversed. Here
    we take the derivative of the $L_2$ loss to be $2 (\hat{y}_V - y_i)$,
    whereas the previous works take this to be $2 (y_i - \hat{y}_V)$. Switching
    from a minimisation to a maximisation requires few changes. In the case of
    the final matrix form below, we would swap the maximisation for
    minimisation and swap the domain of our variables $y_i$ from $y_i \leq 0$
    to $y_i \geq 0$. In my full implementation, I have actually implemented
    this opposite variant following the implementation from the author.

---

## Implementation

Now that we have come to implementing the algorithm, we consider an approach
to determining and storing our isotonicity constraints $\mathfrak{I}$, before
we then implement a `minimum_cut` method to solve the above Linear Program.
Finally, we discuss briefly how they can fit together as an implemention
to the algorithm in the paper.

### Isotonic Constraints

In a single dimension, points can be ordered unambigiously according to a
traditional $\leq$ operator, whereby $0 \leq 2 \leq 5$, etc. Once we have
at least $2$-dimensions, however, things aren't so simple. Which of the
two points $(1, 3)$ and $(3, 1)$ are larger? One option would be to order
the pairs lexicographically, in which case the second is larger. Here, however,
we consider a domination based ordering, whereby a point $(x_1, x_2)$ is
considered smaller than another point $(y_1, y_2)$ if both $x_1 \leq y_1$ and
$x_2 \leq y_2$. Following this relation, the two points $(1, 3)$ and $(3, 1)$
have no relation to on another. The are both, however, larger than the point
$(1, 1)$ and smaller than the point $(3, 3)$. Following this relation, we
construct the ordering below, for a number of points in $2$-dimensional space.

```{=html}
<div id="AdjacencyMatrix"></div>
```

To save space, we save this ordering as a sparse adjacency matrix. Clicking
on the "Adjacency Matrix" button, we see as is typical of an adjacency matrix
that where there is a $1$ at entry $i, j$ the point $i$ is less than or equal to
point $j$. We haven't, however, added a $1$ for every point $i$ that is less
that or equal to point $j$. We could have included all of these and the
implementation would still have worked. Removing them is an optimisation
which removes redundant information and consequently the number of constraints
passed into the Linear Program solver improving performance.

[^BruteForceAlternatives]

[^BruteForceAlternatives]: A potential alternative to the brute force
approach I have coded here, might be to make use of the approach of
[@BentleyMultidimensionalDivideAndConquer] determining a ranking based
on the number of points dominated by each point. This can be done in
$T(N, k) = O \left ( N \text{log}^{k-1} N \right )$ time, with $k$
dimensions and $N$ points. We would then need to only compared points
with those of the previous rank.

Roughly speaking, we build this simplified adjacency matrix,
by sorting lexicographically, then comparing each point, with each of
its predecessors according to the lexicographic ordering. We add any
points that smaller than the current point to the matrix, unless,
they have a successor that is also a predececssor of the currently
considered point. Ignoring the possibility of duplicate points
this leads to the following implementation, which does $N^2 / 2$
comparisons.

```cpp
template<typename V>
Eigen::SparseMatrix<bool>
points_to_adjacency(const Eigen::MatrixXd& points) {

  const uint64_t total_points = points.rows();

  Eigen::SparseMatrix<bool> adjacency(total_points, total_points);
  VectorXu degree = VectorXu::Zero(total_points);

  Eigen::VectorX<bool> is_predecessor =
      Eigen::VectorX<bool>::Zero(total_points);
  Eigen::VectorX<bool> is_equal =
      Eigen::VectorX<bool>::Zero(total_points);

  // Lexicographic Sorting
  const auto& sorted_idxs = argsort(points);
  const Eigen::MatrixXd sorted_points =
      points(sorted_idxs, Eigen::all).transpose();

  for (uint64_t i = 1; i < total_points; ++i) {

    const auto& previous_points = sorted_points(
        Eigen::all,
        VectorXu::LinSpaced(i, 0, i-1)
    ).array();

    const auto& current_point = sorted_points(
        Eigen::all,
        VectorXu::LinSpaced(i, i, i)
    ).array();

    is_predecessor(Eigen::seq(0, i-1)) =
        (
            previous_points <= current_point
        ).colwise().all();

    degree(i) = is_predecessor.count();

    /* If there is a chain of points that are all predecessors,
     * we take only the largest. So, we check if the outgoing
     * edge of a predecessor connects to another predecessor
     * (which would be included instead).
     */
    for (Eigen::Index j = 0; j < adjacency.outerSize(); ++j) {
      if (is_predecessor(j)) {
        for (
          Eigen::SparseMatrix<bool>::InnerIterator it(adjacency, j);
          it;
          ++it
        ) {
          if (it.value() && !is_equal(it.row())) {
            is_predecessor(it.row()) = false;
          }
        }
      }
    }

    adjacency.col(i) = is_predecessor.sparseView();
  }

  return adjacency;
}
```

### Minimum Cut / Maximum Flow

An implementation therefore amounts to reapplying a minimin cut /maximum flow function
to the subset of points with the previously largest serparation
until blocks can no longer be cut.

HiGHS solves the LP in matrix form. which is just
another way of representing the same set of problems.

$$\begin{aligned}
\text{minimise}    &\quad \bm{b}^T \bm{x} \\
\text {subject to} &\quad \bm{A} \bm{x} \leq \bm{c} \\
\end{aligned}
$$

For our example adjacency matrix, our objective is

$$
\bm{b} = \begin{bmatrix}
\frac{\partial f_1(\hat{y})}{\partial \hat{y}} \\
\frac{\partial f_2(\hat{y})}{\partial \hat{y}} \\
\frac{\partial f_3(\hat{y})}{\partial \hat{y}} \\
\frac{\partial f_4(\hat{y})}{\partial \hat{y}} \\
\frac{\partial f_5(\hat{y})}{\partial \hat{y}} \\
\frac{\partial f_6(\hat{y})}{\partial \hat{y}}
\end{bmatrix}
$$

and we build our constraint matrix

$$\begin{aligned}
&\bm{A} &\bm{x} &\leq &\bm{c} \\
&\begin{bmatrix}
 1 &  0 &  0 &  0 &  0 &  0 \\
 0 &  1 &  0 &  0 &  0 &  0 \\
 0 &  0 &  1 &  0 &  0 &  0 \\
 0 &  0 &  0 &  1 &  0 &  0 \\
 0 &  0 &  0 &  0 &  1 &  0 \\
 0 &  0 &  0 &  0 &  0 &  1 \\
-1 &  0 &  0 &  0 &  0 &  0 \\
 0 & -1 &  0 &  0 &  0 &  0 \\
 0 &  0 & -1 &  0 &  0 &  0 \\
 0 &  0 &  0 & -1 &  0 &  0 \\
 0 &  0 &  0 &  0 & -1 &  0 \\
 0 &  0 &  0 &  0 &  0 & -1 \\
 1 & -1 &  0 &  0 &  0 &  0 \\
-1 &  1 &  0 &  0 &  0 &  0 \\
 0 &  1 & -1 &  0 &  0 &  0 \\
 0 &  1 &  0 & -1 &  0 &  0 \\
 0 &  0 &  1 &  0 &  0 & -1 \\
 0 &  0 &  0 &  1 & -1 &  0 \\
 0 &  0 &  0 &  0 &  1 & -1
\end{bmatrix}
&\begin{bmatrix}
x_1 \\
x_2 \\
x_3 \\
x_4 \\
x_5 \\
x_6
\end{bmatrix}
&\leq
&\begin{bmatrix}
1 \\
1 \\
1 \\
1 \\
1 \\
1 \\
1 \\
1 \\
1 \\
1 \\
1 \\
1 \\
0 \\
0 \\
0 \\
0 \\
0 \\
0 \\
0
\end{bmatrix}
\end{aligned}
$$

The first 6 rows then represent $x_i \leq 1$.

The next 6 rows correspond to the constraint $-1 \leq x_i$, where we first
multiplied each side by $-1$ to get $-x_i \leq 1$.

We then have a row for each entry in our adjacency matrix, corresponding
to a constraint $x_i - x_j \leq 0$.

[^MatrixFormAnotherExample]

[^MatrixFormAnotherExample]: Another example of converting a Linear Program
to matrix form can be seen on the first few slides
[here](https://vanderbei.princeton.edu/542/lectures/lec6.pdf).

We could solve this problem, but following the suggestions of the paper's author,
in the name of performance, we instead implement the dual Linear Program switching
from the optimal or minimum cut variant above to a maximum flow variant. As
we have already converted our problem to the matrix form, this is a relatively
mechanical process, where we just need to apply the rules under *Constructing the dual LP*
[here](https://en.wikipedia.org/wiki/Dual_linear_program#Constructing_the_dual_LP).
Our new LP is

$$\begin{aligned}
\text{minimise}    &\quad \bm{b}^T \bm{x}                             &\quad \text{maximise}    &\quad \bm{c}^T \bm{y} \\
\text {subject to} &\quad \bm{A} \bm{x} \leq \bm{c} \quad \rightarrow &\quad \text {subject to} &\quad \bm{A}^T \bm{y} = \bm{b} \\
                   &\quad \bm{x} \in \mathbb{R}^N                     &\quad                    &\quad \bm{y} \leq 0
\end{aligned}
$$

[^DualLinearProgram]

[^DualLinearProgram]: Realising that this Min-Cut problem can be reformulated
as a Max-Flow problem, permits the use of many other algorithms. Just recently,
this was even discussed on [Quanta](https://www.quantamagazine.org/researchers-achieve-absurdly-fast-algorithm-for-network-flow-20220608/)
or the more technical presentation [here](https://www.youtube.com/watch?v=KsMtVthpkzI).
There are also other algorithms such as [Ford Fulkerson](https://en.wikipedia.org/wiki/Ford%E2%80%93Fulkerson_algorithm)
which operate directly on a maximum flow graph. For more information above the correspondence between
the [Max-Flow and Min-Cut problems](https://en.wikipedia.org/wiki/Max-flow_min-cut_theorem#Linear_program_formulation).

The matrix $A^T$ we build directly as follows
we use `considered_idxs` to select a subset of points instead of
building subsets of the sparse matrix `adjacency_matrix` to pass around

```cpp
Eigen::SparseMatrix<int>
adjacency_to_LP_standard_form(
    const Eigen::SparseMatrix<bool>& adjacency_matrix,
    const VectorXu& considered_idxs
) {
  const uint64_t total_constraints =
      constraints_count(adjacency_matrix, considered_idxs);

  const uint64_t total_observations = considered_idxs.rows();
  const uint64_t columns = 2 * total_observations + total_constraints;

  Eigen::SparseMatrix<int, Eigen::ColMajor> standard_form(
      total_observations, columns);

  int idx = 0;
  for (Eigen::Index j = 0; j < total_observations; ++j) {
    auto it = Eigen::SparseMatrix<bool>::InnerIterator it(
        adjacency_matrix, considered_idxs(j));

    for (
      it;
      it;
      ++it
    ) {
      auto row_idx = std::find(
          considered_idxs.begin(),
          considered_idxs.end(),
          it.row());

      // add isotonicity constraint if both points in
      // considered subset
      if (row_idx != considered_idxs.end()) {

        // smaller point
        standard_form.insert(
            std::distance(considered_idxs.begin(), row_idx),
            2 * total_observations + idx) = -1;

        // larger point
        standard_form.insert(
            j,
            2 * total_observations + idx) = 1;

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

now that we can build our constraint matrix
we can set up and solve our linear program
with HiGHS
we start by setting the optimistation direction

```cpp
Eigen::VectorX<bool>
minimum_cut(
    const Eigen::SparseMatrix<bool>& adjacency_matrix,
    const Eigen::VectorXd loss_gradient, // z in the paper
    const VectorXu considered_idxs
) {

  HighsModel model;
  model.lp_.sense_ = ObjSense::kMaximize;
```

create our matrix $A^T$

```cpp
  const auto A =
      adjacency_to_LP_standard_form(
          adjacency_matrix, considered_idxs);

  model.lp_.num_col_ = A.cols();
  model.lp_.num_row_ = A.rows();
```

HiGHS requires that we flatten the contents of the matrix
into a single vector (in this case column wise),
with an additional vector specifying
where each new column starts.

```cpp
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
```

add the last constraint that $\bm{y} \leq 0$

```cpp
  model.lp_.col_lower_ =
      std::vector<double>(A.cols(), -infinity);

  model.lp_.col_upper_ =
      std::vector<double>(A.cols(), 0);
```

calculating our objective, we sum ...

```cpp
  std::vector<double> b(A.cols());
  for (size_t i = 0; i < total_observations * 2; ++i)
    b[i] = 1;
  for (size_t i = 0; i < total_constraints; ++i)
    b[2 * total_observations + i] = 0;

  model.lp_.col_cost_ = std::move(b);
```

and we set gradient

```cpp
  std::vector<double> c(loss_gradient.begin(), loss_gradient.end());
  model.lp_.row_lower_ = c;
  model.lp_.row_upper_ = c;
```

Finally, we run the solver and retrieve the values of $x$
that produce the minimum. Here we take row of the solution
to the dual linear program. (as from original)
The dual provides a single
value for each point, our partitioning.

```cpp
  Highs highs;
  highs.setOptionValue("solver", "simplex");

  highs.passModel(model);
  highs.run();

  auto solution = Eigen::VectorXd::Map(
      &highs.getSolution().row_dual[0],
      highs.getSolution().row_dual.size()).array() > 0;

  return solution.array() > 0; // 0 left = 1 right
}
```

### Algorithm Overview

---


### References

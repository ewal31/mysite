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

### Karush-Kuhn-Tucker (KKT) Conditions

Can make some conclusions about the point $(\bm{x}^*, \bm{\lambda}^*, \bm{v}^*)$
assuming no duality gap
To see why this is the case (although it might seem a little arbitrary at first
these conditions are important in optimsation and the paper we are discussing)
lets assume we have an optimal $\bm{x}^*$ satisfying the LP
and optimal $(\bm{\lambda}^*, \bm{v}^*)$
the lagragian must be minimised at this point and therefore the gradient vanishes

$$
\Delta f_0(\bm{x}^*) + \sum_i \lambda_i^* \Delta f_i (\bm{x}^*) + \sum_j v_j \Delta h_j(\bm{x}^*) = 0
$$

We can make a number of conclusions about this optimal point (the KKT condidtions)

$$\begin{aligned}
f_i(\bm{x}^*)             &\leq 0, &\quad \forall i \\
h_j(\bm{x}^*)             &= 0   , &\quad \forall j \\
\lambda_i^*               &\geq 0, &\quad \forall i \\
\lambda_i^* f_i(\bm{x}^*) &= 0,    &\quad \forall i
\end{aligned}
$$

The first two must be true, as an optimal $\bm{x}^*$ must satisfy the conditions of
the LP, i.e. the inequalities on $f_i$ and the equalities of $h_j$. The third
must be true, as the optimal pair $(\bm{\lambda}^*, \bm{v}^*)$ must satisfy
the conditions of our dual problem, that $\bm{lambda}$ is non-negative. and we arrive at
the final condition through the following (i think)
at our optimal $\bm{x}^*$ it is the case that $f_i(\bm{x}^*) \leq 0$ via our LP
then if $(\bm{\lambda}^*, \bm{v}^*)$ maximises our dual, $\sum_i \lambda_i f_i(\bm{x}) \rightarrow 0$
as a positive $\bm{\lambda}$ will make this summation more negative reducing the maximum
therefore it should be 0 at our optimal points (there is also a proof on page 242)

[^Feasible]

[^Feasible]: This all assumes there is a feasible solution and strong duality, i.e. that
the optimal solution to both the dual and primal problem is identical.

Now, the paper we are wanting to discuss in this post, considers only
convex differentiable loss functions. Luckily, in the case where
convex $f$ and affine $h$, both the original formulation and
the dual problem using the lagrangian produce the same optimal solution (page 244)
or in other words, the point where KKT hold, is the optimal solution
both to the LP and the dual lagrangian
This conclusions follows from the KKT conditions. given an optimal $\bm{x}^*$
and optimal $(\bm{\lambda}^*, \bm{v}^*)$ points that satisfy the conditions
of the primal and the dual, and consequently the conditions above

first we note that at $\bm{x} = \bm{x}^*$ the gradient of $L(\bm{x}, \bm{\lambda}^*, \bm{v}^*)$ is zero
and consequently minimises $L(\bm{x}, \bm{\lambda}^*, \bm{v}^*)$ over $x$

therefore our dual problem is equivalent to L at this point

$$\begin{aligned}
g(\bm{\lambda}^*, \bm{v}^*) &= L(\bm{x}^*, \bm{\lambda}^*, \bm{v}^*) \\
                            &= f_0(\bm{x}^*) + \sum_i \lambda^*_i f_i(\bm{x}^*) + \sum_j v^*_i h_i(\bm{x}^*) \\
                            &= f_0(x^*)
\end{aligned}
$$

here we use conditions 2 and 4 from above leading to equality in the last line.

[@BoydVandenbergheConvexOptimization{}, pages 244-245]

### Example Lagrangian Dual Problem

Lagrangian for the example above is as follows

$$\begin{aligned}
L(p, b, \bm{\lambda}) &= -2p - b + \lambda_1 \left ( 5p + b - s \right ) + \lambda_2 \left ( b - 5 \right ) - \lambda_3 p - \lambda_4 b \\
                      &= p \left ( 5 \lambda_1 - \lambda_3 - 2 \right ) + b \left ( \lambda_1 + \lambda_2 - \lambda_4 - 1 \right ) + \left ( - s \lambda_1 - 5 \lambda_2 \right ) \\
\end{aligned}
$$

considering the dual problem. this is unbounded and equal to negative infinity for every value of $\bm{\lambda}$
except for those, where $(5 \lambda_1 - \lambda_3 - 2) = 0$ and $(\lambda_1 + \lambda_2 - \lambda_4 - 1) = 0$
and we want to maximise this result where these two equal 0

$$\begin{aligned}
\lambda_1 &= 1/5 \lambda_3 + 2/5 \\
\lambda_2 &= \lambda_4 + 1 - \lambda_1 \\
          &= \lambda_4 + 3/5 - 1/5 \lambda_3 \\
\text{maximise}& -\left ( s \lambda_1 + 5 \lambda_2 \right ) \\
\text{maximise}& -\left ( s/5 \lambda_3 + 2s/5 + 5 \lambda_4 + 3 - \lambda_3 \right ) \\
\text{maximise}& -\left ( (s-1)/5 \lambda_3 + 2s/5 + 5 \lambda_4 + 3 \right ) \\
\text{if the } \lambda_i \text{ terms are } > 0 \text{ then we get a small maximum.} \\
\lambda_3 = 0, \lambda_4 = 0: &\quad -\left ( 2s/5 + 3 \right )
\end{aligned}
$$

So, our solution to the dual problem is $-( 2 * 10 / 5 + 3) = -7$ and is accomplished
where $2p + b = 7$ intersects the feasible region. See the picture above.

[^LPExamples]

[^LPExamples]: For many many more examples see this wonderful and free textbook [Convex Optimisation](https://web.stanford.edu/~boyd/cvxbook/bv_cvxbook.pdf)

## Paper

with the background introduced we can look at the paper

where $f$ is our convex differentiable function we are using to guide optimisation
where $f(\bm{x}) = \sum_i f_i(x_i)$ Need to fix notation here
i.e. for $L_2$ loss we have $f_i^{L_2}(x) = (x - x_i)^2$
we have no equalities and so no $h$ part and end up with the following
in our LP standard form

$$\begin{aligned}
&\text{minimise}     &\quad \sum_i f_i(x_i) & \\
&\text{subject to}_i &\quad \hat{y}_k  - \hat{y}_i \leq 0, &\quad \forall \left ( k, i \right ) \in \mathfrak{I} \\
&                    &\quad \hat{y}_i  - \hat{y}_j \leq 0, &\quad \forall \left ( i, j \right ) \in \mathfrak{I}
\end{aligned}
$$

above only kinda, this would have duplicated... the way I have written it

$$
\hat{y}_i \preceq \hat{y}_j \rightarrow \hat{y}_i  - \hat{y}_j \leq 0 \text{and other side}
$$

where there is both subject tos for each $i$ two for each
has to allow for slackness in both directions.

then our Lagrange for a specific point is
each of our subject twos get their own $\lambda_i$

$$
L(y_i, \bm{\lambda}, \bm{v})_i = f^{loss}_i(y_i)  + \sum_j \lambda_j \left ( \hat{y}_i  - \hat{y}_j \right ) + \sum_k \lambda_k \left ( \hat{y}_k  - \hat{y}_i \right )
$$

and our KKT for each $i$ (for the first derive by $y_i$)

$$\begin{aligned}
\delta f_0(y_i^*) + \sum_j \lambda_j^* - \sum_k \lambda_k^* &= 0 \\
y_i^* - y_j^*                              &\leq 0, &\quad \forall (i, j) \in \mathfrak{I} \\
\text{there is no equality constraint} \\
\lambda_i^*                                &\geq 0, &\quad \forall i \\
\lambda_i^* \left ( y_i^* - y_j^* \right ) &= 0,    &\quad \forall (i, j) \in \mathfrak{I}
\end{aligned}
$$

so our optimal solution, satisfies the constraints and
the last condition implies that all $y_i^*$ are the same
if $\lambda > 0$. So our optimal solution will have blocks
of monotonic values $y_i^*$ with each estimate in a block
being identical due to lamba > 0. In cases where lamba
is equal to zero, the estimates can be different/separable.

solution is made of blocks with lambda > 0
with the estimate $\hat{y}$ for each block being identical
for each point in the block

isotonicity between blocks is guaranteed by the second condition above
the last says each subset has the same estimate $y^*$
which minimises the loss in each block

so our optimal solution is a partitioning of the space into $N$ blocks
each of whose optimal value minimises our function $f$ within the block

roughly, the algorithm

1. chooses the block which currently has the highest loss
2. solves a linear program on this block
   if the solution partitions the block then we have two blocks to consider in the future
   otherwise the result of this subset is considered optimal

```{=html}
<div id="UniIsoPlot"></div>
```

determining how and whether to partition points within a block is solved
as an optimal cut problem (which can be solved as a Linear Program) 
[^EquivalentMaxVariance]
this we conclude from the requirement that the optimal solution
satisfies

[^EquivalentMaxVariance]: This is equivalent to maximising the between-group
variance at each step. This is detailed in [@LussRossetEfficientRegularizedIsotonicRegression]

$$
\sum_i \frac{\delta f_0(y_i)}{\delta y_i} = 0
$$

consider three points a < b < c

then summing over first kkt gives

$$
\begin{aligned}
a&: 0 - \lambda_{ab} - \lambda_{ac} \\
b&: \lambda_{ab} - \lambda_{bc} \\
c&: \lambda_{ac} + \lambda_{bc} \\
\end{aligned}
$$

which sums to zero leading to the requirement above

so if a group isn't optimal we will be able to find a partitioning

$$
\sum_i \frac{\delta f_0(y_i)}{\delta y_i} - \sum_i \frac{\delta f_0(y_i)}{\delta y_i} < 0
$$

such a partition increases the estimate of larger ys
while decreasing estimate of smaller ys
thus maintaining isotonicity while improving the overall objective

consequently we want

$$
\text{minimise}_{\left ( V^-, V^+ \right ) \in V} \left ( \sum_{i \in V^+} \frac{\partial f_i(\hat{y}_i)}{\partial \hat{y}_i} - \sum_{i \in V^-} \frac{\partial f_i(\hat{y}_i)}{\partial \hat{y}_i} \right )
$$

[^MinimisationVsMaximisation]

[^MinimisationVsMaximisation]: Previous work from the author of the here
discussed paper typcially considers a maximisation problem instead of
the minimisation problem we discuss here. See for example,
[@LussRossetDecomposingIsotonicRegression{}, equation 3] and
[@LussRossetEfficientRegularizedIsotonicRegression{}, equation 3].
These two variants are equivalent, but the order of comparison is reversed. In
this paper, we take the derivative of the $L_2$ loss to be $2 (\hat{y}_i - y_i)$
whereas, the previous works take this to be $2 (y_i - \hat{y}_i)$. The rest of
the equations presented here remain almost identical if we switch from a
minimisation to a maximisation. In the case of the matrix form below, however,
we both switch from a maximisation to a minimisation and swap the domain of
our varables $y_i$ from $y_i \geq 0$ to $y_i \leq 0$. The full implementation
following the authors work, actually implements the opposite variant.

which leads to the binary program

$$\begin{aligned}
&\text{minimise}   &\quad \sum_{i \in V} x_i \frac{\partial f_i(\hat{y}_i)}{\partial \hat{y}_i} \\
&\text{subject to} &\quad x_i \leq x_j                      &\quad \forall (i,j) \in \mathfrak{I} \\
&                  &\quad x_i \in \left \{ -1, +1 \right \} &\quad \forall i \in V
\end{aligned}
$$

which then through the relaxation of the constraint leads to linear program
used in the paper's algorithm. $-1 \leq x_i \leq 1$

$$\begin{aligned}
&\text{minimise}   &\quad \sum_{i \in V} x_i \frac{\partial f_i(\hat{y}_i)}{\partial \hat{y}_i} \\
&\text{subject to} &\quad x_i \leq x_j       &\quad \forall (i,j) \in \mathfrak{I} \\
&                  &\quad -1 \leq x_i \leq 1 &\quad \forall i \in V
\end{aligned}
$$

can also put this in our standard form from above

$$\begin{aligned}
&\text{minimise}     &\quad \sum_{i \in V} x_i \frac{\partial f_i(\hat{y}_i)}{\partial \hat{y}_i} \\
&\text{subject to}_i &\quad  x_i - x_j \leq 0 &\quad \forall (i,j) \in \mathfrak{I}_V \\
&                    &\quad -x_i - 1   \leq 0 &\quad \forall i \in V \\
&                    &\quad  x_i - 1   \leq 0 &\quad \forall i \in V
\end{aligned}
$$

so this zTc loss value that weights what to do next
is somehow a measure of how well separated the points are either side of the most recent split
loss could be really large, but there might be little potentially gained
but this derivative measure says more something like
the points on the right of split are on mostly larger than the estimator
and the points on the left are mostly smaller.
it might be worth normalising this by the number of points though
which isn't done in the post or code

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

## How does this paper differ?

Regularisation

The approach in the paper
- support convex differentiable functions

supports multidimension

https://www.stat.umn.edu/geyer/8054/notes/isotonic.pdf

TO DELETE
$$\begin{aligned}
&\text{minimise}   &\quad f_0(\bm{x}) \\
&\text{subject to} &\quad f_i(\bm{x}) \leq 0, \quad \forall i \\
&                  &\quad h_j(\bm{x}) = 0,    \quad \forall j
\end{aligned}
$$
TO DELETE

---

### References

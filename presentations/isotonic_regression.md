---
title: Isotonic Regression
theme: serif
include_plotly: true
include_d3: true
js_file: /js/GeneralisedIsotonicRegressionPlots.min.js
---

<section>

  <section>

<h2>Isotonic Regression</h2>
<h4>Edward Wall</h4>

<aside class="notes">
</aside>

  </section>
  <section>

<h2>Isotonic Regression</h2>
$$
\begin{aligned}
&\text{minimise}   &\quad \sum_i \text{loss} (\hat{y}_i, y_i) & \\
&\text{subject to} &\quad \hat{y}_i \preceq \hat{y}_j, &\quad \forall \left ( i, j \right ) \in \mathfrak{I}
\end{aligned}
$$

* $y_i$: input/measured value
* $\hat{y}_i$: regressed value
* $\mathfrak{I}$: constraints

<aside class="notes">
* Generalised Isotonic Regression Framework
* Very general formulation; the constraints can be expressed via
  any arbitrary graph.
</aside>

  </section>
  <section>

<div id="UniIsoPlot"></div>

<aside class="notes">
* example
* fitting to blue points
* green is typical linear regression
* and orange is the result of the algorithm I implemented
* comparatively less shape constraints (linear, polynomial)
* spoiler, can visualise iterative algorithm
</aside>

  </section>
  <section>

<h3>Constraints of existing solutions</h3>

* either limited to 2-dimensions
* or process only a few hundred points
* or only support grids in 3-dimensions
* limited loss function support

<aside class="notes">
* there are plenty of implementatons available to solve
  this 1-dimensional version
* especially non-discretised form
* searched for available implementation
    1. written in R only handle few hundred points not graph etc
    2. in cpp
* both could only supporting grids, not arbitrary loss only l2 l1
</aside>

  </section>
  <section>

![](../img/presentations/GeneralisedIsotonicRegressionAbstract.png)

<aside class="notes">
* potential solution is 2014 ...
* what is it about
* set about understanding and implementing to plug this gap
  in open source
</aside>

  </section>
  <section>

$$
\begin{aligned}
&\text{minimise}   &\quad \sum_i \text{loss} (\hat{y}_i, y_i) & \\
&\text{subject to} &\quad \hat{y}_i \preceq \hat{y}_j, &\quad \forall \left ( i, j \right ) \in \mathfrak{I}
\end{aligned}
$$

<hr>

<div class="fragment fade-in">
$$\begin{aligned}
&\text{minimise}    &\quad \sum_{i \in V} x_i \frac{\partial \text{loss}_i(\hat{y}_V, y_V)}{\partial \hat{y}_V} \\
&\text{subject to}  &\quad  x_i - x_j \leq 0 &\quad \forall (i,j) \in \mathfrak{I}_V \\
&                   &\quad -x_i - 1   \leq 0 &\quad \forall i \in V \\
&                   &\quad  x_i - 1   \leq 0 &\quad \forall i \in V
\end{aligned}
$$
</div>

<aside class="notes">
* paper provides algorithm to change the
* ISO Problem (in case of L2 quadratic constraint weakly polynomial)
* into an iterative series of linear constraint problems faster solve
* if you would like details
    * the paper is freely available on arxiv
    * or you can check my blog post on it
</aside>

  </section>
  <section>

<h3>C++ Implementation</h3>
* higher potential speed
* interface to many languages with \
  minimal adjustments

<aside class="notes">
* lower memory usage
* decided to do in Cpp, for the speed and to minitgate ram usage
* and means, can somewhat easily add bindings for other languages
* supporting more at once, e.g R and Python
* I have even worked on a WASM interface which demo at end
</aside>

  </section>

<section data-background-iframe="../posts/2023-11-22-generalised-isotonic-regression.html"
         data-background-interactive
         data-preload>
</section>

  <section>

<pre><code class="language-plaintext" data-trim data-noescape>
FetchContent_Declare(
    GIR
    GIT_REPOSITORY "https://github.com/ewal31/GeneralisedIsotonicRegression"
    GIT_TAG 0.3.0
    GIT_SHALLOW TRUE
)

FetchContent_MakeAvailable(
    GIR
)
</code></pre>

<pre><code class="language-cpp" data-trim data-noescape>
std::tuple&lt;Eigen::SparseMatrix&lt;bool&gt;, VectorXu, VectorXu&gt;
points_to_adjacency(
    const Eigen::MatrixX&lt;V&gt;&amp; points
);

std::pair&lt;VectorXu, Eigen::VectorXd&gt;
generalised_isotonic_regression(
    const Eigen::SparseMatrix&lt;bool&gt;&amp; adjacency_matrix,
    YType&amp;&amp; _y,
    WeightsType&amp;&amp; _weights,
    const LossFunction&lt;LossType&gt;&amp; loss_fun,
    uint64_t max_iterations = 0
);
</code></pre>


<aside class="notes">
</aside>

  </section>
  <section>

<pre><code class="language-python" data-trim data-noescape style="max-height: 100%" data-line-numbers="|13,19-24">
import multivariate_isotonic_regression as mir

# create some test roughly monotonic points
X = np.array([(i, j)
              for i in range(width)
              for j in range(width)])

y = 3 * ((X[:, 0] // 7) + (X[:, 1] // 4)) + 1 + \
    1.5 * np.random.rand(width ** 2)

# points_to_adjacency rearranges the points roughly
# according to how many other points they dominate
adj, orig_idxs, new_idxs = mir.points_to_adjacency(X)

# we rearrange X and y to have the same ordering
X_reordered = X[new_idxs, :]
y_reordered = y[new_idxs]

group, yhat = mir.generalised_isotonic_regression(
    adj,
    y_reordered,
    loss_function = "pnorm",
    p = 1.1
)
</code></pre>

<aside class="notes">
probs show same picture from above
</aside>

  </section>
  <section>

![](../img/presentations/Multidim_Example.png)

  </section>
  <section>

<h3>Implementation Advantages</h3>

* open source
* more than 2-dimensions
* arbitrary graph structures
* duplicates points
* any convex loss functions

<aside class="notes">
I did
* open source
* duplicate points

* compared to other implementations is only version that...
    * uses barely any RAM
    * can process 80 times more points in same time compared to original solution
* the implementation gets more frustrating when we want to allow for duplicate
  points
* it could handle 500'000 points with barely memory usage finishing faster than
  the dynamic programming solution while solving a much more general and
  complicated problem processes about 80 times more points in the same amount
  of time
</aside>

  </section>
  <section>

<h3>Next Steps</h3>

* interface more robust to user error
* R interface
* improve speed with more than 3-dimensions
* replace linear programming with faster \
  Maximum-Flow algorithm
* support even more loss functions

<aside class="notes">
* few potential options
    * easiest to try would be a divide and conquer that should be somethin like klog(n) check this
    * Directed Minimum Spanding Tree
    * efficient orthogonal range search
* Non-convex via submodular supports some more loss functions
</aside>

  </section>

<section data-background-iframe="../tools/isotonic_regression.html"
         data-background-interactive
         data-preload>

<aside class="notes">
* show simple
* show pnorm, close to median/mean
* show duplicate points
* compiled to WASM
* setup web visualisation so one can try out for their needs
</aside>

  </section>
  <section>

<h2>Thank You</h2>

<aside class="notes">
</aside>

</section>

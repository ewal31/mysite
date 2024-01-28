---
title: Generalised Isotonic Regression
summary: An implementation of the paper Generalised Isotonic Regression in C++ compiled to webassembly.
include_plotly: true
css_file: /css/isotonic_regression.css
js_file: /js/GeneralisedIsotonicRegressionTool.min.js
---

This page provides quick access to an [Emscripten](https://emscripten.org/) compiled version
of the C++ library [here](https://github.com/ewal31/GeneralisedIsotonicRegression). For anything
more than a few thousand points, you should prefer the library or contained CLI. With this
tool, however, you can visualise how the space is partitioned with each subsequent iteration
of the algorithm. For more background on the algorithm see the paper
[Generalized Isotonic Regression](https://arxiv.org/abs/1104.1779) or my attempt at explaining
it [here](../posts/2023-11-22-generalised-isotonic-regression.html).


```{=html}
<div id="regression-form">
    <div class="text-label-block">
        <div class="text-label-block-label">
            <label for="input">Input: </label>
        </div>
        <textarea id="input" name="input" rows="20"></textarea>
    </div>

    <div class="text-label-block">
        <div class="text-label-block-label">
            <label for="output">Result: </label>
            <div class="result-iteration">
                <label for="iteration">Iteration: </label>
                <select name="iteration" id="iteration-select" disabled>
                </select>
            </div>
        </div>
        <textarea id="output" name="output" rows="20" readonly></textarea>
    </div>

    <div class="buttons">
        <div class="option">
            <label for="iterations">Max Iterations: </label>
            <input type="text" id="iterations" name="iterations" value="0">
        </div>

        <div class="option">
            <label for="loss-param" id="loss-param-text">Loss Parameter: </label>
            <input type="text" id="loss-param" name="loss-param" value="0.5" disabled>
        </div>

        <div class="option">
            <label for="loss-function">Choose a Loss Function: </label>
            <select name="loss-function" id="loss-function">
                <option value="L2">L2</option>
                <option value="HUBER">Huber</option>
                <option value="POISSON">Poisson</option>
                <option value="PNORM">p-Norm</option>
            </select>
        </div>

        <div class="buttons-block">
            <button id="run">Run Regression</button>
            <button id="clear">Clear Logs</button>
            <button id="plot">Plot</button>
        </div>

        <div class="buttons-block">
            <button id="example-1">Example 1</button>
            <button id="example-2">Example 2</button>
            <button id="example-3">Example 3</button>
        </div>
    </div>

    <div class="text-label-block">
        <div class="text-label-block-label">
            <label for="console">Runtime Log: </label>
        </div>
        <textarea id="console" name="console" rows="20" readonly></textarea>
    </div>

</div>

<div style="clear: both;"></div>

<div id="IsoPlot"></div>
```

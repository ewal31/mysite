(function() {

    function new_plot_height(plot_id, mult=0.5) {
        // Relative to viewport
        var vh = mult * Math.max(document.documentElement.clientHeight || 0, window.innerHeight || 0);
        return Math.min(Math.max(300, vh), 1200);
    }

    function new_plot_width(plot_id, mult=1) {
        // Width of parent
        var parentNode = document.getElementById(plot_id).parentNode
        var vw = mult * Math.max(parentNode.clientWidth || 0, parentNode.innerWidth || 0);
        return vw;
    }

    function register_relayout_on_window_resize(
        plot_id,
        new_height = new_plot_height,
        new_width = new_plot_width
    ) {
        window.addEventListener('resize', function(){
            Plotly.relayout(plot_id, {
                width: new_width(plot_id),
                height: new_height(plot_id)
            });
        })
    }

    function get_column(rows, key) {
        return rows.map(function(row) { return row[key]; });
    }

    function round(val, digits = 4) {
        if (typeof(val) == "string") {
            val = Number.parseFloat(val)
        }
        var mult = 10 ** digits
        return Math.round((val + Number.EPSILON) * mult) / mult
    }

    /*
     * Plotly plot for 2-d LP Feasible Region.
     */
    const lp_plot = "LinearProgram";
    (function() {

        Plotly.newPlot(
            lp_plot,
            [
                {
                    x: [0, 1, 2, 3],
                    y: [10, 5, 0, -5],
                    type: 'scatter',
                    mode: "lines",
                    name: "5p + b <= s",
                },
                {
                    x: [-1, 0, 1, 2.5],
                    y: [5, 5, 5, 5],
                    type: 'scatter',
                    mode: "lines",
                    name: "b <= 5",
                },
                {
                    x: [0, 0, 0, 0],
                    y: [-2, 0, 5, 8],
                    type: 'scatter',
                    mode: "lines",
                    name: "p >= 0",
                },
                {
                    x: [-1, 0, 2, 2.5],
                    y: [0, 0, 0, 0],
                    type: 'scatter',
                    mode: "lines",
                    name: "b >= 0",
                },
                {
                    x: [-1, 0, 2, 2.5],
                    y: [9, 7, 3, 2],
                    type: 'scatter',
                    mode: "lines",
                    name: "2p + b = 7<br>Dual Solution",
                    visible: 'legendonly',
                },
            ],
            {
                width: new_plot_width(lp_plot),
                height: new_plot_height(lp_plot, 0.4),
                margin: {
                    l: 40,
                    r: 40,
                    b: 40,
                    t: 40,
                    pad: 0,
                },
                showlegend: true,
                xaxis: {
                    range: [-1, 2.5],
                    title: {text: 'p'}
                },
                yaxis: {
                    range: [-2, 8],
                    title: {text: 'b'}
                },
                shapes: [
                    {
                        type: 'path',
                        path: 'M0,0 L0,5 L1,5 L2,0 Z',
                        fillcolor: 'rgba(255, 140, 184, 0.5)',
                        line: {
                            width: 0
                        }
                    }
                ],
                annotations: [
                    {
                        x: 1,
                        y: 5,
                        text: '2 x 1 + 5 = 7',
                        showarrow: true,
                        ax: 30,
                    },
                    {
                        x: 0,
                        y: 5,
                        text: '2 x 0 + 5 = 5',
                        showarrow: true,
                        ax: -40,
                        ay: -45,
                    },
                    {
                        x: 2,
                        y: 0,
                        text: '2 x 2 + 0 = 4',
                        showarrow: true,
                        ax: 40,
                        ay: -25,
                    }
                ]
            },
            {
                scrollZoom: false,
                displayModeBar: false,
            },
        );
        register_relayout_on_window_resize(
            lp_plot,
            id => new_plot_height(id, 0.4),
        );

    })()

    /*
     * Plotly plot for 2-d Isotonic Regression.
     * - shows how the algorithm runs along 1 dimension
     *   with loss values along the bottom
     */
    const uni_iso_plot = "UniIsoPlot";
    d3.csv("/plotdata/uni-iso-regression.csv", function(err, rows){
        d3.csv("/plotdata/uni-iso-regression-iterations.csv", function(err, rows_anim){

            var active_iteration = 3;
            var iterations = { };
            var slider_steps = [];
            var frames = [];

            function get_label(iter) {
                return 'Loss: ' + round(iterations[iter].loss);
            }

            // Parse data into a usable format
            for (var i = 0; i < rows_anim.length; i++) {
                var row = rows_anim[i];

                var trace;
                if (!(trace = iterations[row.iter])) {
                    trace = iterations[row.iter] = {
                        x: [],
                        y: [],
                        groups: {},
                    };
                }
                trace.x.push(row.X_1)
                trace.y.push(row.y_fit)
                if (row.L2) {
                    trace.loss = row.L2
                }

                if (row.group_loss) {
                    var group = trace.groups[row.group] = {};
                    group.loss = round(row.group_loss, 3)
                    group.x    = Number.parseFloat(row.X_1)
                    group.y    = -0.4
                } else {
                    var group = trace.groups[row.group];
                    group.x = (
                        Number.parseFloat(row.X_1) + group.x) / 2
                }
            }

            function get_values(obj, subkey) {
                var result = [];
                for (key of Object.keys(obj)) {
                    result.push(obj[key][subkey]);
                }
                return result;
            }

            // Set up slider effect and values
            for (iter of Object.keys(iterations)) {
                slider_steps.push({
                    method: 'animate',
                    label: iter,
                    args: [
                        [iter],
                        {
                            mode: 'immediate',
                            transition: {duration: 300},
                            // TODO can't have redraw false with layout annotations
                            //      but can maybe change the Loss text to a data
                            //      annotation instead and set to redraw: false
                            //      improving performance
                            frame: {duration: 300},
                        }
                    ]
                });
            }

            // Build animation frames
            for (iter of Object.keys(iterations)) {
                var data = [];
                data[1] = {
                    x: iterations[iter].x.slice(),
                    y: iterations[iter].y.slice(),
                }
                data[3] = {
                    x: get_values(iterations[iter].groups, "x"),
                    y: get_values(iterations[iter].groups, "y"),
                    text: Object.keys(iterations[iter].groups),
                }

                var prev_x = false;
                var down = false;
                var idx = 0;
                var loss_text_x = get_values(iterations[iter].groups, "x");
                var loss_text_y = loss_text_x.
                    map(x => [x, idx++]).
                    sort((a, b) => a[0] - b[0]).
                    map(function (row){
                        if (down || (prev_x && Math.abs(row[0] - prev_x) < 2)) {
                            down = !down;
                        }
                        prev_x = row[0]
                        return [row[1], down ? -1.3 : -.9];
                    }).
                    sort((a, b) => a[0] - b[0]).
                    map(row => row[1]);

                data[4] = {
                    x: loss_text_x,
                    y: loss_text_y,
                    text: get_values(iterations[iter].groups, "loss"),
                }

                frames.push({
                    name: iter,
                    data: data,
                    layout: {
                        "annotations[0].text": get_label(iter),
                        "sliders[0].active": iter,
                    },
                })
            }

            Plotly.newPlot(
                uni_iso_plot,
                {
                    data: [
                        {
                            x: get_column(rows, "X_1"),
                            y: get_column(rows, "y"),
                            type: 'scatter',
                            mode: 'markers',
                            name: "Observations",
                        },
                        {
                            x: iterations[active_iteration].x.slice(),
                            y: iterations[active_iteration].y.slice(),
                            line: {shape: 'hvh'},
                            type: 'scatter',
                            mode: "lines",
                            name: "Isotonic Regression (L2)",
                        },
                        {
                            x: get_column(rows, "X_1"),
                            y: get_column(rows, "L2"),
                            type: 'scatter',
                            mode: "lines",
                            name: "Linear Regression (L2)",
                            visible: 'legendonly',
                        },
                        {
                            x: get_values(iterations[active_iteration].groups, "x"),
                            y: get_values(iterations[active_iteration].groups, "y"),
                            text: Object.keys(iterations[active_iteration].groups),
                            type: 'scatter',
                            mode: 'text',
                            name: 'Group',
                            textfont: {
                                color: '#803a36'
                            },
                        },
                        {
                            x: get_values(iterations[active_iteration].groups, "x"),
                            y: get_values(
                                   iterations[active_iteration].groups, "y"
                               ).map(function (val) {return -.9}),
                            text: get_values(iterations[active_iteration].groups, "loss"),
                            type: 'scatter',
                            mode: 'text',
                            name: 'Group Loss',
                            textfont: {
                                color: '#807d7d'
                            },
                        }
                    ],
                    layout: {
                        width: new_plot_width(uni_iso_plot),
                        height: new_plot_height(uni_iso_plot),
                        margin: {
                            l: 20,
                            r: 20,
                            b: 20,
                            t: 20,
                            pad: 0,
                        },
                        showlegend: true,
                        xaxis: {
                            range: [0, 31]
                        },
                        yaxis: {
                            range: [-1.5, 9.5]
                        },
                        annotations: [{
                            xref: 'paper',
                            yref: 'paper',
                            x: 0.1,
                            xanchor: 'left',
                            y: 0.9,
                            yanchor: 'bottom',
                            text: get_label(active_iteration),
                            font: {size: 19},
                            showarrow: false
                        }],
                        sliders: [{
                            pad: {l: 130, t: 55},
                            currentvalue: {
                                visible: false,
                            },
                            steps: slider_steps,
                            active: active_iteration,
                        }],
                        updatemenus: [{
                            x: 0,
                            y: 0,
                            yanchor: 'top',
                            xanchor: 'left',
                            showactive: true,
                            active: 1,
                            direction: 'left',
                            type: 'buttons',
                            pad: {t: 50, r: 10},
                            buttons: [{
                                method: 'animate',
                                args: [null, {
                                    mode: 'immediate',
                                    fromcurrent: true,
                                    transition: {duration: 300},
                                    frame: {duration: 1000}
                                }],
                                label: 'Play'
                            }, {
                                method: 'animate',
                                args: [[null], {
                                    mode: 'immediate',
                                    transition: {duration: 0},
                                    frame: {duration: 0}
                                }],
                                label: 'Pause'
                            }]
                        }],
                    },
                    config: {
                        scrollZoom: false,
                        displayModeBar: false,
                    },
                    frames: frames
                }
            );
            register_relayout_on_window_resize(
                uni_iso_plot,
            );
        });
    });

    /*
     * Plotly plot for 3-d Isotonic Regression.
     * - shows how the algorithm runs along 2 dimensions
     *   with loss values along the bottom
     */
    multi_iso_plot = "MultiIsoPlot";
    d3.csv("/plotdata/multi-iso-regression.csv", function(err, rows){
        d3.csv("/plotdata/multi-iso-regression-iterations.csv", function(err, rows_anim){

            var active_iteration = 7;
            var iterations = { };
            var slider_steps = [];
            var frames = [];

            function get_label(iter) {
                return 'Iteration: ' + iter + '<br>Loss: ' + round(iterations[iter].loss);
            }


            // The uniqe x and y values.
            var x = Array.from(new Set(get_column(rows, 'X_1')));
            var y = Array.from(new Set(get_column(rows, 'X_2')));

            // Put Linear regression result in correct form (row-wise)
            // for surface plot
            var z_reg = [];
            for (j = 0; j < y.length; j++) {
                z_reg.push([])
            }
            for (i = 0; i < x.length; i++) {
                for (j = 0; j < y.length; j++) {
                    z_reg[j].push(rows[i * y.length + j]["L2"])
                }
            }

            // Put each Isotonic step in the correct form (row-wise)
            // for surface plot
            var rows_anim_idx = 0;
            var iter = 0;
            while (true) {

                if (rows_anim_idx >= rows_anim.length) {
                    break
                }

                var z_iso = [];
                for (j = 0; j < y.length; j++) {
                    z_iso.push([])
                }
                for (i = 0; i < x.length; i++) {
                    for (j = 0; j < y.length; j++) {
                        z_iso[j].push(rows_anim[rows_anim_idx + i * y.length + j]["y_fit"])
                    }
                }

                iterations[iter] = {
                    x: x.slice(),
                    y: y.slice(),
                    z: z_iso,
                    loss: rows_anim[rows_anim_idx]["L2"],
                }

                iter++;
                rows_anim_idx += x.length * y.length;

            }

            // Set up slider effect and values
            for (iter of Object.keys(iterations)) {
                slider_steps.push({
                    method: 'animate',
                    label: iter,
                    args: [
                        [iter],
                        {
                            mode: 'immediate',
                            transition: {duration: 300},
                            // TODO can't have redraw false with layout annotations
                            //      but can maybe change the Loss text to a data
                            //      annotation instead and set to redraw: false
                            //      improving performance
                            frame: {duration: 300},
                        }
                    ]
                });
            }

            // Build animation frames
            for (iter of Object.keys(iterations)) {
                var data = [];
                data[1] = {
                    z: iterations[iter].z.slice(),
                }

                frames.push({
                    name: iter,
                    data: data,
                    layout: {
                        "annotations[0].text": get_label(iter),
                        "sliders[0].active": iter,
                    },
                })
            }

            Plotly.newPlot(
                multi_iso_plot,
                {
                    data: [
                        {
                            x: get_column(rows, 'X_1'),
                            y: get_column(rows, 'X_2'),
                            z: get_column(rows, 'y'),
                            mode: 'markers',
                            marker: {
                                size: 1,
                                color: 'rgb(0, 0, 0)',
                                opacity: 0.8
                            },
                            type: 'scatter3d',
                            name: "Observations",
                        },
                        {
                            x: iterations[active_iteration].x,
                            y: iterations[active_iteration].y,
                            z: iterations[active_iteration].z,
                            type: 'surface',
                            showscale: false,
                            opacity: 0.7,
                            colorscale: 'YlOrRd',
                            name: "Isotonic Regression (L2)",
                            showlegend: true,
                        },
                        {
                            x: x,
                            y: y,
                            z: z_reg,
                            type: 'surface',
                            showscale: false,
                            opacity: 0.7,
                            colorscale: 'Greens',
                            name: "Linear Regression (L2)",
                            visible: 'legendonly',
                            showlegend: true,
                        },
                    ],
                    layout: {
                        width: new_plot_width(multi_iso_plot),
                        height: new_plot_height(multi_iso_plot),
                        margin: {
                            l: 20,
                            r: 20,
                            b: 20,
                            t: 20,
                            pad: 0,
                        },
                        scene: {
                            // Starting rotation and position of 3d scene
                            camera: {
                                eye: {
                                    x: -0.4,
                                    y: -1.6,
                                    z: 0.6
                                },
                                center: {
                                    x: 0,
                                    y: 0,
                                    z: -0.15
                                }
                            },
                            xaxis: {
                                range: [0, 20]
                            },
                            yaxis: {
                                range: [0, 20]
                            },
                            zaxis: {
                                range: [0, 15]
                            },
                        },
                        showlegend: true,
                        annotations: [{
                            xref: 'paper',
                            yref: 'paper',
                            x: 1.03,
                            xanchor: 'left',
                            y: 0.5,
                            yanchor: 'bottom',
                            align: "left",
                            text: get_label(active_iteration),
                            font: {size: 19},
                            showarrow: false
                        }],
                        sliders: [{
                            pad: {l: 130, t: 55},
                            currentvalue: {
                                visible: false,
                            },
                            steps: slider_steps,
                            active: active_iteration,
                        }],
                        updatemenus: [{
                            x: 0,
                            y: 0,
                            yanchor: 'top',
                            xanchor: 'left',
                            showactive: true,
                            active: 1,
                            direction: 'left',
                            type: 'buttons',
                            pad: {t: 50, r: 10},
                            buttons: [{
                                method: 'animate',
                                args: [null, {
                                    mode: 'immediate',
                                    fromcurrent: true,
                                    transition: {duration: 300},
                                    frame: {duration: 1000}
                                }],
                                label: 'Play',
                            }, {
                                method: 'animate',
                                args: [[null], {
                                    mode: 'immediate',
                                    transition: {duration: 0},
                                    frame: {duration: 0}
                                }],
                                label: 'Pause'
                            }]
                        }],
                    },
                    config: {
                        displayModeBar: false,
                    },
                    frames: frames
                }
            );
            register_relayout_on_window_resize(multi_iso_plot);
        })
    });

    /*
     * 1,1------
     *   |      |
     *   ------1,1-------------
     *                  |      |
     *                 3,1     |
     *                  |      |
     *                  |     1,3--------
     *                  |                |
     *                  |               2,3 -----
     *                  |                        |
     *                  -------------------------3,3
     *
     *
     * Points to Adjacency Matrix Animation
     * Plotly plot for 3-d Isotonic Regression.
     * - shows how the algorithm runs along 2 dimensions
     *   with loss values along the bottom
     */
    (function() {
        var animation_state = 0;

        // TODO resize with window instead of just on reload
        const width = new_plot_width("AdjacencyMatrix")
        const height = width * 0.6
        const textwidth = 35;
        const textheight = 25;

        const svg = d3.select("#AdjacencyMatrix")
            .append("svg")
            .attr("width", width + 'px')
            .attr("height", height + 'px');

        /*
         * Frame 1. Graph showing points and links
         *          between
         */
        const points = [
            { x: width * 2 / 9, y: height * 2 / 9, text: "(1, 1)"},
            { x: width * 3 / 9, y: height * 3 / 9, text: "(1, 1)"},
            { x: width * 4 / 9, y: height * 4 / 9, text: "(3, 1)"},
            { x: width * 5 / 9, y: height * 5 / 9, text: "(1, 3)"},
            { x: width * 6 / 9, y: height * 6 / 9, text: "(2, 3)"},
            { x: width * 7 / 9, y: height * 7 / 9, text: "(3, 3)"},
        ]

        const links = [
            { source: 0, dest: 1, dir: true },
            { source: 1, dest: 0, dir: true, rev: true },
            { source: 1, dest: 2, dir: true },
            { source: 1, dest: 3, dir: true },
            { source: 2, dest: 5, dir: false },
            { source: 3, dest: 4, dir: true },
            { source: 4, dest: 5, dir: true },
        ]

        // Defines how an arrow head should be drawn
        svg
            .append("svg:defs")
            .append("svg:marker")
            .attr("id", "arrow")
            .attr("viewBox", "0 0 10 10")
            .attr("refX", 0)
            .attr("refY", 5)
            .attr("markerUnits", "strokeWidth")
            .attr("markerWidth", 8)
            .attr("markerHeight", 6)
            .attr("orient", "auto")
            .append("svg:path")
            .attr("d", "M 0 0 L 10 5 L 0 10 z")

        // Draw Points
        for (c of ['pointHoriz', 'pointVert']) {
            svg
                .selectAll(c)
                .data(points)
                .enter().append('text')
                .attr('class', c)
                .attr('x', d => d.x)
                .attr('y', d => d.y)
                .text(d => d.text)
                .attr({
                    //dy: 6, // account for text height
                    'text-anchor': 'middle',
                    'dominant-baseline': 'middle'
                })
        }

        // Draw Arrows
        svg
            .selectAll('links')
            .data(links)
            .enter().append('path')
            .attr('class', 'link')
            .attr('d', function(d) {
                op = d.rev ? (a, b) => a - b : (a, b) => a + b
                if (d.dir) {
                    return d3.svg.line()([
                        [op(points[d.source].x, textwidth), points[d.source].y],
                        [points[d.dest].x, points[d.source].y],
                        [points[d.dest].x, op(points[d.dest].y, -textheight)],
                    ])
                } else {
                    return d3.svg.line()([
                        [points[d.source].x, op(points[d.source].y, textheight - 10)],
                        [points[d.source].x, points[d.dest].y],
                        [op(points[d.dest].x, -textwidth), points[d.dest].y],
                    ])
                }
            })
            .attr({
                stroke: 'black',
                'stroke-width': 2,
                fill: 'none',
                'marker-end': 'url(\#arrow)'
            })

        /*
         * Frame 2. Matrix with 1's for simplified
         *          less than
         */
        var matrixcontents = []
        for (i = 0; i < points.length; i++) {
            for (j = 0; j < points.length; j++) {
                matrixcontents.push({
                    x: (i+2) / 9 * width,
                    y: (j+2) / 9 * height,
                    text: '0'
                })
            }
        }

        for (link of links) {
            var {source, dest} = link;
            matrixcontents[source + points.length * dest].text = "1";
        }

        // Draw Matrix Brackets
        svg
            .selectAll('matrixborder')
            .data([
                [
                    [1.7/9 * width, 1.5/9 * height],
                    [1.5/9 * width, 1.5/9 * height],
                    [1.5/9 * width, 7.5/9 * height],
                    [1.7/9 * width, 7.5/9 * height],
                ],
                [
                    [7.3/9 * width, 1.5/9 * height],
                    [7.5/9 * width, 1.5/9 * height],
                    [7.5/9 * width, 7.5/9 * height],
                    [7.3/9 * width, 7.5/9 * height],
                ],
            ]).enter().append('path')
            .attr('class', 'matrixborder')
            .attr('d', d => d3.svg.line()(d))
            .attr({
                stroke: 'black',
                'stroke-width': 0,
                'fill': 'none'
            })

        // Draw Matrix Contents
        svg
            .selectAll('matrixcontents')
            .data(matrixcontents)
            .enter().append('text')
            .attr('class', 'matrixcontents')
            .attr('x', d => d.x)
            .attr('y', d => d.y)
            .text(d => d.text)
            //.attr('dy', 6) // account for text height
            .attr({
                'text-anchor': 'middle',
                'dominant-baseline': 'middle',
                'visibility': 'hidden',
            })

        /*
         * Buttons. For switching between different states
         */
        var button_data = [
            [{x: -1 * (230 / 2 - 70 / 2) - 1, width: 70, height: 33, text: "Graph", state: 0, selected: true}],
            [{x:  (230/2 - 160 / 2) + 1, width: 160, height: 33, text: "Adjacency Matrix", state: 1, selected: false}]
        ]

        function update_adjacency_matrix_state(button) {
            var chosen_state = button.state;

            if (chosen_state == animation_state) {
                return;
            }

            animation_state = chosen_state

            button_group
                .selectAll('.button')
                .transition()
                .attr('fill', d => d.state == chosen_state ? '#f4faff' : '#ffffff')

            if (chosen_state == 0) {
                d3
                    .selectAll('.pointHoriz')
                    .transition()
                    .attr('x', d => d.x)
                    .duration(2000)
                    .delay(500)

                d3
                    .selectAll('.pointVert')
                    .transition()
                    .attr('y', d => d.y)
                    .duration(2000)
                    .delay(500)

                d3
                    .selectAll('.link')
                    .transition()
                    .attr('stroke-width', 2)
                    .duration(2500)
                    .delay(2000)

                d3
                    .selectAll('.matrixborder')
                    .transition()
                    .attr('stroke-width', 0)
                    .duration(500)

                d3
                    .selectAll('.matrixcontents')
                    .transition()
                    .attr('visibility', 'hidden')
                    .duration(500)

            } else if (chosen_state == 1) {
                d3
                    .selectAll('.pointHoriz')
                    .transition()
                    .attr('x', 1/9 * width)
                    .duration(2000)
                    .delay(500)

                d3
                    .selectAll('.pointVert')
                    .transition()
                    .attr('y', 1/9 * height)
                    .duration(2000)
                    .delay(500)

                d3
                    .selectAll('.link')
                    .transition()
                    .attr('stroke-width', 0)
                    .duration(700)

                d3
                    .selectAll('.matrixborder')
                    .transition()
                    .attr('stroke-width', 6)
                    .duration(2500)
                    .delay(2000)

                d3
                    .selectAll('.matrixcontents')
                    .transition()
                    .attr('visibility', 'visible')
                    .delay(2500)
            }

        }

        // Outside group containing rectangle for button
        // and text
        var button_group = svg
            .selectAll('button-group')
            .data(button_data)
            .enter().append('g')
            .attr('class', 'button-group')
            .attr('transform', d => 'translate(' + (width / 2 - d[0].width / 2 + d[0].x) + ',' + (height - 33 - 2) + ')')
            .on('click', d => update_adjacency_matrix_state(d[0]))
            .on({
                "mouseover": function(d) {
                    d3.select(this).style("cursor", "pointer");
                },
                "mouseout": function(d) {
                    d3.select(this).style("cursor", "default");
                }
            });

        // rectangle for button
        button_group
            .selectAll('button')
            .data(d => d)
            .enter().append('rect')
            .attr('class', 'button')
            .attr('fill', d => d.selected ? '#f4faff' : '#ffffff')
            .attr('width', d => d.width)
            .attr('height', d => d.height)
            .attr({
                rx: 2,
                ry: 2,
                stroke: '#bec8d9',
                'stroke-opacity': 1,
                'fill-opacity': 1,
                'stroke-width': '1px',
                'shape-rendering': 'crispEdges',
            })

        // text for button
        button_group
            .selectAll('buttontext')
            .data(d => d)
            .enter().append('text')
            .attr('class', 'buttontext')
            .attr('dx', d => d.width / 2)
            .attr('dy', d => d.height / 2)
            .text(d => d.text)
            .attr({
                'text-anchor': 'middle',
                'dominant-baseline': 'middle'
            })

    })()

})()

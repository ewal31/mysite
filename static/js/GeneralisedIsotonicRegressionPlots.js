(function() {

    function new_plot_height(plot_id, mult=0.5) {
        // Relative to viewport
        var vh = mult * Math.max(document.documentElement.clientHeight || 0, window.innerHeight || 0);
        return Math.min(Math.max(300, vh), 1200);
    }

    function new_plot_width(plot_id) {
        // Width of parent
        var parentNode = document.getElementById(plot_id).parentNode
        var vw = Math.max(parentNode.clientWidth || 0, parentNode.innerWidth || 0);
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

    uni_iso_plot = "UniIsoPlot";
    d3.csv("/plotdata/uni-iso-regression.csv", function(err, rows){
        d3.csv("/plotdata/uni-iso-regression-iterations.csv", function(err, rows_anim){
            var iterations = { }
            for (var i = 0; i < rows_anim.length; i++) {
                var trace;
                var row = rows_anim[i];
                var iter = row.iter;
                if (!(trace = iterations[iter])) {
                    trace = iterations[iter] = {
                        x: [],
                        y: [],
                        loss: [],
                    };
                }
                trace.x.push(row.X_1)
                trace.y.push(row.y_fit)
                trace.loss.push(row.L2)
            }

            function get_label(iter) {
                return 'Loss: ' + round(iterations[iter].loss[0]);
            }

            var active_iteration = 3;
            var slider_steps = [];
            for (iter of Object.keys(iterations)) {
                slider_steps.push({
                    method: 'animate',
                    label: iter,
                    args: [
                        [iter],
                        {
                            mode: 'immediate',
                            transition: {duration: 300}, //, easing: "cubic"},
                            frame: {duration: 300}, // can't have redraw false with annotations
                        }
                    ]
                });
            }

            var frames = [];
            for (iter of Object.keys(iterations)) {
                var data = [];
                data[1] = {
                    x: iterations[iter].x.slice(),
                    y: iterations[iter].y.slice(),
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
                            mode: 'markers',
                            type: 'scatter',
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
                        sliders: [{
                            pad: {l: 130, t: 55},
                            currentvalue: {
                                visible: false,
                            },
                            steps: slider_steps,
                            active: active_iteration,
                        }],
                        annotations: [{
                            xref: 'paper',
                            yref: 'paper',
                            x: 0.1,
                            xanchor: 'left',
                            y: 0.9,
                            yanchor: 'bottom',
                            text: get_label(iter),
                            font: {size: 19},
                            showarrow: false
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

    multi_iso_plot = "MultiIsoPlot";
    d3.csv("/plotdata/multi-iso-regression.csv", function(err, rows){
        d3.csv("/plotdata/multi-iso-regression-iterations.csv", function(err, rows_anim){
            var x = Array.from(new Set(get_column(rows, 'X_1')));
            var y = Array.from(new Set(get_column(rows, 'X_2')));

            var z_reg = [];
            for (j = 0; j < y.length; j++) {
                z_reg.push([])
            }
            for (i = 0; i < x.length; i++) {
                for (j = 0; j < y.length; j++) {
                    z_reg[j].push(rows[i * y.length + j]["L2"])
                }
            }

            var iterations = { }
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

            function get_label(iter) {
                return 'Iteration: ' + iter + '<br>Loss: ' + round(iterations[iter].loss);
            }

            var active_iteration = 7;
            var slider_steps = [];
            for (iter of Object.keys(iterations)) {
                slider_steps.push({
                    method: 'animate',
                    label: iter,
                    args: [
                        [iter],
                        {
                            mode: 'immediate',
                            transition: {duration: 300}, //, easing: "cubic"},
                            frame: {duration: 300}, // can't have redraw false with annotations
                        }
                    ]
                });
            }

            var frames = [];
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
                        sliders: [{
                            pad: {l: 130, t: 55},
                            currentvalue: {
                                visible: false,
                            },
                            steps: slider_steps,
                            active: active_iteration,
                        }],
                        annotations: [{
                            xref: 'paper',
                            yref: 'paper',
                            x: 1.03,
                            xanchor: 'left',
                            y: 0.5,
                            yanchor: 'bottom',
                            align: "left",
                            text: get_label(iter),
                            font: {size: 19},
                            showarrow: false
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

})()

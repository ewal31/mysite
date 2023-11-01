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

    uni_iso_plot = "UniIsoPlot";
    d3.csv(document.getElementById(uni_iso_plot).getAttribute("csvsrc"), function(err, rows){
        Plotly.newPlot(
            uni_iso_plot,
            [
                {
                    x: get_column(rows, "X_1"),
                    y: get_column(rows, "y"),
                    mode: 'markers',
                    type: 'scatter',
                },
                {
                    x: get_column(rows, "X_1"),
                    y: get_column(rows, "iso"),
                    line: {shape: 'hvh'},
                    type: 'scatter',
                }
            ],
            {
                width: new_plot_width(uni_iso_plot),
                height: new_plot_height(uni_iso_plot, 0.3),
                margin: {
                    l: 20,
                    r: 20,
                    b: 20,
                    t: 20,
                    pad: 0,
                },
                showlegend: false,
            },
            {
                scrollZoom: false,
                displayModeBar: false,
            }
        );
        register_relayout_on_window_resize(
            uni_iso_plot,
            new_height = (plot_id) => new_plot_height(plot_id, 0.3)
        );
    });

    multi_iso_plot = "MultiIsoPlot";
    d3.csv(document.getElementById(multi_iso_plot).getAttribute("csvsrc"), function(err, rows){
        var x = Array.from(new Set(rows.map(function(row) { return row["X_1"]; })));
        var y = Array.from(new Set(rows.map(function(row) { return row["X_2"]; })));

        var z_iso = [];
        for (i = 0; i < x.length; i++) {
            var z_row = [];
            for (j = 0; j < y.length; j++) {
                z_row.push(rows[i * y.length + j]["iso"])
            }
            z_iso.push(z_row);
        }

        Plotly.newPlot(
            multi_iso_plot,
            [
                {
                    x: x,
                    y: y,
                    z: z_iso,
                    type: 'surface',
                    showscale: false,
                    opacity: 0.7,
                },
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
                    type: 'scatter3d'
                },
            ],
            {
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
                            x: -0.3,
                            y: -1.5,
                            z: 0.6
                        },
                        center: {
                            x: 0,
                            y: 0,
                            z: -0.2
                        }
                    }
                },
                showlegend: false,
            },
            {
                displayModeBar: false,
            }
        );
        register_relayout_on_window_resize(multi_iso_plot);
    });

})()

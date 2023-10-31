function new_plot_height(plot_id) {
    // Half viewport height
    var vh = 0.5 * Math.max(document.documentElement.clientHeight || 0, window.innerHeight || 0);
    return Math.min(Math.max(300, vh), 1200);
}

function new_plot_width(plot_id) {
    // Width of parent
    var parentNode = document.getElementById(plot_id).parentNode
    var vw = Math.max(parentNode.clientWidth || 0, parentNode.innerWidth || 0);
    return vw;
}

function create_2d_plot(plot_id) {
    d3.csv(document.getElementById(plot_id).getAttribute("csvsrc"), function(err, rows){
        var data = [
            {
                x: rows.map(function(row) { return row["X_1"]; }),
                y: rows.map(function(row) { return row["y"]; }),
                mode: 'markers',
                type: 'scatter',
            },
            {
                x: rows.map(function(row) { return row["X_1"]; }),
                y: rows.map(function(row) { return row["iso"]; }),
                line: {shape: 'hvh'},
                type: 'scatter',
            }
        ];

        var layout = {
            width: new_plot_width(plot_id),
            height: new_plot_height(plot_id),
            margin: {
                l: 20,
                r: 20,
                b: 20,
                t: 20,
                pad: 0,
            },
            showlegend: false,
        };

        var config = {
            scrollZoom: true,
            displayModeBar: false,
        };

        Plotly.newPlot(plot_id, data, layout, config);

        window.addEventListener('resize', function(){
            Plotly.relayout(plot_id, {
                width: new_plot_width(plot_id),
                height: new_plot_height(plot_id)
            });
        })
    })
};

create_2d_plot("UniIsoPlot");

function create_3d_plot(plot_id) {
    d3.csv(document.getElementById(plot_id).getAttribute("csvsrc"), function(err, rows){

        function unpack(rows, key) {
            return rows.map(function(row) { return row[key]; });
        }

        var z_data=[ ]
        for(i=0;i<24;i++) {
            z_data.push(unpack(rows,i));
        }

        var data = [
            {
                z: z_data,
                type: 'surface',
                showscale: false,
            }
        ];

        var layout = {
            title: 'Mt Bruno Elevation',
            width: new_plot_width(plot_id),
            height: new_plot_height(plot_id),
            scene: {
                camera: {
                    eye: {
                        x: 1.2,
                        y: 1.2,
                        z: 0.1
                    },
                    center: {
                        x: 0,
                        y: 0,
                        z: -0.2
                    }
                }
            },
            margin: {
                l: 20,
                r: 20,
                b: 20,
                t: 40,
                pad: 0,
            }
        };

        var config = {
            scrollZoom: true,
            displayModeBar: false,
        };

        Plotly.newPlot(plot_id, data, layout, config);

        window.addEventListener('resize', function(){
            Plotly.relayout(plot_id, {
                width: new_plot_width(plot_id),
                height: new_plot_height(plot_id)
            });
        })
    })
};

create_3d_plot("3dPlot");

// for (plotcontainer of document.getElementsByClassName("plotlycontainer")) {
//     create_plot(
//         plotcontainer.id,
//         plotcontainer.getAttribute("csvsrc")
//     );
// }

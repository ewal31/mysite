function getNewPlotHeight() {
    // Currently half the viewport height
    var vh = 0.5 * Math.max(document.documentElement.clientHeight || 0, window.innerHeight || 0);
    return Math.min(Math.max(600, vh), 1200);
}

function getNewPlotWidth() {
    // Currently the width of the parent
	var parentNode = document.getElementById("UniIsoPlot").parentNode
    var vw = Math.max(parentNode.clientWidth || 0, parentNode.innerWidth || 0);
    return vw;
}

d3.csv('https://raw.githubusercontent.com/plotly/datasets/master/api_docs/mt_bruno_elevation.csv', function(err, rows){
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
        width: getNewPlotWidth(),
        height: getNewPlotHeight(),
    };

    var config = {
        scrollZoom: true,
        displayModeBar: false,
    };

    Plotly.newPlot('UniIsoPlot', data, layout, config);
});

window.addEventListener('resize', function(){
    Plotly.relayout("UniIsoPlot", {
        width: getNewPlotWidth(),
        height: getNewPlotHeight()
    });
})

// Canvas setup
let canvas = new fabric.Canvas('canvas');
canvas.isDrawingMode = true;
canvas.freeDrawingBrush.width = 20;
canvas.freeDrawingBrush.color = "#ffffff";
canvas.backgroundColor = "#000000";
canvas.renderAll();

// Retrieve prediction model to be used from selector
let clf_path = $('#clf_model').val();
$(document.body).on('change', '#clf_model', _ => {
    clf_path = $('#clf_model').val();
});

// Clear button callback
$("#clear-canvas").click(_ => {
	canvas.clear();
	canvas.backgroundColor = "#000000";
	canvas.renderAll();
	updateChart(zeros);
	$("#status").removeClass();
});

// Predict button callback
$("#predict").click(_ => {

	if (clf_path == "Choose the classification model" ) {
		alert('Choose a classification model first!');
		return;
	}

	// Change status indicator
	$("#status").removeClass().toggleClass("fa fa-spinner fa-spin");

	// Get canvas contents as url
	let fac = (1.) / 13.;
	let drawing_data = canvas.toDataURLWithMultiplier('png', fac);
	
	// Post url to python script
	let jq = $.post(`/${clf_path}/?drawing_data=${drawing_data}`)
		.done( (json) => {
			if (json.result) {
				$("#status").removeClass().toggleClass("fa fa-check");
				$('#svg-chart').show();
				updateChart(json.data);
			} else {
				$("#status").removeClass().toggleClass("fa fa-exclamation-triangle");
				console.log(`Script Error: {json.error}`);
			}
		})
		.fail( (xhr, textStatus, error) => {
			$("#status").removeClass().toggleClass("fa fa-exclamation-triangle");
			console.log("POST Error: " + xhr.responseText + ", " + textStatus + ", " + error);
		}
		);

});

// Decision Tree Predictions Chart
$('#svg-chart').hide();

let labels = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9'];
let zeros = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0];

let margin = { top: 0, right: 0, bottom: 50, left: 0 },
	width = 360 - margin.left - margin.right,
	height = 180 - margin.top - margin.bottom;

let svg = d3.select("svg")
	.attr("width", width + margin.left + margin.right)
	.attr("height", height + margin.top + margin.bottom)
	.append("g")
	.attr("transform", "translate(" + margin.left + "," + margin.top + ")");

let x = d3.scale.ordinal()
	.rangeRoundBands([0, width], .1)
	.domain(labels);

let y = d3.scale.linear()
	.range([height, 0])
	.domain([0, 1]);

let xAxis = d3.svg.axis()
	.scale(x)
	.orient("bottom")
	.tickSize(0);

svg.selectAll(".bar")
	.data(zeros)
	.enter().append("rect")
	.attr("class", "bar")
	.attr("x", function (d, i) { return x(i); })
	.attr("width", x.rangeBand())
	.attr("y", function (d) { return y(d); })
	.attr("height", function (d) { return height - y(d); });

svg.append("g")
	.attr("class", "x axis")
	.attr("transform", "translate(0," + height + ")")
	.call(xAxis);

// Update chart data
function updateChart(data) {
	d3.selectAll("rect")
		.data(data)
		.transition()
		.duration(500)
		.attr("y", function (d) { return y(d); })
		.attr("height", function (d) { return height - y(d); });
}

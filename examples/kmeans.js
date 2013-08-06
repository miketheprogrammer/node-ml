var Canvas = require('canvas')
var canvas = new Canvas(400,400);
var ctx = canvas.getContext('2d');
var height = 400;
var width = 400;
var KMeans = require("../lib/index").classifiers.KMeans;
var readline = require("readline");
var rl = readline.createInterface({
    input: process.stdin,
    output: process.stdout
});
var data = [
    [1, 2],
    [2, 1],
    [2, 4], 
    [1, 3],
    [2, 2],
    [3, 1],
    [1, 1],

    [7, 3],
    [8, 2],
    [6, 4],
    [7, 4],
    [8, 1],
    [9, 2],

    [10, 8],
    [9, 10],
    [7, 8],
    [7, 9],
    [8, 11],
    [9, 9],
];

var x = new KMeans(data, 3);

x.train(function(err, trainedModel) {

    if ( err ) {
        console.log(err);
        return;
    }
    var inp = function() {
        rl.question("Input: ", function(answer) {
            var arr = answer.split(",");
            var x = parseFloat(arr[0]);
            var y = parseFloat(arr[1]);
            trainedModel.perceive([x,y], function(err, result) {
                console.log(err);
                console.log(result);
                inp();
            });
        });
    }
    inp();
});

function draw(assignments, dataExtremes, data, means) {

    ctx.clearRect(0,0,width, height);

    ctx.globalAlpha = 0.3;
    for (var point_index in assignments)
    {
        var mean_index = assignments[point_index];
        var point = data[point_index];
        var mean = means[mean_index];

        ctx.save();

        ctx.strokeStyle = 'blue';
        ctx.beginPath();
        ctx.moveTo(
            (point[0] - dataExtremes[0].min + 1) * (width / (dataExtremes[0].range() + 2) ),
            (point[1] - dataExtremes[1].min + 1) * (height / (dataExtremes[1].range() + 2) )
        );
        ctx.lineTo(
            (mean[0] - dataExtremes[0].min + 1) * (width / (dataExtremes[0].range() + 2) ),
            (mean[1] - dataExtremes[1].min + 1) * (height / (dataExtremes[1].range() + 2) )
        );
        ctx.stroke();
        ctx.closePath();
    
        ctx.restore();
    }
    ctx.globalAlpha = 1;

    for (var i in data)
    {
        ctx.save();

        var point = data[i];

        var x = (point[0] - dataExtremes[0].min + 1) * (width / (dataExtremes[0].range() + 2) );
        var y = (point[1] - dataExtremes[1].min + 1) * (height / (dataExtremes[1].range() + 2) );

        ctx.strokeStyle = '#333333';
        ctx.translate(x, y);
        ctx.beginPath();
        ctx.arc(0, 0, 5, 0, Math.PI*2, true);
        ctx.stroke();
        ctx.closePath();

        ctx.restore();
    }

    for (var i in means)
    {
        ctx.save();

        var point = means[i];

        var x = (point[0] - dataExtremes[0].min + 1) * (width / (dataExtremes[0].range() + 2) );
        var y = (point[1] - dataExtremes[1].min + 1) * (height / (dataExtremes[1].range() + 2) );

        ctx.fillStyle = 'green';
        ctx.translate(x, y);
        ctx.beginPath();
        ctx.arc(0, 0, 5, 0, Math.PI*2, true);
        ctx.fill();
        ctx.closePath();

        ctx.restore();

    }

}
try {
    draw(x.assignments, x.dimensions, x.data, x.mu);
    canvas.toDataURL(function(err, str){
        console.log(err);
        //console.log(str);
    });
} catch ( e ) {}
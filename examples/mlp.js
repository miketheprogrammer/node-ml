var mlp = require("../lib/index").perceptrons.MultiLayerPerceptron;
var readline = require("readline");
var rl = readline.createInterface({
    input: process.stdin,
    output: process.stdout
});
var data = [
    [0.10, 0.03, 0],
    [0.11, 0.11, 0],
    [0.11, 0.82, 0],
    [0.13, 0.17, 0],
    [0.20, 0.81, 0],
    [0.21, 0.57, 1],
    [0.25, 0.52, 1],
    [0.26, 0.48, 1],
    [0.28, 0.17, 1],
    [0.28, 0.45, 1],
    [0.37, 0.28, 1],
    [0.41, 0.92, 0],
    [0.43, 0.04, 1],
    [0.44, 0.55, 1],
    [0.47, 0.84, 0],
    [0.50, 0.36, 1],
    [0.51, 0.96, 0],
    [0.56, 0.62, 1],
    [0.65, 0.01, 1],
    [0.67, 0.50, 1],
    [0.73, 0.05, 1],
    [0.73, 0.90, 0],
    [0.73, 0.99, 0],
    [0.78, 0.01, 1],
    [0.83, 0.62, 0],
    [0.86, 0.42, 1],
    [0.86, 0.91, 0],
    [0.89, 0.12, 1],
    [0.95, 0.15, 1],
    [0.98, 0.73, 0]
];
m = new mlp(data);


m.train(function(trainedModel) {
    var inp = function() {
        rl.question("Input: ", function(answer) {
            var arr = answer.split(",");
            var x = parseFloat(arr[0]);
            var y = parseFloat(arr[1]);
            trainedModel.perceive([x,y], function(result) {
                console.log(result);
                inp();
            });
        });
    }
    inp();
});


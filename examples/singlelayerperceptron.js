var SingleLayerPerceptron = require("../lib/index").perceptrons.SingleLayerPerceptron;
var readline = require("readline");
var rl = readline.createInterface({
    input: process.stdin,
    output: process.stdout
});

var inputs = 
[
    [ 0.72, 0.82 ], [ 0.91, -0.69 ], [ 0.46, 0.80 ],
    [ 0.03, 0.93 ], [ 0.12, 0.25 ], [ 0.96, 0.47 ],
    [ 0.79, -0.75 ], [ 0.46, 0.98 ], [ 0.66, 0.24 ],
    [ 0.72, -0.15 ], [ 0.35, 0.01 ], [ -0.16, 0.84 ],
    [ -0.04, 0.68 ], [ -0.11, 0.10 ], [ 0.31, -0.96 ],
    [ 0.00, -0.26 ], [ -0.43, -0.65 ], [ 0.57, -0.97 ],
    [ -0.47, -0.03 ], [ -0.72, -0.64 ], [ -0.57, 0.15 ],
    [ -0.25, -0.43 ], [ 0.47, -0.88 ], [ -0.12, -0.90 ],
    [ -0.58, 0.62 ], [ -0.48, 0.05 ], [ -0.79, -0.92 ],
    [ -0.42, -0.09 ], [ -0.76, 0.65 ], [ -0.77, -0.76 ] 
];

var outputs = [
-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1,
 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1 
];

slp = new SingleLayerPerceptron(inputs, outputs, 0.001);

var acceptInput = function(trainedModel) {
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
}

slp.train(acceptInput);

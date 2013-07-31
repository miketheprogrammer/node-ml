var SingleLayerPerceptron = require("../lib/index").perceptrons.SingleLayerPerceptron;
var readline = require("readline");
var rl = readline.createInterface({
    input: process.stdin,
    output: process.stdout
});


var inputs = [
    [1,1],
    [-1,-1]
];

var outputs = [
    1,
    -1
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

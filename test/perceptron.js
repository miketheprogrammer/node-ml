var SingleLayerPerceptron = require("../lib/index").perceptrons.SingleLayerPerceptron;
var test = require('tap').test;
var async = require('async');

test('SimpleModel', function (t) {
    var inputs = [
        [1,1],
        [-1,-1]
    ]

    var outputs = [
        1,
            -1
    ]
    slp = new SingleLayerPerceptron(inputs, outputs, 0.001);
    slp.train(function(trainedModel) {
        trainedModel.perceive([1,1], function(result) {
            t.same(result,1);
            trainedModel.perceive([-1,-1], function(result) {
                t.same(result,-1);
                t.end();
            });
        });
    });
});
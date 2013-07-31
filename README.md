node-ml
=======

A Collection of Machine Learning algorithms built for use with NodeJS

[![build status](https://secure.travis-ci.org/miketheprogrammer/node-ml.png)](http://travis-ci.org/miketheprogrammer/node-ml)


The Single Layer Perceptron
========

With the single layer perceptron is it possible to solve Linearly Seperable Problems. This makes the SLP a fast tool for solving
simple classification problems.
-------------------------------------------------------------------

The SLP takes as input a list of 1x2 vectors as in
````javascript
[
  [1,1],
  [-1,-1]
]
````
We must also provide the SLP a list of expected outputs for each vector. Currently the system only supports 1 | -1
These outputs define the side of the line the elements fall on. It is not important which value you give to which inputs.
Just that these inputs correspond in a linear way to the outputs.

So for the above input we prove
````javascript
[
  1,
  -1
]
````

Now the SLP will solve for where [1,1] === 1 and [-1,-1] === -1

Training the SLP
---------------
SingleLayerPerceptron(inputs, outputs, learningRate)
````javascript
slp = new SingleLayerPerceptron(inputs, outputs, 0.001);
slp.train(function(trainedModel) { 
    trainedModel.perceive([1,1], function(result) {
      console.log(result);
      //should print out 1
    }); 
    trainedModel.perceive([-1,-1], function(result) {
      console.log(result);
      //should print out -1
    });
});
````

Even Better remember the above trained model is a Line seperating a 2d dimension space from -1 to 1 
We can input any value in this range and get an output however this limited training set is a bad choice but heres some outputs

Obtained From Running examples/singlelayerperceptron2.js
````javascript
Input: 1,1
1
Input: -1,1
-1
Input: 1,-1
1
Input: -.5,1
1
Input: .5,-1
-1
Input: .2,.45634
1
Input: .2,-.45634
-1
Input: -.4,-.4
-1
Input: -1,-1
-1
````





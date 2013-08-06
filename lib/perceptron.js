var BaseModel = require("./base").BaseModel;
var util = require("util");
usePlotter = false;
/*
If you want to use Plotter first install 
sudo apt-get install gnuplot ghostscript
then npm install plotter
*/

var plot;
if (usePlotter) {
    var plot = require('plotter').plot;
}

function SingleLayerPerceptron(inputs, outputs, alpha) {
    BaseModel.call(this);
    this.inputs = inputs;
    this.outputs = outputs;
    this.weights = [ Math.random(), Math.random() ];
    this.alpha = alpha;
}

util.inherits(SingleLayerPerceptron, BaseModel);

SingleLayerPerceptron.prototype.train = function(callback) {
    var ref = this;
    callback = this.checkCallback(callback, 'trained');

    if ( this.inputs[0].length != 2 ) {
        process.nextTick(function() {
            callback(new Error("Inputs must be a 1x2 Vector") );
        });

        return false;
        
    }
    if ( Math.abs(this.inputs[0][0]) > 1 
         || Math.abs(this.inputs[0][1]) > 1 ) {
        process.nextTick(function() {
            callback(new Error("Input must be between -1 and 1") );
        });
        return false;
    }
    
    this.iteration = 0;
    var globalError = 0;

    var test = function() { if (globalError != 0) return false; else return true }

    var ref = this;
    var _train = function() {
        
        globalError = 0;
        for (var i = 0; i < ref.inputs.length; i++ ) {
            ref.perceive(ref.inputs[i], function(err, output) {
                output = output.out;
                var localError = ref.outputs[i] - output;
                if ( localError != 0) {
                    for ( var j = 0; j < 2; j++ ){
                        ref.weights[j] += ref.alpha * localError * ref.inputs[i][j];
                    }
                }
                globalError += Math.abs(localError);
            });
        }
        ref.iteration++;

        if ( test() ) {
            clearImmediate(_track)
            console.log(callback);
            callback(null, ref);
        } else {
            setImmediate(_train);
        }
    }
    var _track = setImmediate(_train);
}

//Returns 1 for Class 1, 0 for Class 2
SingleLayerPerceptron.prototype.perceive = function(input, callback) {
    callback = this.checkCallback(callback, 'response');
    var res = (((input[0] * this.weights[0] + input[1] * this.weights[1]) >= 0 ) ? 1 : -1 );
    callback(null, {in:input, out: res});
}


function MultiLayerPerceptron(trainingSet) {
    BaseModel.call(this);
    this._hiddenDims = 2;
    this._inputDims = trainingSet[0].length - 1;
    this._iteration = 0;
    this._restartAfter = 6000;
    this._hidden = [];
    this._inputs = [];
    this._workers = [];
    this._output = undefined;
    this._errors = [];
    this.networkMessage = "Network Initialized";
    this.initialize();
    this.load(trainingSet);
}

util.inherits(MultiLayerPerceptron, BaseModel);

var mlp = MultiLayerPerceptron;

mlp.prototype.train = function(callback) {
    var error;
    callback = this.checkCallback(callback,'trained');

    if ( this._data[0].length - 1 != this._inputDims ) {
        callback(new Error("Input does not match Configuration" ));
        return false;
    }
    do {
        error = 0;
        var ref = this;
        this._workers.forEach(function(worker) {
            var delta = worker.Output() - ref.activate(worker);
            ref.adjustWeights(delta);
            error += Math.pow(delta,2);
        });
        this._errors.push(error);
        this._iteration += 1;
        if (this._iteration > this._restartAfter) {
            var _data = this._data;
            this.networkMessage = "Local Minimum Found: Restaring Network";
            this._iteration = 0;
            this.initialize();
            this.load(_data);
            this.train();
        }
    } while ( error > 0.1 )

    if (usePlotter) {
        plot({
            data:       this._errors,
            filename:   'output.pdf'
        });
    }
    var ref = this;
    process.nextTick(function() {
        callback(null, ref);
    });
}

mlp.prototype.load = function(inputs) {
    var ref = this;
    this._data = inputs;
    this._workers = [];
    inputs.forEach(function(input) {
        ref._workers.push(new Worker(input));
    });
}

mlp.prototype.activate = function(worker) {
    for (var i = 0; i < worker._inputs.length; i++){
        this._inputs[i].Output(worker._inputs[i]);
    }

    this._hidden.forEach(function(neuron) {
        neuron.activate();
    });
    this._output.activate();
    return this._output.Output();
}

mlp.prototype.perceive = function(input, callback) {
    callback = this.checkCallback(callback, 'response');
    input.push(0); //set fake output var
    var result = this.activate(new Worker(input));
    input.pop();
    callback(null, {in:input, out: result});
}

mlp.prototype.adjustWeights = function(delta) {
    this._output.adjustWeights(delta);
    var ref = this;
    this._hidden.forEach(function getFeedback(neuron) {
        var out = ref._output.errorFeedback(neuron);
        if (out == undefined) {
            callback(new Error( "Critical Error while training model\n Received undefined output from a feedback neuron." ) );
            
        }
        neuron.adjustWeights(out);
    });
}

function Layer(size, layer) {
    var ret = [];
    if (layer == undefined) {
        for(var i = 0; i < size; i++ ) {
            ret.push(new Neuron());
        }
    } else {
        for (var i = 0; i < size; i++ ) {
            ret.push(new Neuron(layer));
        }
    }

    return ret;
    
}

mlp.prototype.initialize = function() {
    this._inputs = Layer(this._inputDims);
    this._hidden = Layer(this._hiddenDims, this._inputs);
    this._output = new Neuron(this._hidden);

    console.log(this.networkMessage);
}

function Neuron(inputs) {
    this._bias = 0;
    this._error = 0;
    this._input = 0;
    this._lambda = 6;
    this._learnRate = .5;
    this._output = Number.MIN_VALUE;
    this._weights = [];

    var ref = this;
    if (inputs != undefined){
        inputs.forEach(function(neuron) {
            var p = Math.random() * 2 - 1;
            ref._weights.push(new Weight(neuron, p));
            iter += 1;
        });
    }
}
var iter = 0;
Neuron.prototype.activate = function() {
    this._input = 0;
    var ref = this;
    this._weights.forEach(function(weight) {
        ref._input += weight.value * weight.input.Output();
    });
}

Neuron.prototype.errorFeedback = function(input) {
    var ref = this;
    for (var i = 0; i < this._weights.length; i++){
        var weight = this._weights[i];
        if (weight.input == input ){
            var v = ref._error * ref.Derivative() * weight.value;
            if (v == undefined){
                callback(new Error("Could not calculate error feedback"));
                return false;
            }
            return v
        } else {
        }
    }
}

Neuron.prototype.adjustWeights = function(value) {
    this._error = value;
    var ref = this;
    this._weights.forEach(function(weight) {
        weight.value += ref._error 
            * ref.Derivative() 
            * ref._learnRate 
            * weight.input.Output();
    });
    
    this._bias += this._error * this.Derivative() * this._learnRate;
    if (isNaN(this._bias)){
        callback(new Error("Bias is NaN. Please restart"));
    }
}

Neuron.prototype.Derivative = function() {
    var activation = this.Output();
    return activation * ( 1 - activation );
}

Neuron.prototype.Output = function(value) {
    if (value != undefined) {
        this._output = value;
    } else {
        if (this._output != Number.MIN_VALUE || this._output == 0){
            return this._output;
        } else {
            var o = 1 / (1 + Math.exp( (0-this._lambda) * (this._input + this._bias ) ) );
            return o
        }
    }
}

function Weight(input, value) {
    this.input = input;
    this.value = value;
}




function Worker( inputs ) {
    
    this._inputs = inputs.slice(0, inputs.length - 1);
    this._output = inputs[2];
    
}

Worker.prototype.Inputs = function() {
    return this._inputs;
}

Worker.prototype.Output = function() {
    return this._output;
}




exports.SingleLayerPerceptron = SingleLayerPerceptron;
exports.MultiLayerPerceptron = mlp;

var pretty = require( "../../sales_ops_services/shared/lib/pretty");
var console = new pretty.pretty(pretty.LAUNCHER);
console.log = function() {}
function SingleLayerPerceptron(inputs, outputs, alpha) {
    this.inputs = inputs;
    this.outputs = outputs;
    this.weights = [ Math.random(), Math.random() ];
    this.alpha = alpha;
}

SingleLayerPerceptron.prototype.train = function(callback) {
    this.iteration = 0;
    var globalError = 0;

    var test = function() { if (globalError != 0) return false; else return true }

    //We need a handle to the Model
    var ref = this;
    var _train = function() {
        globalError = 0;
        for (var i = 0; i < ref.inputs.length; i++ ) {
            ref.perceive(ref.inputs[i], function(output) {
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
            (callback != undefined) ? callback(ref) : false;
        } else {
            setImmediate(_train);
        }
    }
    var _track = setImmediate(_train);
}

//Returns 1 for Class 1, 0 for Class 2
SingleLayerPerceptron.prototype.perceive = function(input, callback) {
    callback( ( (input[0] * this.weights[0] + input[1] * this.weights[1]) >= 0 )? 1 : -1 );
}



function MultiLayeredPerceptron() {
    this._hiddenDims = 2;
    this._inputDims = 2;
    this._iteration = 0;
    this._restartAfter = 2000;
    this._hidden = [];
    this._inputs = [];
    this._workers = [];
    this._output = undefined;
}

var mlp = MultiLayeredPerceptron;

mlp.prototype.train = function() {
    var error;

    do {
        error = 0;
        var ref = this;
        this._workers.forEach(function(worker) {
            console.log("New Worker");
            console.log(worker.Output());
            var delta = worker.Output() - ref.activate(worker);
            ref.adjustWeights(delta);
            error += Math.pow(delta,2);
        });
        console.warn("Iteration: " + this._iteration + "  Error: " + error);
        this._iteration += 1;
        //if (this._iteration > this._restartAfter) this.initialize();
    } while ( error > 0.1 )
}

mlp.prototype.load = function(inputs) {
    var ref = this;
    inputs.forEach(function(input) {
        ref._workers.push(new Worker(input));
    });
}

mlp.prototype.activate = function(neuron) {
    for (var i = 0; i < neuron.Inputs.length; i++){
        this._inputs[i].Output(neuron.Inputs[i]);
    }

    this._hidden.forEach(function(neuron) {
        neuron.activate();
    });
    this._output.activate();
    return this._output.Output();
}

mlp.prototype.adjustWeights = function(delta) {
    this._output.adjustWeights(delta);
    var ref = this;
    this._hidden.forEach(function getFeedback(neuron) {
        
        
        var out = ref._output.errorFeedback(neuron)
        console.log("Expect Error feedback");
        if (out == undefined) {
            console.log("Critical Error");
            process.exit();
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

    console.log("Network Initialized");
}

function Neuron(inputs) {
    this._bias = 0;
    this._error = 0;
    this._input = 0;
    this._lambda = 6;
    this._learnRate = 0.5;
    this._output = 0.0;
    this._weights = [];

    var ref = this;
    if (inputs != undefined){
        inputs.forEach(function(neuron) {
            ref._weights.push(new Weight(neuron, Math.random()));
        });
    }
}

Neuron.prototype.activate = function() {
    this._input = 0;
    var ref = this;
    this._weights.forEach(function(weight) {
        console.log(weight.value);
        console.log(weight.input.Output());
        ref._input += weight.value * weight.input.Output();
        console.log("Whats up");
        console.log(ref._input);
    });
}

Neuron.prototype.errorFeedback = function(input) {
    var ref = this;
    console.log(this._weights);
    for (var i = 0; i < this._weights.length; i++){
        var weight = this._weights[i];
        console.log(weight.input);
        console.log(input);
        if (weight.input == input ){
            console.log("Found");
            var v = ref._error * ref.Derivative() * weight.value;
            console.log(v);
            if (v == undefined){
                console.log(ref._error);
                console.log(ref.Derivative());
                console.log(weight.value);
                process.exit();
            }
            
            return v
        }
    }
}

Neuron.prototype.adjustWeights = function(value) {
    this._error = value;
    var ref = this;
    console.log(this._weights);
    this._weights.forEach(function(weight) {
        weight.value += ref._error 
            * ref.Derivative() 
            * ref._learnRate 
            * weight.input.Output();
        console.log(ref.Derivative())
        console.log(ref._learnRate)
        console.log(ref._error)
        console.log(weight.input.Output());
        console.log(weight.value);
    });
    
    console.log(this._weights);

    this._bias += this._error * this.Derivative() * this._learnRate;

    console.log(this._bias);
    console.log(this._error, this.Derivative(), this._learnRate);
    if (isNaN(this._bias)){
        console.log(this._error, this.Derivative(), this._learnRate);
        process.exit();
    }
}

Neuron.prototype.Derivative = function() {
    var activation = this.Output();
    return activation * ( 1 - activation );
}

Neuron.prototype.Output = function(value) {
    if (value != undefined) 
        this._output = value;
    else {
        //console.log("hello");
        //console.log(this._lambda, this._input, this._bias);
        if (this._output != 0)
            return this._output;
        else return 1 / (1 + Math.exp( -this._lambda * (this._input + this._bias ) ) );
    }
}

function Weight(input, value) {
    this.input = input;
    this.value = value;
}




function Worker( inputs ) {
    this._inputs = inputs.slice(0, inputs.length - 1)
    this._output = inputs.slice(inputs.length - 1, inputs.length)[0];

    
}

Worker.prototype.Inputs = function() {
    return this._inputs;
}

Worker.prototype.Output = function() {
    return this._output;
}








exports.SingleLayerPerceptron = SingleLayerPerceptron;
exports.MultiLayerPerceptron = mlp;
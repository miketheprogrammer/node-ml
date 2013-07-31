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

exports.SingleLayerPerceptron = SingleLayerPerceptron;
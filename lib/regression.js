var s = require("sylvester")

function LinearRegression(trainingData) {
    
    for (var i = 0; i < trainingData.length; i++) {
        var input = trainingData[i];
        trainingData[i] = [1, input[0], input[1]];
    };
    this.data = $M(trainingData);
    this.x = this.data.slice(1,trainingData.length,2,2);
    this.y = this.data.slice(1,trainingData.length,3,3);
    this.m = trainingData.length;
}

LinearRegression.prototype.train = function(callback) {
    this.theta = s.Matrix.Zeros(2,1);
    var iterations = 1500;
    this.alpha = 0.01;

    var cost = this.computeCost(this.x, this.y, this.theta);
    this.theta = this.gradientDescent(this.x,this.y,this.theta,this.alpha, iterations);

    callback(this);
}

LinearRegression.prototype.predict = function(value, callback) {
    value = $M([[1,value]]);
    var mult = value.multiply(this.theta);

    callback(mult.sum());
}

LinearRegression.prototype.perceive = LinearRegress.prototype.predict;

LinearRegression.prototype.computeCost = function(x, y, theta) {
    
    var predictions = x.elementMultiply(theta);
    var tmp = predictions.subtract(y);
    var sqrErrors = tmp.elementMultiply(tmp);
    var J = 1/(2*this.m) * sqrErrors.sum();
    return J
}

LinearRegression.prototype.gradientDescent = function(x, y, theta, alpha, numIters) {
    for (var i = 0; i < numIters; i++) {
        var theta = this.theta;
        var theta_1 = theta.slice(1,1,1,1);
        var theta_2 = theta.slice(2,2,1,1);
        
        var h = x.multiply(theta_2.sum()).add(theta_1.sum());
        var theta_1_prime = theta_1.subtract(alpha * (1/this.m) * h.subtract(y).sum());
        var theta_2_prime = theta_2.subtract(alpha * (1/this.m) * x.elementMultiply(h.subtract(y)).sum());

        
        this.theta = $M([[theta_1_prime.sum()],[theta_2_prime.sum()]]);
    }
    return this.theta;

}

exports.LinearRegression = LinearRegression;
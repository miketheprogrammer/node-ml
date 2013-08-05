var underscore = require('underscore');


function KMeans(data, k) {
    this.data = data;
    this.k = k;
    this.assignments = [];
}

KMeans.prototype._initialize = function() {
    this.mu = [];

    //Holds the min and max of a dimension
    this.dimensions = [];
    for (var i = 0; i < this.data.length; i++ ) {
        for (var j = 0; j < this.data[i].length; j++) {
            if ( this.dimensions[j] == undefined ) {
                this.dimensions[j] = { 
                    min: this.data[i][j],
                    max: this.data[i][j],
                    range: function() {
                        return (this.max - this.min);
                    }
                }
            }
            
            if ( this.data[i][j] < this.dimensions[j].min ) 
                this.dimensions[j].min = this.data[i][j];
            
            if ( this.data[i][j] > this.dimensions[j].max )
                this.dimensions[j].max = this.data[i][j];
            
        }
    }

    for ( var i = 0; i < this.k; i++ ) {
        this.mu[i] = [];
        for ( var j = 0; j < this.dimensions.length; j++ ) {
            this.mu[i][j] = (this.dimensions[j].min + ( Math.random() * this.dimensions[j].range()) )
        }
    }
}


KMeans.prototype.train = function(callback) {
    //this.assignClusters();
    this._initialize();

    var moved = true;

    while(moved) {
        moved = this.moveCentroids();
    }

    callback(this);
}

KMeans.prototype.perceive = function(input, callback) { 
    var distances = [];

    for ( var i = 0; i < this.mu.length; i++ ) {
        var sum = 0;

        for ( var j = 0; j < input.length; j++ ) {
            sum += Math.pow(input[j] - this.mu[i][j], 2 );
        }

        distances[i] = Math.sqrt(sum);
    }

    callback(distances.indexOf( Math.min.apply(null, distances) ) );
}
KMeans.prototype.assignClusters = function() {
    
    for ( var i = 0; i < this.data.length; i++ ) {
        var distances = [];
        
        for ( var j = 0; j < this.mu.length; j++ ) {
            var sum = 0;

            for ( var k = 0; k < this.data[i].length; k++ ) {
                sum += Math.pow(this.data[i][k] - this.mu[j][k], 2);
            }
            distances[j] = Math.sqrt(sum);
        }
        
        this.assignments[i] = distances.indexOf( Math.min.apply(null, distances));
        if ( this.assignments[i] == -1 ) {
            console.log("Error, failed to converge" );
            process.exit();
        }

    }
}

KMeans.prototype.moveCentroids = function() {
    this.assignClusters();

    var sums = new Array( this.mu.length )
    , counts = new Array( this.mu.length )
    , moved = false;

    for ( var i = 0; i < this.mu.length; i++ ) {
        counts[i] = 0;
        sums[i] = new Array ( this.mu[i].length ) ;
        
        for ( var j = 0; j < this.dimensions.length; j++) {
            sums[i][j] = 0;
        }
    }

    for ( var i = 0; i < this.assignments.length; i++ ) {
        var mu_index = this.assignments[i];

        counts[mu_index] += 1;
        for( var j = 0; j < this.mu[mu_index].length; j++ ) {
            sums[mu_index][j] += this.data[i][j];
        }
    }

    for ( var i = 0; i < sums.length; i++ ) {
        for ( var j = 0; j < sums[i].length; j++ ) {
            sums[i][j] /= counts[i];
        }
    }

    if ( this.mu.toString() !== sums.toString() ) {
        moved = true;
    }

    this.mu = sums;

    return moved;
}

exports.KMeans = KMeans;
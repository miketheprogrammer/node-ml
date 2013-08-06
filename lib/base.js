var evt = require("events");
var util = require("util");

BaseModel = function() {
    evt.EventEmitter.call(this);
}

util.inherits(BaseModel, evt.EventEmitter);


/**
Callback Here is optional,
If no callback, a response event will be emitted
*/
BaseModel.prototype.perceive = function(input, callback) {
    this.emit('error', null);
    this.emit('response', null);
}

// We want the name conventions to relate to the style of model
// we are currently using, even though they generally have the same
// code;
BaseModel.prototype.predict = BaseModel.prototype.perceive;

BaseModel.prototype.checkCallback = function(callback, responseEvent) {
    var ref = this;
    callback = callback ? callback : function(err, response) {
        if ( err ) {
            ref.emit('error', err);
        }

        if ( response ) {
            ref.emit(responseEvent, response );
        }
    }

    return callback;

}

BaseModel.prototype.train = function() {
    this.emit('error', null);
    this.emit('trained', this);
}

exports.BaseModel = BaseModel;
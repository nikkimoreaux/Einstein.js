/*
 Project:   Einstein.js - v0.0.1
 Subject:   A JavaScript browser-based artificial neural network
 Copyright: 2014 - Nikki Moreaux - http://diplodoc.us
 License:   MIT
*/

// Polyfill
if (!Date.now) {
	Date.now = function now() {
		return new Date().getTime();
	};
}


// Constructor
var Einstein = function(options){
	
	var options = options || {};
	
	var allowed_options = [
		"hidden_layers",
		"learning_rate",
		"momentum",
		"targeted_mse",
		"max_training_time_ms",
		"overfitting_protection",
		"training_progress_callback"
	];
	for (var i = allowed_options.length - 1; i >= 0; i--) {
		if (options.hasOwnProperty(allowed_options[i])) {
			this["_"+allowed_options[i]] = options[allowed_options[i]];
		}
	}
	
	return this;
	
}

Einstein.prototype = {
	constructor: Einstein,
	
	// Options
	_hidden_layers: "auto",
	_learning_rate: 0.3,
	_momentum: 0.1,
	_targeted_mse: 0.005,
	_max_training_time_ms: 1000,
	_overfitting_protection: true,
	_training_progress_callback: false,
	
	// Stores
	_learned_patterns: [],
	_neural_network: [], // [layer][neuron], [inputs to outputs], [top to bottom]
	_neural_network_training_status: "TO_TRAIN",
	_neural_network_trained_callbacks: [],
	_neural_network_training_iterations: 0,
	_neural_network_training_start_time_ms: undefined,
	_test_mean_squared_errors: [],
	
	// Utils
	_logError: function(e){
		if(console && console.error)
			console.error("Einstein.js: "+e);
	},
	_isFun: function(f){
		return typeof f === "function";
	},
	_isNum: function(n){
		return typeof n === "number";
	},
	_isInt: function(n){
		return this._isNum(n) && n % 1 == 0;
	},
	_isArray: function(obj){
		return Object.prototype.toString.call( obj ) === "[object Array]";
	},
	_shuffleArray: function(array){
	    for (var i = array.length - 1; i > 0; i--) {
	        var j = Math.floor(Math.random() * (i + 1));
	        var temp = array[i];
	        array[i] = array[j];
	        array[j] = temp;
	    }
	    return array;
	},
	_randomWeight: function(){
		return Math.random() * 0.4 - 0.2;
	},
	_getAllKeysFromArraySubObjectKey: function(array,sub_object_key){
		// Used to find all different I/O keys
		var keys_obj = {};
		var keys_arr = [];
		for (var i = 0; i < array.length; i++) {
			for (key in array[i][sub_object_key]) {
				if (array[i][sub_object_key].hasOwnProperty(key) && !keys_obj.hasOwnProperty(key.toString())) {
					keys_obj[key.toString()] = 1;
					keys_arr.push(key.toString());
				}
			}
		}
		return keys_arr;
	},
	_objectIsIOCompatible: function(object){
		// Check if values of object are numbers, and between 0 and 1
		for(var i in object){
			if(object.hasOwnProperty(i)){
				if(!(this._isNum(object[i]) && object[i] >= 0 && object[i] <= 1)){
					return false;
				}
			}
		}
		return true;
	},
	
	// Neural network setup and training skeleton
	_resetNeuralNetwork: function(){
		this._neural_network = [];
		this._neural_network_training_status = "TO_TRAIN";
		this._neural_network_training_iterations = 0;
		this._neural_network_training_start_time_ms = undefined;
	},
	_setupAndTrainNeuralNetwork: function(){
		if(this._neural_network_training_status == "TO_TRAIN"){
			this._neural_network_training_status = "TRAINING";
			this._setupNeuralNetwork();
			this._neural_network_training_start_time_ms = Date.now();
			this._launchAsyncTrainer();
		}
		this._pingCallbacks();
	},
	_setupNeuralNetwork: function(){

		if(this._learned_patterns.length == 0){
			this._logError("You must learn() some patterns first");
			return false;
		}
		
		// Setup _neural_network
		
		// Inputs layer
		var neural_network_layers = [this._getAllKeysFromArraySubObjectKey(this._learned_patterns,"inputs")];
		
		// Hidden layers
		var hidden_layers = this._hidden_layers;
		if(hidden_layers == "auto"){
			hidden_layers = [Math.max(3,Math.ceil(neural_network_layers[0].length/2))];
		}
		for (var i = 0; i < hidden_layers.length; i++) {
			if(hidden_layers[i] > 0){
				neural_network_layers.push(new Array(hidden_layers[i]));
			}
		}
		
		// Outputs layer
		neural_network_layers.push(this._getAllKeysFromArraySubObjectKey(this._learned_patterns,"outputs"));
		

		// Push neuron layers
		for (var i = 0; i < neural_network_layers.length; i++){
			if(i == 0){
				this._neural_network.push(
					this._createNeuralLayer(
						neural_network_layers[i],false
					)
				);
			}else{
				this._neural_network.push(
					this._createNeuralLayer(
						neural_network_layers[i],neural_network_layers[i-1]
					)
				);
			}
		}
		
		return true;
	},
	_createNeuralLayer: function(current_layer,previous_layer){
		var neural_layer = new Array(current_layer.length);
		for (var i = 0; i < neural_layer.length; i++){
			
			if(previous_layer !== false){ // hidden/outputs neurons
				neural_layer[i] = {
				
					// Last output value
					output: 0,
					
					// Training helpers, bias, bias inertia, weights...
					error: Number.POSITIVE_INFINITY,
					delta: 0,
					
					bias: this._randomWeight(),
					// TODO Other functions
					function_mode: "sigmoid",
					
					weights: new Array(previous_layer.length)
				
				}
				for (var j = 0; j < neural_layer[i]["weights"].length; j++) {
					neural_layer[i]["weights"][j] = {
						weight: this._randomWeight(),
						last_change: 0
					}
				}
			}else{ // inputs neurons
				neural_layer[i] = {
					output: 0
				}
			}
			
			// Add name to neuron if provided
			if(typeof current_layer[i] === "string"){
				neural_layer[i]["name"] = current_layer[i];
			}
		}
		return neural_layer;
	},
	_launchAsyncTrainer: function(subsequent_launch){
		// Train the network for 50 ms, if unsuccessful, retry
		if(subsequent_launch && this._trainNeuronsFor(50)){

			this._neural_network_training_status = "TRAINED";
			
		}else{
			var that = this;
			window.setTimeout(function(){
				that._launchAsyncTrainer(true);
			},1);
		}
		if(subsequent_launch){
			this._pingCallbacks();
		}
	},
	_trainNeuronsFor: function(ms){
		var start_time = Date.now();
		
		var i_pattern = 0,mean_squared_error,mean_squared_errors;
		while(Date.now() < (start_time + ms)){
			
			mean_squared_errors = 0;
			
			// Randomize _learned_patterns, try help solving
			this._learned_patterns = this._shuffleArray(this._learned_patterns);
			for (i_pattern = this._learned_patterns.length - 1; i_pattern >= 0; i_pattern--){

				// Forward and backward propagation
				this._runInputs(this._learned_patterns[i_pattern]["inputs"]);
				this._calculateErrorsAndDeltasFor(this._learned_patterns[i_pattern]["outputs"]);
				mean_squared_error = this._meanSquaredError();
				mean_squared_errors += mean_squared_error;
				
				if(this._overfitting_protection && this._learned_patterns[i_pattern]["testing_only"]){
					this._test_mean_squared_errors.push(mean_squared_error);
				}else{
					this._adjustWeightsAndBiases();
				}
				

				this._neural_network_training_iterations++;
				
			}
			
			mean_squared_error = (mean_squared_errors/this._learned_patterns.length);
			if(mean_squared_error < this._targeted_mse ||
				Date.now() > (this._neural_network_training_start_time_ms+this._max_training_time_ms) ||
				this._isOverfitting()){
				return true;
			}
			
		}
		return false;
	},
	
	// Forward & Backward propagations
	_neuronFunction: function(neuron_sum,neuron_function_mode){
		// String === string if fairly fast in fact
		if(neuron_function_mode === "sigmoid"){
			return 1 / (1+Math.exp(-neuron_sum));
		}
	},
	_runInputs: function(inputs){
		// Set inputs layer outputs
		for (var i_input = this._neural_network[0].length - 1; i_input >= 0; i_input--) {
			this._neural_network[0][i_input]["output"] = 0;
			if(typeof inputs[this._neural_network[0][i_input]["name"]] === "number"){
				this._neural_network[0][i_input]["output"] = inputs[this._neural_network[0][i_input]["name"]];
			}
		}
		
		// Propagation
		var i_layer,i_neuron,neuron,neuron_sum,i_weight;
		for (i_layer = 1; i_layer < this._neural_network.length; i_layer++) {
			for (i_neuron = 0; i_neuron < this._neural_network[i_layer].length; i_neuron++) {
				
				neuron_sum = this._neural_network[i_layer][i_neuron]["bias"];
				for (i_weight = 0; i_weight < this._neural_network[i_layer][i_neuron]["weights"].length; i_weight++) {
					neuron_sum += (this._neural_network[i_layer][i_neuron]["weights"][i_weight]["weight"] * 
									this._neural_network[i_layer-1][i_weight]["output"]);
				}
				this._neural_network[i_layer][i_neuron]["output"] = this._neuronFunction(neuron_sum,this._neural_network[i_layer][i_neuron]["function_mode"]);
			}
		}
		
		// Return outputs
		var outputs = {},last_layer = this._neural_network[this._neural_network.length-1],i_output;
		for (i_output = last_layer.length - 1; i_output >= 0; i_output--) {
			outputs[last_layer[i_output]["name"]] = last_layer[i_output]["output"];
		}
		return outputs;
	},
	_calculateErrorsAndDeltasFor: function(targeted_outputs){

		var i_outputs_layer = this._neural_network.length - 1;
		var output,error,delta,i_layer,i_neuron,i_prev_neuron;
		
		for (i_layer = i_outputs_layer; i_layer > 0; i_layer--) {
			for(i_neuron = this._neural_network[i_layer].length - 1; i_neuron >= 0; i_neuron--){
				
				output = this._neural_network[i_layer][i_neuron]["output"];
				
				if(i_layer === i_outputs_layer){ // Outputs layer
					
					targeted_output = 0;
					if(typeof targeted_outputs[this._neural_network[i_layer][i_neuron]["name"]] === "number"){
						targeted_output = targeted_outputs[this._neural_network[i_layer][i_neuron]["name"]];
					}
					error = targeted_output - output;
					
				}else{ // Hidden layers

					error = 0;
					for (i_prev_neuron = this._neural_network[i_layer+1].length - 1; 
						i_prev_neuron >= 0; i_prev_neuron--) {
						error += this._neural_network[i_layer+1][i_prev_neuron]["delta"] * 
						this._neural_network[i_layer+1][i_prev_neuron]["weights"][i_neuron]["weight"];
					}
					
				}
				

				delta = error * output * (1 - output);
				this._neural_network[i_layer][i_neuron]["error"] = error;
				this._neural_network[i_layer][i_neuron]["delta"] = delta;
			}
		}
		
	},
	_adjustWeightsAndBiases: function(){
		var i_layer,i_neuron,i_w,delta,change;
		// Adjust outputs and hidden neurons
		for (i_layer = this._neural_network.length - 1; i_layer > 0; i_layer--){
			for (i_neuron = this._neural_network[i_layer].length - 1; i_neuron >= 0; i_neuron--) {
				delta = this._neural_network[i_layer][i_neuron]["delta"];
				for(i_w = this._neural_network[i_layer][i_neuron]["weights"].length - 1; i_w >= 0; i_w--){
					change = this._neural_network[i_layer][i_neuron]["weights"][i_w]["last_change"];
		            change = (this._learning_rate * delta * this._neural_network[i_layer - 1][i_w]["output"]) + (this._momentum * change);
					this._neural_network[i_layer][i_neuron]["weights"][i_w]["last_change"] = change;
					this._neural_network[i_layer][i_neuron]["weights"][i_w]["weight"] += change;
				}
				this._neural_network[i_layer][i_neuron]["bias"] += this._learning_rate * delta;
			}
		}
	},
	
	// Error calculation
	_meanSquaredError: function(){
		var i_outputs_layer = this._neural_network.length - 1, i_neuron, errors = [];
		for (var i_neuron = this._neural_network[i_outputs_layer].length - 1; i_neuron >= 0; i_neuron--) {
			errors.push(this._neural_network[i_outputs_layer][i_neuron]["error"]);
		}
		var sum = 0;
		for (var i = errors.length - 1; i >= 0; i--) {
			sum += errors[i]*errors[i];
		}
		return sum / errors.length;
	},
	_isOverfitting: function(){
		var l_p_length = this._learned_patterns.length;
		var errors_panels_size = l_p_length;
		
		if(this._test_mean_squared_errors.length < (errors_panels_size*2)){
			//console.log("too small: "+this._test_mean_squared_errors.length);
			return false;
		}
		
		while(this._test_mean_squared_errors.length > (errors_panels_size*2)){
			this._test_mean_squared_errors.shift();
		}
		var oldest_errors = 0,newest_errors = 0;
		for (var i = errors_panels_size - 1; i >= 0; i--) {
			oldest_errors += this._test_mean_squared_errors[i];
			newest_errors += this._test_mean_squared_errors[i+errors_panels_size];
		}
		if(newest_errors > oldest_errors){
			//console.log("overfitting after i" + this._neural_network_training_iterations);
			return true;
		}
		//console.log("not overfitting");
		return false;
	},
	
	// Training callbacks
	_addTrainedCallback: function(trained_callback){
		this._neural_network_trained_callbacks.push(trained_callback);
	},
	_pingCallbacks: function(){
		if(this._isFun(this._training_progress_callback)){
			if(this._neural_network_training_status !== "TO_TRAIN"){

				var training_infos = {
					status: this._neural_network_training_status,
					mean_squared_error: this._meanSquaredError(),
					training_iterations: this._neural_network_training_iterations
				};
				this._training_progress_callback.apply(window,[
					training_infos
				]);
				
			}
		}
		if(this._neural_network_training_status === "TRAINED"){
			while(this._neural_network_trained_callbacks.length > 0){
				this._neural_network_trained_callbacks[0].apply(this);
				this._neural_network_trained_callbacks.shift();
			}
		}
	},
	
	// Public functions
	learn: function(inputs,outputs){
		
		if(this._isNum(inputs)){
			inputs = [inputs];
		}
		if(this._isNum(outputs)){
			outputs = [outputs];
		}
		
		// Check validity
		if(!this._objectIsIOCompatible(inputs)){
			this._logError("Inputs object not valid, please use 0 to 1 numbers");
			return false;
		}
		if(!this._objectIsIOCompatible(outputs)){
			this._logError("Outputs object not valid, please use 0 to 1 numbers");
			return false;
		}
		
		// Overfitting detection
		var testing_only = false;
		if(this._learned_patterns.length > 10){
			testing_only = (this._learned_patterns.length%3 == 0);
		}
		
		this._resetNeuralNetwork();
		
		this._learned_patterns.push({
			"inputs": inputs,
			"outputs": outputs,
			"testing_only": testing_only
		});
		return true;
	},
	guess: function(inputs,callback,immediate_response){
		if(!this._objectIsIOCompatible(inputs)){
			this._logError("Inputs object not valid, please use 0 to 1 numbers");
			return false;
		}
		if(!this._isFun(callback)){
			this._logError("Callback function is not valid or not defined");
			return false;
		}

		this._setupAndTrainNeuralNetwork();
		if(immediate_response){
			callback.apply(window,[
				this._runInputs(inputs),
				inputs
			]);
		}else{
			this._addTrainedCallback(function(){
				callback.apply(window,[
					this._runInputs(inputs),
					inputs
				]);
			});
		}
		return true;
	},
	
}

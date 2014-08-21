Einstein.js
===========

A JavaScript browser-based artificial neural network made for client-side execution.

Einstein.js is 20% faster than [brain](https://github.com/harthur/brain) and works without any dependency.

Live demo
---------

Check the demo at http://project-einsteinjs-demo.diplodoc.us/

Usage
-----


```javascript
var einstein = new Einstein({training_progress_callback:yourCallback, targeted_mse:0.005});

// Learn XOR
einstein.learn([0,0],0);
einstein.learn([0,1],1);
einstein.learn([1,0],1);
einstein.learn([1,1],0);

einstein.guess([X,X], function(outputs){...});

// training_progress_callback is called until targeted_mse is reached
Status: TRAINING mean_squared_error: 0.04529366824648675
Status: TRAINED mean_squared_error: 0.004889117464346838

Einstein guessed 0.056 for inputs [0,0]
Einstein guessed 0.931 for inputs [0,1]
Einstein guessed 0.931 for inputs [1,0]
Einstein guessed 0.084 for inputs [1,1]
```

Einstein.js
===========

A JavaScript browser-based artificial neural network

```javascript
var einstein = new Einstein(training_progress_callback:yourCallback);

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

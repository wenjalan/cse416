# Pre-Lecture 2: Assessing Performance
Usually, the residual sum of squares (RSS) loss function does a pretty good job. But a side effect of using RSS is that can introduce overfitting, where the model's polynomial perfectly hits every training point, resulting in a 100% training accuracy. This might not reflect the real world. This is the generalization/memorization problem.

![Overfitting](./img/2-1.png)

## Future Performance
In the real world, a given input (like square footage) might result in many different outputs (many house prices). To fix this, we can define our own loss function to determine ourselves exactly how good or bad a guess is.

> `L(y, f(x))`

Where `L()` is our Loss Function, `y` is the true value, and `f(x)` is our prediction. The function will theoretically tell you how bad a prediction was, given the fact that there can be many different valid predictions for a given `x`. In other words, it makes getting a close prediction less painful. A **true loss** function will take into account a weight as to how likely a prediction is to occur in the real world, in theory. The problem is how such a function should be constructed.

## Model Assessment
One strategy is to hide a portion of our input data from the model when training, and then using that data to predict the "real world" performance of the model. This is referred to as the "train" and "test" data sets.

Evaluating the model using the test data, we can create a **test loss**, which is an approximation of the true loss.

One tradeoff is, however, that the more data you reserve for testing, the less data you have for training the model. So a decision needs to be made whether or not the sacrifice testing quality for model training quality. IN practice, it's generally 80/20 or 90/10.
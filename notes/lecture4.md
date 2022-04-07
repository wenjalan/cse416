# Prelecture 4.1 - Regularization and Selecting Lambda
## Regularization Recap
Overfitting can occur with too complex a model with too small dataset, and with too many features with not enough dataset to get resolution on a pattern. We can tell if a model is overfitting when the coefficients of the model grow large.

Therefore, we developed a new quality metric that takes into account the size of the coefficients.

> `w = min(RSS(W)) + l * ||w||`

`l`, or lambda, is how much the coefficient term `||w||` affects the quality metric's output. If lambda is 0, then it has no effect, but when it is infinity, then even small coefficients will incur heavy costs. Generally, when putting lambda between 0 and infinity, it reduces the size of coefficients.

## Selecting Lambda
Since we want to choose a lambda that allows us to find the best on future data, we're going to use the validation data to find a lambda.

![Choosing Lambda](./img/4-1.png)

First, train the model using a possible value of lambda. Then, calculate its validation error, *without the regularization term*, and keep track of which setting of lambda results in the minimum validation error.

# Prelecture 4.2 - Feature Selection
Since using many features tends to make models overfit, the process of selecting which features to use from a set of possible features is of interest. This has benefits for complexity, interpretability, and speed. Mathematically, we're trying to find a model such that only a few features have non-zero coefficients. There are several methods of doing this.

## All Subsets
Look at all possible subsets of features. That is, all models with 0 features, 1 feature, 2 features, etc. for every combination of features. Keep track of what subset of features worked the best for a given number of features.

![n features vs error](./img/4-2.png)

Generally, as more features are introduced, true error will decrease at first, then begin to increase again. Therefore, the problem of finding an optimal set of features is an ongoing research problem. Current methods:

1. Assess on validation set
2. Cross validation
3. Other methods that penalize complexity, like BIC
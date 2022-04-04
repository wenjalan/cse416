# Prelecture 3.1 - Cross Validation
Reminder that choosing a model with the lowest train error usually results in overfitting. Choosing a model on lowest test error invalidates the point of the test error. So we're exploring ways to prevent overfitting while not using train or test error. The first way was validation sets, and this one is **cross validation**.

## Validation Sets
Split the dataset into train, validation, and test sets. Train the model using the train set, validate using the validation set. This has the drawback of sacrificing training quality.

## Cross Validation
As a reminder, we use train and test sets to allow us to train then test our model. Setting aside the test data set, if we want to valdiate our model, we can train the model on different chunks of the training set, and use the other sections of the training set as validation sets, then repeat for each chunk.

![Chunking](./img/3-2.png)

Code for this is as follows:

![Cross Validation Code](./img/3-1.png)

This has the benefit of not sacrificing any training data, but now we're training `k` models, where `k = # of chunks`. For the best results, `k = n`, that is, one chunk per data point, but that's so slow nobody does that. So in practice people use `k = [5, 10]`.

# Prelecture 3.2 - Overfitting, Coefficients, and Regularization
## Coefficients
Coefficients in a linear regression are the terms a and b.

> `y = a + bx`

Interpreting what `a` and `b` are in the context of the problem is what coefficient interpretation is. When the number of coefficients grows greater, it can be useful to hold some coefficients constant to analyze the other coefficients. For instance, if we hold a coefficient constant in a `p = 3` regression, we get a slice:

![Constant Coefficient](./img/3-3.png)

With that slice, we can get a 2D graph:

![2D Graph](./img/3-4.png)

This can tell us how a system changes if one coefficient is fixed. This can also be done with as many features you want by holding all but two coefficients constant. Note that this won't work if multiple coefficients use the same feature (for instance if a is square foot and b is square foot squared).

## Coefficients and Overfitting
When dealing with an overfit regression, the coefficients of your equation will be quite large.

> If the coefficients of a regression are large, it can be a sign of overfitting. Therefore, limiting the magnitude of a coefficient can be a method to prevent overfitting.

![Coefficient Magnitude vs. Degree](./img/3-5.png)

> Coefficients are sometimes written as a list, such as `w = [w1, w2, w3 ... wj]`

Overfitting can also occur if a model doesn't have enough features. For instance, a 2D model needs a few hundred points to prevent overfitting which is hard, but as dimensions increase, exponentially more data is needed to fill the gaps between. This is the **Curse of Dimensionality**. Will be reviewed later in the course.

## Preventing Overfitting
When trying to prevent overfitting, we can take a more retroactive approach compared to validation. One such way is regularization, which is the topic of today's class.
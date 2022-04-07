# Reflection 2
## Summary
This week in CSE 416, we learned about Validation, Regularization, and Feature Selection. These items have to do with preventing the model from overfitting or underfitting by controlling the coefficients of the model in different ways.

## Concepts
### Validation
Validation is the process of verifying that a learned predictor does not overfit or underfit its training data. Two methods of performing this are using a Validation Set and Cross Validation.

### Regularization
Regularization is the process of limiting, in some way, the magnitudes of the coefficients that a model learns. This is achieved using Ridge Regression or LASSO Regression.

### Feature Selection
Feature selection is the practice of vetting which features of a model should be used to reduce the complexity, and therefore training time, of a given model.

## Uncertainties
* For datasets of simple sizes (say, < 12 features), how does feature selection come into play? Might we want a model with fewer features and worse performance, if it means it's cheaper to deploy?
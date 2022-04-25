# 1.
## a. 
Incorrect. Non-convergence in gradient descent is a symptom of too large a learning rate. Pranav should decrease the learning rate.

## b.
Incorrect. A baseline model should be able to outperform a majority class classifier, in this case, with an accuracy of 40%, since the majority class makes up 40% of the cases.

## c.
Incorrect. A false positive would mean a patient that does not have cancer is predicted to have cancer, false negatives mean a patient with cancer is predicted to not have it. He is correct in saying that we should consider FN and FP error when improving model performance.

## d.
Correct. So long as subsets are randomized, gradient descent will usually point in the right direction.

## e.
Incorrect. Rahul will have to train a new model for every combination of lambda, learning rate, and iteration. That is 4 * 4 * 2, or 32 models.

# 2.
I can't do math.

# 3.
Majority class classifier. It is less prone to overfit than logistic regression.
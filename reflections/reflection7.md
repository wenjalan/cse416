# Learning Reflection 7
## Summary
This week introduces Dimensionality Reduction and Recommender Systems.

## Concepts
### Principle Component Analysis
PCA or Principle Component Analysis is a very popular algorithm. It uses a linear projection from a d-dimensional space to a k-dimensional space that minimizes reconstruction error.

Basically, if you were to pick only k dimensions from a set of d dimensions, and tried to reverse the process, which set of k would lead to the least errors?

## Cooccurrence Matrix
The basic idea of a co-occurrence matrix is that people who like one thing also like other things. In other words, if a user buys diapers, they may also buy baby wipes.

Imagine we have an `m x m` matrix `C`, where each `[i, j]` is the coincidence of two products. Each row is an item, each column is an item.

## Coordinate Descent
Once we have the quality/fitness metric, we now need to perform some sort of optimization. Gradient descent is a little hard wih two vectors, so instead, **coodinate descent** is used instead. Essentially, the `L` and `R` matrices are optimized in alternating fashion. This looks like a step-wise optimization.

## Uncertainties
How might a recommender system be able to distinguish between the various moods of a user? For instance, someone might be in the mood for an action movie, or a romance movie, at different times.
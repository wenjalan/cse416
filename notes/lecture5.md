# Prelecture 5 - Classification Overview
In regression, the input to the model is numbers, and the output is a number. This works for a wide number of cases, but what if the thing we want as output isn't an number? For instance, what if we want to categorize something as good or bad, dog or cat, or blue or orange?

These tasks are known as classification tasks. Generally, they involve categorizing an input as one or more discrete values. For instance, an email filtering program that categorizes emails as spam or not spam.

| Task | Type of Classification |
| - | - |
| Spam or Not Spam Filter | Binary Classification |
| Object Detection | Multiclass Classification |

## Sentiment Classifier: Restaurant Reviews
What we're going to be doing is building a sentiment classifier to rate restaurant reviews.

![Classifier](./img/5-1.png)

Some input will be had (the review) and an output (from -1 to 1, from negative to positive) will be output. Here are a few naive ways to perform this.

### Simple Threshold Classifier
Count the number of positive and negative words. Return whichever positive or negative words were greater.

```py
count_positive = countPositive(review)
count_negative = countNegative(review)
return count_positive > count_negative ? 1 : -1
```

This runs into issues, because we'd need a way to classify words themselves as positive or not, and sometimes there are words that can seem positive at first but actually be negative (not good, isn't great).

To attempt resolving these issues, we can try to create a word classifier to determine the positivity of words, and get into using unigrams vs. bigrams. We won't go into NLP too much, but it is a thing.

### Linear Classifier
Instead of counting words, we can sum up words with weights attached depending on how positive or negative they are.

![Word Weights](./img/5-2.png)

We can learn those weights later. If we have this table, we can sum up the weights of all the words in the review, and output a score.

> `Score = Sum(Sentence.Words -> Weight(Word))`

Then we return:

```py
score = sum(sentence.forEach(word -> {
    return word.weight
}))
return signum(score)
```

For future reference:

> `s = score(x)`
> `y = sign(s)`

| Symbol | Description |
| - | - |
| `s` | Score |
| `y` | Output |

## Decision Boundaries
When deciding the positivity/negativity of words, we need to decide what rules a negative word, to place all negative words below that and all positive words above that. The line that divides the positive from the negative is the decision boundary.

![Decision Bound](./img/5-3.png)


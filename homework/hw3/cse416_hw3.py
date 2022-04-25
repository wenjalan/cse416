import string
import numpy as np
import pandas as pd

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer

# Add any additional imports here
# TODO


np.random.seed(416)

# Load data
products = pd.read_csv('food_products.csv')
products = products[products['rating'] != 3]
products['sentiment'] = products['rating'].apply(lambda rating : +1 if rating > 3 else -1)

# Data Processing: Remove punctuation
def remove_punctuation(text):
    if type(text) is str:
        return text.translate(str.maketrans('', '', string.punctuation))
    else:
        return ''
    
products['review_clean'] = products['review'].apply(remove_punctuation)

# Feature Extraction: Count words
vectorizer = CountVectorizer()
count_matrix = vectorizer.fit_transform(products['review_clean'])

product_data = pd.DataFrame(count_matrix.toarray(),
        index=products.index,
        columns=vectorizer.get_feature_names())

#  Create a new DataFrame that has all these features as columns plus the sentiment label!
product_data['sentiment'] = products['sentiment']
product_data['review_clean'] = products['review_clean']
product_data['summary'] = products['summary']

# Train/Validation/Test Split
train_data, test_data = train_test_split(product_data, test_size=0.2)
validation_data, test_data = train_test_split(test_data, test_size=0.5)

# Q1: Majority class classifier
### edTest(test_q1_majority_classifier) ###

# TODO "Train" a majority class classifier and calculate its validation accuracy
n = product_data.size
n_positive = product_data[product_data['sentiment'] == 1].size
majority_label = -1 if n > n_positive else 1

n_val = validation_data.size
n_val_correct = validation_data[validation_data['sentiment'] == majority_label].size
majority_classifier_validation_accuracy = n_val_correct / n_val
# print(majority_classifier_validation_accuracy)

# Train a sentiment model
features = vectorizer.get_feature_names()
sentiment_model = LogisticRegression(penalty='l2', C=1e23)
sentiment_model.fit(train_data[features], train_data['sentiment'])

# Q2: Compute most positive/negative
### edTest(test_q2_most_pos_neg_words) ###

# TODO Find the most positive word and most negative word in the sentiment_model
min_index = np.where(coefficients == coefficients.min())
max_index = np.where(coefficients == coefficients.max())

features = sentiment_model.feature_names_in_
most_negative_word = features[min_index][0]
most_positive_word = features[max_index][0]

# Q3: Most positive/negative review
### edTest(test_q3_most_positive_negative_review) ###

# TODO Find the review_clean values for the most positive and most negative review
val_pred = sentiment_model.predict_proba(validation_data[features])
val_negatives = np.array([x[0] for x in val_pred])
val_positives = np.array([x[1] for x in val_pred])
max_negative = np.max(val_negatives)
max_positive = np.max(val_positives)
val_max_negative_i = np.where(val_negatives == max_negative)[0][0]
val_max_positive_i = np.where(val_positives == max_positive)[0][0]

most_negative_review = validation_data.iloc[val_max_negative_i]['review_clean']
most_positive_review = validation_data.iloc[val_max_positive_i]['review_clean']

# Q4: Sentiment model validation accuracy 
### edTest(test_q4_sentiment_model_accuracy) ###
from sklearn.metrics import accuracy_score
# TODO Find the validation accuracy of the sentiment model
val_pred = sentiment_model.predict(validation_data[features])
sentiment_model_validation_accuracy = accuracy_score(validation_data['sentiment'], val_pred)
# print(sentiment_model_validation_accuracy)

# Q5: Confusion matrix
### edTest(test_q5_confusion_matrix) ###

# TODO Compute the four values tp, fp, fn, tn and plot them using plot_confusion_matrix
validation_data['predicted'] = val_pred
tp = validation_data[(validation_data['sentiment'] == 1) & (validation_data['predicted'] == 1)].size
fp = validation_data[(validation_data['sentiment'] == -1) & (validation_data['predicted'] == 1)].size
tn = validation_data[(validation_data['sentiment'] == -1) & (validation_data['predicted'] == -1)].size
fn = validation_data[(validation_data['sentiment'] == 1) & (validation_data['predicted'] == -1)].size

# Q6 and Q7

### edTest(test_q6_q7_train_models) ###

# TODO Fill in the loop below

# Set up the regularization penalities to try
l2_penalties = [0.01, 1, 4, 10, 1e2, 1e3, 1e5]
l2_penalty_names = [f'coefficients [L2={l2_penalty:.0e}]' 
                    for l2_penalty in l2_penalties]

# Q6: Add the coefficients to this coef_table for each model
coef_table = pd.DataFrame(columns=['word'] + l2_penalty_names)
coef_table['word'] = features

# Q7: Set up an empty list to store the accuracies (will convert to DataFrame after loop)
accuracy_data = []

for l2_penalty, l2_penalty_column_name in zip(l2_penalties, l2_penalty_names):
    # TODO(Q6 and Q7): Train the model 
    model = LogisticRegression(C=(1/l2_penalty))
    model.fit(train_data[features], train_data['sentiment'])
    
    # TODO(Q6): Save the coefficients in coef_table
    coef_table[l2_penalty_column_name] = model.coef_[0]
    
    # TODO(Q7): Calculate and save the train and validation accuracies
    train_pred = model.predict(train_data[features])
    train_acc = accuracy_score(train_data['sentiment'], train_pred)
    val_pred = model.predict(validation_data[features])
    val_acc = accuracy_score(validation_data['sentiment'], val_pred)
    accuracy_data.append({
        'l2_penalty': l2_penalty,
        'train_accuracy': train_acc,
        'validation_accuracy': val_acc
    })

accuracies_table = pd.DataFrame(accuracy_data)

# Q8 
### edTest(test_q8_most_positive_negative) ###


# TODO Compute words with the 5 largest coefficients and 5 smallest coefficients
l2p1_name = 'coefficients [L2=1e+00]'
positive_words = pd.Series(coef_table.nlargest(5, l2p1_name)['word'])
negative_words = pd.Series(coef_table.nsmallest(5, l2p1_name)['word'])

positive_words = None
negative_words = None
import numpy as np
import pandas as pd
import scipy.stats
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split

# Add any additional imports here
# TODO

np.random.seed(416)

loans = pd.read_csv('lending-club-data.csv')
loans['safe_loans'] = loans['bad_loans'].apply(lambda x: +1 if x == 0 else -1)
loans = loans.drop(columns='bad_loans')

# Q1: Write code to find most frequent grade
### edTest(test_q1_most_common_loan_grade) ###

# Q1: Write code to find most frequent grade
mode_grade = loans['grade'].mode()[0]
# print(mode_grade)

# Q2: Write code to find percent of loans for rent
### edTest(test_q2_percent_rent) ###

# Q2: Write code to find percent of loans for rent
percent_rent = len(loans[loans['home_ownership'] == 'RENT']) / len(loans)

# Preprocess data
features = [
    'grade',  # grade of the loan (e.g. A or B)
    'sub_grade',  # sub-grade of the loan (e.g. A1, A2, B1)
    'short_emp',  # one year or less of employment (0 or 1)
    'emp_length_num',  # number of years of employment (a number)
    'home_ownership',  # home_ownership status (one of own, mortgage, rent or other)
    'dti',  # debt to income ratio (a number)
    'purpose',  # the purpose of the loan (one of many values)
    'term',  # the term of the loan (36 months or 60 months)
    'last_delinq_none',  # has borrower had a delinquincy (0 or 1)
    'last_major_derog_none',  # has borrower had 90 day or worse rating (0 or 1)
    'revol_util',  # percent of available credit being used (number between 0 and 100)
    'total_rec_late_fee',  # total late fees received to day (a number)
]

target = 'safe_loans'  # prediction target (y) (+1 means safe, -1 is risky)

# Extract the feature columns and target column
loans = loans[features + [target]]

loans = pd.get_dummies(loans)
features = list(loans.columns)
features.remove('safe_loans')

train_data, validation_data = train_test_split(loans, test_size=0.2)

# Q3: Train a model with max_depth=6
### edTest(test_q3_decision_tree_model) ###

from sklearn.tree import DecisionTreeClassifier

# Q3: Train a model with max_depth=6
decision_tree_model = DecisionTreeClassifier(max_depth=6)
decision_tree_model.fit(train_data[features], train_data['safe_loans'])

# Q4: Find train and validation accuracy
### edTest(test_q4_validation_accuracy) ###

# Q4: Find train and validation accuracy
train_pred = decision_tree_model.predict(train_data[features])
val_pred = decision_tree_model.predict(validation_data[features])

decision_train_accuracy = len(train_data[train_data['safe_loans'] == train_pred]) / len(train_data)
decision_validation_accuracy = len(validation_data[validation_data['safe_loans'] == val_pred]) / len(validation_data)
# print(decision_train_accuracy)
# print(decision_validation_accuracy)

# Q5: Train a decision tree model with max_depth=10
### edTest(test_q5_big_tree) ###
from sklearn.metrics import accuracy_score

# Q5: Train a decision tree model with max_depth=10
big_tree_model = DecisionTreeClassifier(max_depth=10)
big_tree_model.fit(train_data[features], train_data['safe_loans'])

big_train_pred = big_tree_model.predict(train_data[features])
big_val_pred = big_tree_model.predict(validation_data[features])

big_train_accuracy = accuracy_score(big_train_pred, train_data['safe_loans'])
big_validation_accuracy = accuracy_score(big_val_pred, validation_data['safe_loans'])
# big_train_accuracy = train_data[train_data['safe_loans'] == big_train_pred].size / train_data.size
# big_validation_accuracy = validation_data[validation_data['safe_loans'] == big_val_pred].size / validation_data.size
# print(big_train_accuracy)
# print(big_validation_accuracy)


# Q6: Use GridSearchCV to find best settings of hyperparameters
### edTest(test_q6_grid_search) ###
from sklearn.model_selection import GridSearchCV

# Q6: Use GridSearchCV to find best settings of hyperparameters
hyperparameters = {
    'min_samples_leaf': [1, 10, 50, 100, 200, 300],
    'max_depth': [1, 5, 10, 15, 20]
}

search = GridSearchCV(
    estimator=DecisionTreeClassifier(), 
    param_grid=hyperparameters, 
    cv=6, 
    return_train_score=True
)
search.fit(train_data[features], train_data['safe_loans'])


# Q7
import scipy.stats 
from numpy.random import randint

class RandomForest416: 
    """
    This class implements the common sklearn model interface (has a fit and predict function).
    
    A random forest is a collection of decision trees that are trained on random subsets of the 
    dataset. When predicting the value for an example, takes a majority vote from the trees.
    """
    
    def __init__(self, num_trees, max_depth=1):
        """
        Constructs a RandomForest416 that uses the given numbner of trees, each with a 
        max depth of max_depth.
        """
        self._trees = [
            DecisionTreeClassifier(max_depth=max_depth) 
            for i in range(num_trees)
        ]
        
    def fit(self, X, y):
        """
        Takes an input dataset X and a series of targets y and trains the RandomForest416.
        
        Each tree will be trained on a random sample of the data that samples the examples
        uniformly at random (with replacement). Each random dataset will have the same number
        of examples as the original dataset, but some examples may be missing or appear more 
        than once due to the random sampling with replacement.
        """    
        # Q7
        for tree in self._trees:
            # generate a random sample from X
            subset = []
            for i in range(len(X)):
                subset.append(X.iloc[randint(0, len(X))])
            
            # convert back to DataFrame
            # subset = pd.DataFrame(subset)
            
            # train tree on subset
            tree.fit(subset, y)
           
    def predict(self, X):
        """
        Takes an input dataset X and returns the predictions for each example in X.
        """
        # Builds up a 2d array with n rows and T columns
        # where n is the number of points to classify and T is the number of trees
        predictions = np.zeros((len(X), len(self._trees)))
        for i, tree in enumerate(self._trees):
            # Make predictions using the current tree
            preds = tree.predict(X)
            
            # Store those predictions in ith column of the 2d array
            predictions[:, i] = preds
            
        # For each row of predictions, find the most frequent label (axis=1 means across columns)
        return scipy.stats.mode(predictions, axis=1)[0]


# Q7
### edTest(test_q7_random_forest) ###
# Q7
forest = RandomForest416(num_trees=2, max_depth=1)
forest.fit(train_data[features], train_data['safe_loans'])

forest_train_pred = forest.predict(train_data[features]).flatten()
forest_val_pred = forest.predict(validation_data[features]).flatten()

rf_train_accuracy = accuracy_score(forest_train_pred, train_data['safe_loans'])
rf_validation_accuracy = accuracy_score(forest_val_pred, validation_data['safe_loans'])
# rf_train_accuracy = train_data[train_data['safe_loans'] == forest_train_pred].size / train_data.size
# rf_validation_accuracy= validation_data[validation_data['safe_loans'] == forest_val_pred].size / validation_data.size
# print(rf_train_accuracy)
# print(rf_validation_accuracy)

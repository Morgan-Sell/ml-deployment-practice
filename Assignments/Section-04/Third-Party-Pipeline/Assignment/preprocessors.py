import numpy as np
import pandas as pd

from sklearn.base import BaseEstimator, TransformerMixin

import config

# Add binary variable to indicate missing values
class MissingIndicator(BaseEstimator, TransformerMixin):

    def __init__(self, variables=None):
        # Checks to see if a list is passed.
        # If string is passed, changes type to list.
        if not isinstance(variables, list):
            self.variables = [variables]
        else:
            self.variables = variables

    def fit(self, X, y=None):
        # fit the statement to accommodate sklearn pipeline functionality
        return self


    def transform(self, X):
        # add indicator
        X = X.copy()
        for feature in self.variables:
            X[feature+'_NA'] = np.where(X[feature].isnull(), 1, 0)
        return X

# categorical missing value imputer
class CategoricalImputer(BaseEstimator, TransformerMixin):

    def __init__(self, variables=None):
    # Checks to see if a list is passed.
    # If string is passed, changes type to list.
        if not isinstance(variables, list):
            self.variables = [variables]
        else:
            self.variables = variables

    def fit(self, X, y=None):
        # fit statement to accommodate the sklearn pipeline
        return self

    def transform(self, X):
        X = X.copy()
        for feature in self.variables:
            X[feature] = X[feature].fillna('Missing')
        return X

# Numerical missing value imputer
class NumericalImputer(BaseEstimator, TransformerMixin):

    def __init__(self, variables=None):
    # Checks to see if a list is passed.
    # If string is passed, changes type to list.
        if not isinstance(variables, list):
            self.variables = [variables]
        else:
            self.variables = variables

    def fit(self, X, y=None):
        # persist mode in a dictionary
        self.imputer_dict_ = {}
        for feature in self.variables:
            self.imputer_dict_[feature] = X[feature].mean()

    def transform(self, X):

        X = X.copy()
        for feature in self.variables:
            X[feature].fillna(self.imputer_dict_[feature], inplace=True)
        return X



# Extract first letter from string variable
class ExtractFirstLetter(BaseEstimator, TransformerMixin):

    def __init__(self, variables=None):
        if not isinstance(variables, list):
            self.variables = [variables]
        else:
            self.variables = variables

    def fit(self, X, y=None):
        # we need this step to fit the sklearn pipeline
        return self

    def transform(self, X):
        X = X.copy()
        for feature in self.variables:
            X[feature] = X[feature].str[0]
        return X        

# frequent label categorical encoder
class RareLabelCategoricalEncoder(BaseEstimator, TransformerMixin):

    def __init__(self, tol=0.05, variables=None):
        self.tolerance = tol

        if isinstance(variables, list): 
            self.variables = variables
        else:
            self.variables = variables

    def fit(self, X, y=None):

        # persist frequent labels in dictionary
        self.freq_labels_dict_ = {}
        for feature in self.variables:
            freq_var = pd.Series(X[feature].value_counts() / np.float(len(X)))
            self.freq_labels_dict_[feature] = list(freq_var[freq_var >= self.tolerance].index)

    def transform(self, X):
        # if feature value is not in the frequent labels...
        # ... then replace with 'Rare'
        X = X.copy()
        for feature in self.variables:
            X[feature] = np.where(X[feature].isin(self.freq_labels_dict_[feature]),
                                    X[feature], 'Rare')
        
        return X
            
                        


# add one-hot encoder variables and remove original categorical variables
class CategoricalEncoder(BaseEstimator, TransformerMixin):

    def __init__(self, variables=None):
        if not isinstance(variables, list):
            self.variables = [variables]
        else:
            self.variables = variables

    def fit(self, X, y=None):

        # HINT: persist the dummy variables found in train set
        self.dummies = pd.get_dummies(X[self.variables], drop_first=True).columns
        
        return self

    def transform(self, X):
        # encode labels
        X = X.copy()
        # get dummies
        for feature in self.variables:
            X = pd.concat([X,
                        pd.get_dummies(X[feature], prefix=feature, drop_first=True)],
                        axis=1)
        # drop original variables
        X.drop(self.variables, axis=1, inplace=True)
        # add missing dummies if any
        missing_vars = list(set(self.dummies - X.columns))

        if len(missing_vars) == 0:
            print('All dummy variables were added.')
        
        else:
            for var in  missing_vars:
                X[var] = 0
    
        return X

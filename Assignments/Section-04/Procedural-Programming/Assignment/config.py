# ====   PATHS ===================

PATH_TO_DATASET = "titanic.csv"
OUTPUT_SCALER_PATH = 'scaler.pkl'
OUTPUT_MODEL_PATH = 'logistic_regression.pkl'


# ======= PARAMETERS ===============

# imputation parameters
# features and the respective values used to fill NAs.
# Values are the median of the specified features in the training set.
IMPUTATION_DICT = {
    'age': 28.0,
    'far': 14.4542
}


# encoding parameters
FREQUENT_LABELS = {
    'sex': ['female', 'male'],
    'cabin': ['C', 'Missing'],
    'embarked': ['C', 'Q', 'S'],
    'title': ['Miss', 'Mr', 'Mrs']
}


DUMMY_VARIABLES = ['age_NA', 'fare_NA', 'sex_male', 'cabin_Missing',
    'cabin_Rare', 'embarked_Q', 'embarked_Rare', 'embarked_S',
    'title_Mr', 'title_Mrs', 'title_Rare']


# ======= FEATURE GROUPS =============

TARGET = 'survived'

CATEGORICAL_VARS = ['sex', 'cabin', 'embarked', 'title']

NUMERICAL_TO_IMPUTE = ['pclass', 'age', 'sibsp', 'parch', 'fare']

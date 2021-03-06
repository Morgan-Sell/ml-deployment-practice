    import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression

import joblib
import config


# Individual pre-processing and training functions
# ================================================

def load_data(df_path):
    # Function loads data for training
    return pd.read_csv(df_path)



def divide_train_test(df, target):
    # Function divides data set in train and test
    X = df.drop(target, axis=1)
    y = df[target]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=0)

    return X_train, X_test, y_train, y_test


def extract_cabin_letter(df):
    # captures the first letter
    return df['cabin'].str[0]



def add_missing_indicator(df, var):
    # function adds a binary missing value indicator
    df = df.copy()
    df[var+'_NA'] = np.where(df[var].isnull(), 1, 0)
    return df



def impute_na(df, var, replacement='Missing'):
    # function replaces NA by value entered by user
    # or by string Missing (default behaviour)
    return df[var].fillna(replacement)



def remove_rare_labels(df, var, freq_labels):
    # groups labels that are not in the frequent list into the umbrella
    # group Rare
    return np.where(df[var].isin(freq_labels), df[var], 'Rare')



def encode_categorical(df, vars_categorical):
    # adds ohe variables and removes original categorical variable

    df = df.copy()
    for var in vars_categorical:
        df = pd.concat([df,
                        pd.get_dummies(df[var], prefix=var, drop_first=True)],
                        axis=1)

    df.drop(labels=vars_categorical, axis=1, inplace=True)
    return df



def check_dummy_variables(df, dummy_list):
    # check that all missing variables where added when encoding, otherwise
    # add the ones that are missing
    df = df.copy()
    missing_vars = list(set(dummy_list) - set(df.columns))

    # If no features are missing, do not revise df.
    if len(missing_vars) == 0:
        print('All dummy variables were added.')

    for var in missing_vars:
        df[var] = 0

    return df



def train_scaler(df, output_path):
    # train and save scaler
    scaler = StandardScaler()
    scaler.fit(df)
    joblib.dump(scaler, output_path)
    return scaler



def scale_features(df, output_path):
    # load scaler and transform data
    scaler = joblib.load(output_path)
    return scaler.transform(df)



def train_model(df, target, output_path):
    # train and save model
    log_reg = LogisticRegression(C=0.0005, random_state=0)
    log_reg.fit(df, target)
    joblib.dump(log_reg, output_path)

    return None



def predict(df, model):
    # load model and get predictions
    model = joblib.load(model)
    return model.predict(df)

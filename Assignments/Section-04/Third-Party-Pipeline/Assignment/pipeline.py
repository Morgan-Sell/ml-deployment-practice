from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

import preprocessors2 as pp
import config


titanic_pipe = Pipeline(
    # complete with the list of steps from the preprocessors file
    # and the list of variables from the config
    [
        

        ('categorical_imputer',
            pp.CategoricalImputer(variables=config.CATEGORICAL_VARS)),
        
        ('identify_missing_values',
            pp.MissingIndicator(variables=config.ALL_FEATURES)),
        
        ('numerical_imputer',
            pp.NumericalImputer(variables=config.NUMERICAL_VARS)),
        
        ('extract_first_letter',
            pp.ExtractFirstLetter(variables=config.CABIN)),
        
        ('encode_rare_categorical_labels',
            pp.RareLabelCategoricalEncoder(
                tol=0.05,
                variables=config.CATEGORICAL_VARS)),
        
        ('one_hot_encoder_categorical',
            pp.CategoricalEncoder(variables=config.CATEGORICAL_VARS)),
        
        ('scaler', StandardScaler()),

        ('logistic_regression', LogisticRegression(C=0.0005, random_state=0))

    ]
)
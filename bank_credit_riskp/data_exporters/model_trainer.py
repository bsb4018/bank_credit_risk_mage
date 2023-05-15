import pandas as pd
from pandas import DataFrame
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split
import os
import pickle
import json

if 'data_exporter' not in globals():
    from mage_ai.data_preparation.decorators import data_exporter


LABEL_COLUMN = 'credit_risk'


def build_training_and_test_set(df: DataFrame):
    X = df.drop(columns=[LABEL_COLUMN])
    y = df[LABEL_COLUMN]

    return train_test_split(X, y, random_state=68)


def train_model(X, y):
    model = GradientBoostingClassifier(ccp_alpha=0.0, criterion='friedman_mse', init=None,
                           learning_rate=0.1, loss='log_loss', max_depth=3,
                           max_features=None, max_leaf_nodes=None,
                           min_impurity_decrease=0.0, min_samples_leaf=1,
                           min_samples_split=2, min_weight_fraction_leaf=0.0,
                           n_estimators=100, n_iter_no_change=None,
                           random_state=8255, subsample=1.0, tol=0.0001,
                           validation_fraction=0.1, verbose=0,
                           warm_start=False)
    model.fit(X, y)

    return model


def score_model(model, X, y):
    y_pred = model.predict(X)
    f1 = f1_score(y, y_pred, average='weighted')
    accuracy = accuracy_score(y, y_pred)

    return f1, accuracy


@data_exporter
def export_data(df, *args, **kwargs):
    """
    Exports data to some source.

    Args:
        data: The output from the upstream parent block
        args: The output from any additional upstream blocks (if applicable)

    Output (optional):
        Optionally return any object and it'll be logged and
        displayed when inspecting the block run.
    """
    # Specify your data exporting logic here

    # Create a folder "artifacts" in the same location as "bank_credit_riskp" and 
    # inside it create sub folders "model" and "data_split" 

    X_train, X_test, y_train, y_test = build_training_and_test_set(df)
    model = train_model(X_train, y_train)

    f1,accuracy = score_model(model, X_test, y_test)
    print(f'F1 Score: {f1}')
    print(f'Accuracy: {accuracy}')

    cwd = os.getcwd()
    print(f'Saving model')
    model_filename = f'{cwd}/artifacts/model/finalized_model.lib'
    pickle.dump(model, open(model_filename, 'wb'))

    print(f'Saving model report')
    model_report = {"f1_score" : f1 , "accuracy" : accuracy}
    model_report_filename = f'{cwd}/artifacts/model/model_report.json'
    jsonString = json.dumps(model_report)
    jsonFile = open(model_report_filename, "w")
    jsonFile.write(jsonString)
    jsonFile.close()

    print(f'Saving training and test set')
    X_train.to_csv(f'{cwd}/artifacts/data_split/X_train.csv', index = False)
    X_test.to_csv(f'{cwd}/artifacts/data_split/X_test.csv', index = False)
    y_train.to_csv(f'{cwd}/artifacts/data_split/y_train.csv', index = False)
    y_test.to_csv(f'{cwd}/artifacts/data_split/y_test.csv', index = False)


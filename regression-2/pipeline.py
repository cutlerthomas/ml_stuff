#!/usr/bin/env python3

import sys
import argparse
import logging
import os.path

import pandas as pd
import numpy as np
import sklearn.linear_model
import sklearn.preprocessing
import sklearn.pipeline
import sklearn.base
import sklearn.metrics
import joblib


class PipelineNoop(sklearn.base.BaseEstimator, sklearn.base.TransformerMixin):
    """
    Just a placeholder with no actions on the data.
    """
    
    def __init__(self):
        return

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        return X

def get_test_filename(test_file, filename):
    if test_file == "":
        basename = get_basename(filename)
        test_file = "{}-test.csv".format(basename)
    return test_file

def get_basename(filename):
    root, ext = os.path.splitext(filename)
    dirname, basename = os.path.split(root)
    logging.info("root: {}  ext: {}  dirname: {}  basename: {}".format(root, ext, dirname, basename))

    stub = "-train"
    if basename[len(basename)-len(stub):] == stub:
        basename = basename[:len(basename)-len(stub)]

    return basename

def get_model_filename(model_file, filename):
    if model_file == "":
        basename = get_basename(filename)
        model_file = "{}-model.joblib".format(basename)
    return model_file

def get_data(filename):
    """
    Assumes column 0 is the instance index stored in the
    csv file.  If no such column exists, remove the
    index_col=0 parameter.
    """
    data = pd.read_csv(filename, index_col=0)
    return data

def load_data(my_args, filename):
    data = get_data(filename)
    feature_columns, label_column = get_feature_and_label_names(my_args, data)
    X = data[feature_columns]
    y = data[label_column]
    return X, y

def get_feature_and_label_names(my_args, data):
    label_column = my_args.label
    feature_columns = my_args.features

    if label_column in data.columns:
        label = label_column
    else:
        label = ""

    features = []
    if feature_columns is not None:
        for feature_column in feature_columns:
            if feature_column in data.columns:
                features.append(feature_column)

    # no features specified, so add all non-labels
    if len(features) == 0:
        for feature_column in data.columns:
            if feature_column != label:
                features.append(feature_column)

    return features, label

def make_numerical_feature_pipeline(my_args):
    items = []
    
    if my_args.use_polynomial_features:
        items.append(("polynomial-features", sklearn.preprocessing.PolynomialFeatures(degree=my_args.use_polynomial_features)))
    if my_args.use_scaler:
        items.append(("scaler", sklearn.preprocessing.StandardScaler()))
    items.append(("noop", PipelineNoop()))
    
    numerical_pipeline = sklearn.pipeline.Pipeline(items)
    return numerical_pipeline

def make_SGD_fit_pipeline(my_args):
    items = []
    items.append(("features", make_numerical_feature_pipeline(my_args)))
    items.append(("model", sklearn.linear_model.SGDRegressor()))
    return sklearn.pipeline.Pipeline(items)

def do_fit(my_args):
    train_file = my_args.train_file
    if not os.path.exists(train_file):
        raise Exception("training data file: {} does not exist.".format(train_file))

    X, y = load_data(my_args, train_file)
    
    pipeline = make_SGD_fit_pipeline(my_args)
    pipeline.fit(X, y)

    model_file = get_model_filename(my_args.model_file, train_file)

    joblib.dump(pipeline, model_file)
    return

def get_feature_names(pipeline, X):
    primary_feature_names = list(X.columns[:])
    if 'polynomial-features' in pipeline['features'].named_steps:
        secondary_powers = pipeline['features']['polynomial-features'].powers_
        feature_names = []
        for powers in secondary_powers:
            s = ""
            for i in range(len(powers)):
                for j in range(powers[i]):
                    if len(s) > 0:
                        s += "*"
                    s += primary_feature_names[i]
            feature_names.append(s)
            logging.info("powers: {}  s: {}".format(powers, s))
    else:
        logging.info("polynomial-features not in features: {}".format(pipeline['features'].named_steps))
        feature_names = primary_feature_names
    return feature_names

def get_scale_offset(pipeline, count):
    if 'scaler' in pipeline['features'].named_steps:
        scaler = pipeline['features']['scaler']
        logging.info("scaler: {}".format(scaler))
        logging.info("scale: {}  mean: {}  var: {}".format(scaler.scale_, scaler.mean_, scaler.var_))
        theta_scale = 1.0 / scaler.scale_
        intercept_offset = scaler.mean_ / scaler.scale_
    else:
        theta_scale = np.ones(count)
        intercept_offset = np.zeros(count)
        logging.info("scaler not in features: {}".format(pipeline['features'].named_steps))
    return theta_scale, intercept_offset

def show_function(my_args):
    train_file = my_args.train_file
    if not os.path.exists(train_file):
        raise Exception("training data file: {} does not exist.".format(train_file))
    model_file = get_model_filename(my_args.model_file, train_file)
    if not os.path.exists(model_file):
        raise Exception("Model file, '{}', does not exist.".format(model_file))
    
    X, y = load_data(my_args, train_file)
    pipeline = joblib.load(model_file)

    feature_names = get_feature_names(pipeline, X)
    scale, offset = get_scale_offset(pipeline, len(feature_names))

    features = pipeline['features']
    X = features.transform(X)
    regressor = pipeline['model']

    intercept_offset = 0.0
    for i in range(len(regressor.coef_)):
        intercept_offset += regressor.coef_[i] * offset[i]

    s = "{}".format(regressor.intercept_[0]-intercept_offset)
    for i in range(1, len(regressor.coef_)):
        if len(feature_names[i]) > 0:
            t = "({}*{})".format(regressor.coef_[i]*scale[i], feature_names[i])
        else:
            t = "({})".format(regressor.coef_[i])
        if len(s) > 0:
            s += " + "
        s += t

    basename = get_basename(train_file)
    print("{}: {}".format(basename, s))
    return


def show_score(my_args):

    train_file = my_args.train_file
    if not os.path.exists(train_file):
        raise Exception("training data file: {} does not exist.".format(train_file))
    
    test_file = get_test_filename(my_args.test_file, train_file)
    if not os.path.exists(test_file):
        raise Exception("testing data file, '{}', does not exist.".format(test_file))
    
    model_file = get_model_filename(my_args.model_file, train_file)
    if not os.path.exists(model_file):
        raise Exception("Model file, '{}', does not exist.".format(model_file))

    X_train, y_train = load_data(my_args, train_file)
    X_test, y_test = load_data(my_args, test_file)
    pipeline = joblib.load(model_file)
    regressor = pipeline['model']
    
    basename = get_basename(train_file)
    score_train = regressor.score(pipeline['features'].transform(X_train), y_train)
    if my_args.show_test:
        score_test = regressor.score(pipeline['features'].transform(X_test), y_test)
        print("{}: train_score: {} test_score: {}".format(basename, score_train, score_test))
    else:
        print("{}: train_score: {}".format(basename, score_train))
    return

def show_loss(my_args):

    train_file = my_args.train_file
    if not os.path.exists(train_file):
        raise Exception("training data file: {} does not exist.".format(train_file))
    
    test_file = get_test_filename(my_args.test_file, train_file)
    if not os.path.exists(test_file):
        raise Exception("testing data file, '{}', does not exist.".format(test_file))
    
    model_file = get_model_filename(my_args.model_file, train_file)
    if not os.path.exists(model_file):
        raise Exception("Model file, '{}', does not exist.".format(model_file))

    X_train, y_train = load_data(my_args, train_file)
    X_test, y_test = load_data(my_args, test_file)
    pipeline = joblib.load(model_file)

    y_train_predicted = pipeline.predict(X_train)
    y_test_predicted = pipeline.predict(X_test)

    basename = get_basename(train_file)
    
    loss_train = sklearn.metrics.mean_squared_error(y_train, y_train_predicted)
    if my_args.show_test:
        loss_test = sklearn.metrics.mean_squared_error(y_test, y_test_predicted)
        print("{}: L2(MSE) train_loss: {} test_loss: {}".format(basename, loss_train, loss_test))
    else:
        print("{}: L2(MSE) train_loss: {}".format(basename, loss_train))

    loss_train = sklearn.metrics.mean_absolute_error(y_train, y_train_predicted)
    if my_args.show_test:
        loss_test = sklearn.metrics.mean_absolute_error(y_test, y_test_predicted)
        print("{}: L1(MAE) train_loss: {} test_loss: {}".format(basename, loss_train, loss_test))
    else:
        print("{}: L1(MAE) train_loss: {}".format(basename, loss_train))

    loss_train = sklearn.metrics.r2_score(y_train, y_train_predicted)
    if my_args.show_test:
        loss_test = sklearn.metrics.r2_score(y_test, y_test_predicted)
        print("{}: R2 train_loss: {} test_loss: {}".format(basename, loss_train, loss_test))
    else:
        print("{}: R2 train_loss: {}".format(basename, loss_train))
    return

def show_model(my_args):

    train_file = my_args.train_file
    if not os.path.exists(train_file):
        raise Exception("training data file: {} does not exist.".format(train_file))
    
    test_file = get_test_filename(my_args.test_file, train_file)
    if not os.path.exists(test_file):
        raise Exception("testing data file, '{}', does not exist.".format(test_file))
    
    model_file = get_model_filename(my_args.model_file, train_file)
    if not os.path.exists(model_file):
        raise Exception("Model file, '{}', does not exist.".format(model_file))

    pipeline = joblib.load(model_file)
    regressor = pipeline['model']
    features = pipeline['features']

    print("Model Information:")
    print("coef_: {}".format(regressor.coef_))
    print("intercept_: {}".format(regressor.intercept_))
    print("n_iter_: {}".format(regressor.n_iter_))
    print("n_features_in_: {}".format(regressor.n_features_in_))


    try:
        scaler = features["scaler"]
        print("scaler.mean_: {}".format(scaler.mean_))
        print("scaler.var_: {}".format(scaler.var_))
    except:
        print("No scaler.")
    return



def parse_args(argv):
    parser = argparse.ArgumentParser(prog=argv[0], description='Fit Data With Linear Regression Using Pipeline')
    parser.add_argument('action', default='SGD',
                        choices=[ "SGD", "show-function", "score", "loss", "show-model" ], 
                        nargs='?', help="desired action")
    parser.add_argument('--train-file',    '-t', default="",    type=str,   help="name of file with training data")
    parser.add_argument('--test-file',     '-T', default="",    type=str,   help="name of file with test data (default is constructed from train file name)")
    parser.add_argument('--model-file',    '-m', default="",    type=str,   help="name of file for the model (default is constructed from train file name when fitting)")
    parser.add_argument('--random-seed',   '-R', default=314159265,type=int,help="random number seed (-1 to use OS entropy)")
    parser.add_argument('--features',      '-f', default=None, action="extend", nargs="+", type=str,
                        help="column names for features")
    parser.add_argument('--label',         '-l', default="label",   type=str,   help="column name for label")
    parser.add_argument('--use-polynomial-features', '-p', default=0,         type=int,   help="degree of polynomial features.  0 = don't use (default=0)")
    parser.add_argument('--use-scaler',    '-s', default=0,         type=int,   help="0 = don't use scaler, 1 = do use scaler (default=0)")
    parser.add_argument('--show-test',     '-S', default=0,         type=int,   help="0 = don't show test loss, 1 = do show test loss (default=0)")

    my_args = parser.parse_args(argv[1:])

    #
    # Do any special fixes/checks here
    #
    
    return my_args

def main(argv):
    my_args = parse_args(argv)
    # logging.basicConfig(level=logging.INFO)
    logging.basicConfig(level=logging.WARN)

    if my_args.action == 'SGD':
        do_fit(my_args)
    elif my_args.action == "show-function":
        show_function(my_args)
    elif my_args.action == "score":
        show_score(my_args)
    elif my_args.action == "loss":
        show_loss(my_args)
    elif my_args.action == "show-model":
        show_model(my_args)
    else:
        raise Exception("Action: {} is not known.".format(my_args.action))
        
    return

if __name__ == "__main__":
    main(sys.argv)
    

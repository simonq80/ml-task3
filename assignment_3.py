import ConfigParser
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn import datasets, linear_model, svm
from sklearn.metrics import mean_squared_error, r2_score, explained_variance_score, accuracy_score, fbeta_score, f1_score
from sklearn.preprocessing import Imputer, LabelEncoder
from sklearn.naive_bayes import GaussianNB

K = 10  # Number of folds
ROWS = 20050 # Number of rows to read

def get_input_data():
    f = pd.read_csv("gender-classifier-DFE-791531.csv", header=0, nrows=ROWS)
    data = f.iloc[:,13:15]
    target = f["gender"]
    target_val = f["gender"]

    """
    Convert the Target string values to int values
    e.g.:
    'Very Small Number' -> 0,
    'Small Number' -> 1,
    'Medium Number' -> 2,
    'Large Number' -> 3,
    'Very Large Number' -> 4
    """

    le = LabelEncoder()
    target_l = le.fit_transform(target)
    data_l = data.apply(LabelEncoder().fit_transform)
    #print data
    #print data_l
    return data_l, target_val.as_matrix(), target_l

def perform_logistic_regression(train_X, train_Y, test_X, test_Y):
    print train_X
    regr = linear_model.LogisticRegression(C=1e5)
    regr.fit(train_X, train_Y)
    pred_Y = regr.predict(test_X)
    return fbeta_score(test_Y, pred_Y, 0.1, average='macro'), accuracy_score(test_Y, pred_Y)

def perform_naive_bayes(train_X, train_Y, test_X, test_Y):
    # Split data into 2 to avoid memory error
    partial_size = ROWS / 2
    train_X0 = train_X[partial_size:]
    train_X1 = train_X[:partial_size]
    train_Y0 = train_Y[partial_size:]
    train_Y1 = train_Y[:partial_size]

    gnb = GaussianNB()
    gnb.partial_fit(train_X0, train_Y0, classes=np.arange(0, 5))
    gnb.partial_fit(train_X1, train_Y1)
    pred_Y = gnb.predict(test_X)
    return fbeta_score(test_Y, pred_Y, 0.1, average='macro'), accuracy_score(test_Y, pred_Y)

def perform_linear_regression(train_X, train_Y, test_X, test_Y):
    regr = linear_model.LinearRegression()
    regr.fit(train_X, train_Y)
    pred_Y = regr.predict(test_X)
    return mean_squared_error(test_Y, pred_Y), r2_score(test_Y, pred_Y)

def seventy_thirty_test(data, target, algorithm):
    data_train_size = (ROWS / 10) * 7

    train_X = data[:data_train_size]
    test_X = data[data_train_size:]

    train_Y = target[:data_train_size]
    test_Y = target[data_train_size:]

    print train_X

    return algorithm(train_X, train_Y, test_X, test_Y)

def k_separate_data(input_data, input_target):
    data_split = []
    target_split = []
    for i in range (0, K):
        data_split.append(input_data[(i * ROWS / K) : ((i+1) * ROWS / K)])
        target_split.append(input_target[(i * ROWS / K) : ((i+1) * ROWS / K)])
    return data_split, target_split

def k_fold_test(input_data, input_target, algorithm):
    data_split, target_split = k_separate_data(input_data, input_target)

    total_m1 = 0.0
    total_m2 = 0.0

    for i in range(0, K):
        test_X = data_split[i]
        test_Y = target_split[i]

        train_X = []
        train_Y = []

        for j in range(0, K):
            if (j != i):
                train_X.extend(data_split[i])
                train_Y.extend(target_split[i])

        m1, m2 = algorithm(train_X, train_Y, test_X, test_Y)
        total_m1 += m1
        total_m2 += m2

    return total_m1/K, total_m2/K

def main():
    data, target, target_labels = get_input_data()

    #result0, result1 = seventy_thirty_test(data, target_labels, perform_naive_bayes)
    #result2, result3 = k_fold_test(data, target_labels, perform_naive_bayes)

    result4, result5 = seventy_thirty_test(data, target_labels, perform_logistic_regression)
    #result6, result7 = k_fold_test(data, target_labels, perform_logistic_regression)

    #result8, result9 = seventy_thirty_test(data, target, perform_linear_regression)
    #result10, result11 = k_fold_test(data, target, perform_linear_regression)

    """
    print("\nNAIVE BAYES ALGORITHM")
    print("### 70/30 TEST ###")
    print("FBeta Score = %f" % result0)
    print("Accuracy = %f" % result1)
    print("### 10 FOLD TEST ###")
    print("FBeta Score = %f" % result2)
    print("Accuracy = %f\n" % result3)
    """
    print("LOGISTIC REGRESSION ALGORITHM")
    print("### 70/30 TEST ###")
    print("FBeta Score = %f" % result4)
    print("Accuracy = %f" % result5)
    #print("### 10 FOLD TEST ###")
    #print("FBeta Score = %f" % result6)
    #print("Accuracy = %f\n" % result7)
    """
    print("LINEAR REGRESSION ALGORITHM")
    print("### 70/30 TEST ###")
    print("Mean Squared Error = %f" % result8)
    print("R2 Score = %f" % result9)
    print("### 10 FOLD TEST ###")
    print("Mean Squared Error = %f" % result10)
    print("R2 Score = %f\n" % result11)
    """
main()

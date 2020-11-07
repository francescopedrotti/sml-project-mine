import pandas as pd
from sklearn.linear_model import RidgeClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
import numpy as np


def regularized_classifier(classifier, reg_lambda):
    if classifier == 'Ridge':
        return RidgeClassifier(alpha=reg_lambda)
    elif classifier == 'LogReg':
        return LogisticRegression(C=1/reg_lambda, max_iter=1000, tol=100)
    elif classifier == 'SVM':
        return SVC(C=1/reg_lambda, tol=1)


classifier_name = 'SVM'  # Choose between 'Ridge', 'LogReg', 'SVM'

data = pd.read_csv('train-data.csv', sep=',', header=None)
labels = pd.read_csv('train-labels.csv', sep=',', header=None)

number_of_data = labels.shape[0]  # There are 57500 samples.
shuffling_key = np.random.permutation(range(number_of_data))  # Shuffling is always good
data_shuffled = data.iloc[shuffling_key]
labels_shuffled = labels.iloc[shuffling_key]

train_data = data_shuffled.iloc[:(4 * number_of_data // 5), 1:]
train_labels = labels_shuffled.iloc[:(4 * number_of_data // 5), [1]]
inner_test_data = data_shuffled.iloc[(4 * number_of_data // 5):, 1:]
inner_test_labels = labels_shuffled.iloc[(4 * number_of_data // 5):, [1]]
number_of_train_data = train_labels.shape[0]

eval_err = [[], []]  # Let's do a cross validation on lambda as last time...
for exp in range(-15, 16):
    eval_err_cv = 0
    for i in range(5):
        eval_data = train_data.iloc[i * (number_of_train_data // 5):(i + 1) * (number_of_train_data // 5), :]
        eval_labels = train_labels.iloc[i * (number_of_train_data // 5):(i + 1) * (number_of_train_data // 5)]
        eval_train_data = train_data[~train_data.index.isin(eval_data.index)]
        eval_train_labels = train_labels[~train_labels.index.isin(eval_labels.index)]

        eval_labels = eval_labels.values.ravel()
        eval_train_labels = eval_train_labels.values.ravel()

        classifier = regularized_classifier(classifier_name, 2 ** exp)
        classifier = classifier.fit(eval_train_data, eval_train_labels)

        eval_err_cv += (1 - classifier.score(eval_data, eval_labels))
        if i == 4:
            eval_err_cv = eval_err_cv / 5
            eval_err[0].append(exp)
            eval_err[1].append(eval_err_cv)

        print('Training at {}%'.format((5*(exp+15)+i)*100//154))

min_eval_err = min(eval_err[1])
best_exp = eval_err[0][eval_err[1].index(min_eval_err)]

final_classifier = regularized_classifier(classifier_name, 2 ** best_exp)
final_classifier = final_classifier.fit(train_data, train_labels.values.ravel())


inner_test_err = (1 - final_classifier.score(inner_test_data, inner_test_labels.values.ravel()))

print('Inner test error with {}: {:.3f}'.format(classifier_name, inner_test_err))

'''
Inner test error with Ridge: 0.145
SVM takes too long... Haven't tried
Inner test error with LogReg: 0.091
'''

import numpy as np
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report, cohen_kappa_score
from operator import truediv
from imblearn.metrics import geometric_mean_score
from imblearn.metrics import classification_report_imbalanced


def AA_andEachClassAccuracy(confusion_matrix):
    counter = confusion_matrix.shape[0]
    list_diag = np.diag(confusion_matrix)
    list_raw_sum = np.sum(confusion_matrix, axis=1)
    each_acc = np.nan_to_num(truediv(list_diag, list_raw_sum))
    average_acc = np.mean(each_acc)
    return each_acc, average_acc


def reports(y_pred, y_test):
    classification = classification_report(y_test, y_pred)
    oa = accuracy_score(y_test, y_pred)
    confusion = confusion_matrix(y_test, y_pred)
    each_acc, aa = AA_andEachClassAccuracy(confusion)
    kappa = cohen_kappa_score(y_test, y_pred)
    return classification, confusion, np.array([oa, aa, kappa] + list(each_acc)) * 100


def reports(y_pred, y_test):
    classification = classification_report(y_test, y_pred)
                                          
    gmean  = geometric_mean_score(y_test, y_pred)
    allstd = classification_report_imbalanced(y_test, y_pred, digits=6, output_dict=True)
    
    oa = accuracy_score(y_test, y_pred)
    confusion = confusion_matrix(y_test, y_pred)
    each_acc, aa = AA_andEachClassAccuracy(confusion)
    kappa = cohen_kappa_score(y_test, y_pred)
    #return classification, confusion, np.array([oa, aa, kappa] + list(each_acc)) * 100
    return classification, confusion, np.array([gmean, oa, aa, kappa] + list(each_acc)) * 100, allstd

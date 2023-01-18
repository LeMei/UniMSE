import numpy as np
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score, f1_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import plot_confusion_matrix
from sklearn.metrics import classification_report
from sklearn.metrics import f1_score
from itertools import chain
from utils.tools import *
import operator
import functools

def multiclass_acc(preds, truths):
    """
    Compute the multiclass accuracy w.r.t. groundtruth

    :param preds: Float array representing the predictions, dimension (N,)
    :param truths: Float/int array representing the groundtruth classes, dimension (N,)
    :return: Classification accuracy
    """
    return np.sum(np.round(preds) == np.round(truths)) / float(len(truths))


def weighted_accuracy(test_preds_emo, test_truth_emo):
    true_label = (test_truth_emo > 0)
    predicted_label = (test_preds_emo > 0)
    tp = float(np.sum((true_label==1) & (predicted_label==1)))
    tn = float(np.sum((true_label==0) & (predicted_label==0)))
    p = float(np.sum(true_label==1))
    n = float(np.sum(true_label==0))

    return (tp * (n/p) +tn) / (2*n)

def eval_emotionlines(results, truths):
    #主要通过混淆矩阵来计算
    truths = list(chain(*truths))
    # cm = confusion_matrix(results, truths)
    report = classification_report(truths, results)
    print(report)
def eval_laptops_restants(results, truths):
    truths = list(chain(*truths))
    true_count = 0
    for pred, truth in zip(results, truths):
        if pred == truth:
            true_count += 1
    print('true_count:{}, total_count:{}, precision:{}'.format(true_count, len(truths), true_count/len(truths)))
    # f1 = f1_score(t_targets_all.cpu(), torch.argmax(t_outputs_all, -1).cpu(), labels=[0, 1, 2], average='macro')


def eval_mosei_senti(results, truths, ids_list, exclude_zero=False):
    # test_truth = truths.view(-1).cpu().detach().numpy()
    test_truth = np.array(functools.reduce(operator.concat, truths))
    test_truth = np.array([float(e) for e in test_truth])

    non_zeros = np.array([i for i, e in enumerate(test_truth) if e != 0])
    preds = []
    for ele in results:
        if is_number(ele):
            preds.append(float(ele))
        else:
            preds.append(0.0)
            
    print('test_truth:{}'.format(test_truth))
    print('preds:{}'.format(preds))

    test_preds = np.array(preds)

    test_preds_a7 = np.clip(test_preds, a_min=-3., a_max=3.)
    test_truth_a7 = np.clip(test_truth, a_min=-3., a_max=3.)
    test_preds_a5 = np.clip(test_preds, a_min=-2., a_max=2.)
    test_truth_a5 = np.clip(test_truth, a_min=-2., a_max=2.)

    mae = np.mean(np.absolute(test_preds - test_truth))   # Average L1 distance between preds and truths
    corr = np.corrcoef(test_preds, test_truth)[0][1]
    mult_a7 = multiclass_acc(test_preds_a7, test_truth_a7)
    mult_a5 = multiclass_acc(test_preds_a5, test_truth_a5)

    # f_score = f1_score((test_preds[non_zeros] > 0), (test_truth[non_zeros] > 0), average='weighted')
    tmp = test_truth[non_zeros] > 0
    binary_truth_non0 = np.array([int(ele) for ele in tmp])
    tmp = test_preds[non_zeros] > 0
    binary_preds_non0 =  np.array([int(ele) for ele in tmp])
    f_score_non0 = f1_score(binary_truth_non0, binary_preds_non0, average='weighted')

    acc_2_non0 = accuracy_score(test_truth, test_preds, binary_truth_non0, binary_preds_non0, ids_list)
    binary_truth_has0 = test_truth >= 0
    binary_preds_has0 = test_preds >= 0
    acc_2 = accuracy_score(test_truth, test_preds, binary_truth_has0, binary_preds_has0, ids_list)
    f_score = f1_score(binary_truth_has0, binary_preds_has0, average='weighted')

    print("MAE: ", mae)
    print("Correlation Coefficient: ", corr)
    print("mult_acc_7: ", mult_a7)
    print("mult_acc_5: ", mult_a5)
    print("F1 score all/non0: {}/{} over {}/{}".format(np.round(f_score,4), np.round(f_score_non0,4), binary_truth_has0.shape[0], binary_truth_non0.shape[0]))
    print("Accuracy all/non0: {}/{}".format(np.round(acc_2,4), np.round(acc_2_non0,4)))

    print("-" * 50)
    return {'mae':mae, 'corr':corr, 'mult':mult_a7, 'f1':f_score, 'acc2':acc_2}

def eval_mosi(results, truths, ids_list, exclude_zero=False):
    return eval_mosei_senti(results, truths, ids_list, exclude_zero)

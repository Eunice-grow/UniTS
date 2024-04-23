import numpy as np
from sklearn.metrics import precision_recall_curve, roc_curve, auc, roc_auc_score, precision_score, recall_score, \
    accuracy_score, fbeta_score, average_precision_score


# function: calculate the point-adjust f-scores(whether top k)
def get_point_adjust_scores(y_test, pred_labels, true_events, thereshold_k=0, whether_top_k=False):
    tp = 0 # 真正例
    fn = 0 # 假反例
    for true_event in true_events.keys():
        true_start, true_end = true_events[true_event]
        '''如果 whether_top_k 为 False，
        仅检查预测标签序列在真实事件范围内的元素是否有任何一个大于0，如果有，则累加该区间内元素个数到 tp，否则累加到 fn
        异常时间段只要有1个预测为异常，那么这段异常时间都认为预测出来了异常，一个时间段内的所有点都判定预测正确了
        '''
        if whether_top_k is False:
            if pred_labels[true_start:true_end].sum() > 0:
                tp += (true_end - true_start)
            else:
                fn += (true_end - true_start)
        else:
            if pred_labels[true_start:true_end].sum() > thereshold_k:
                tp += (true_end - true_start)
            else:
                fn += (true_end - true_start)
    fp = np.sum(pred_labels) - np.sum(pred_labels * y_test)

    prec, rec, fscore = get_prec_rec_fscore(tp, fp, fn)
    return fp, fn, tp, prec, rec, fscore

def get_adjust_F1PA(pred, gt):
    anomaly_state = False
    for i in range(len(gt)):
        # 如果真实值和预测值都异常 且不是异常状态
        if gt[i] == 1 and pred[i] == 1 and not anomaly_state:
            anomaly_state = True
            for j in range(i, 0, -1): # 从i位置向左遍历直到0
                if gt[j] == 0: 
                    break
                else:
                    if pred[j] == 0: # 向左遍历时如果发现真实值为异常，预测值正常那么把预测值调整为异常
                        pred[j] = 1
            for j in range(i, len(gt)): # 向右遍历真实值，同样地调整预测值
                if gt[j] == 0:
                    break
                else:
                    if pred[j] == 0:
                        pred[j] = 1
        elif gt[i] == 0: 
            anomaly_state = False
        # 循环结束如果仍为异常状态 设置预测值异常
        if anomaly_state:
            pred[i] = 1
            
    from sklearn.metrics import precision_recall_fscore_support
    from sklearn.metrics import accuracy_score

    accuracy = accuracy_score(gt, pred)
    precision, recall, f_score, support = precision_recall_fscore_support(gt, pred,
                                                                          average='binary')
    return accuracy, precision, recall, f_score


# calculate the point-adjusted f-score
def get_prec_rec_fscore(tp, fp, fn):
    if tp == 0:
        precision = 0
        recall = 0
    else:
        precision = tp / (tp + fp)
        recall = tp / (tp + fn)
    fscore = get_f_score(precision, recall)
    return precision, recall, fscore


def get_f_score(prec, rec):
    if prec == 0 and rec == 0:
        f_score = 0
    else:
        f_score = 2 * (prec * rec) / (prec + rec)
    return f_score


# function: calculate the normal edition f-scores
def get_accuracy_precision_recall_fscore(y_true: list, y_pred: list):
    accuracy = accuracy_score(y_true, y_pred)
    # warn_for=() avoids log warnings for any result being zero
    # precision, recall, f_score, _ = prf(y_true, y_pred, average='binary', warn_for=())
    precision = precision_score(y_true, y_pred)
    recall = recall_score(y_true, y_pred)
    f_score = (2 * precision * recall) / (precision + recall)
    
    '''
    对于F0.5得分，函数使用 fbeta_score 并设定 beta=0.5 来计算，
    beta<1重视精度,beta>1重视召回率
    相比于F1得分,F_0.5得分更加重视精确率。
    当精确率和召回率都为0时.直接将F0.5得分设为0，以避免除以零的错误。
    '''
    
    if precision == 0 and recall == 0:
        f05_score = 0
    else:
        f05_score = fbeta_score(y_true, y_pred, average='binary', beta=0.5)
    return accuracy, precision, recall, f_score, f05_score



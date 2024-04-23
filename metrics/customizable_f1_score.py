# used by paper: Exathlon: A Benchmark for Explainable Anomaly Detection over Time Series_VLDB 2021
# github: https://github.com/exathlonbenchmark/exathlon
import numpy as np
from metrics.evaluate_utils import range_convers_new

'''基于偏置类型的权重函数。根据给定的偏置类型 (bias) 和位置索引 (i) 计算一个权重值。
权重可以是常数1（对于“flat”无偏置情况）、当前位置到序列结尾的距离（对于“front-end bias”，前向偏置）
或当前位置自身（对于“back-end bias”，后向偏置）。
'''
# the existence reward on the bias
def b(bias, i, length):
    if bias == 'flat':
        return 1
    elif bias == 'front-end bias':
        return length - i + 1
    elif bias == 'back-end bias':
        return i
    else:
        if i <= length / 2:
            return i
        else:
            return length - i + 1

'''给定一个异常范围 (AnomalyRange) 和一组位置列表 (p)，计算在整个异常范围内每个位置的加权和，
并根据其中存在的预测位置进一步调整总体加权值。
'''
def w(AnomalyRange, p):
    MyValue = 0
    MaxValue = 0
    start = AnomalyRange[0]
    AnomalyLength = AnomalyRange[1] - AnomalyRange[0] + 1
    # flat/'front-end bias'/'back-end bias'
    bias = 'flat'
    for i in range(start, start + AnomalyLength):
        bi = b(bias, i, AnomalyLength)
        MaxValue += bi
        if i in p:
            MyValue += bi
    return MyValue / MaxValue

'''计算两个区间集合间的交集数量（即重叠度量），并返回归一化后的交集计数值（即交集计数值的倒数）'''
def Cardinality_factor(Anomolyrange, Prange):
    score = 0
    start = Anomolyrange[0]
    end = Anomolyrange[1]
    for i in Prange:
        if start <= i[0] <= end:
            score += 1
        elif i[0] <= start <= i[1]:
            score += 1
        elif i[0] <= end <= i[1]:
            score += 1
        elif start >= i[0] and end <= i[1]:
            score += 1
    if score == 0:
        return 0
    else:
        return 1 / score

'''检查预测区间 preds 中是否存在至少一个点落在真实区间 labels 内，若存在则累加得分'''
def existence_reward(labels, preds):
    '''
    labels: list of ordered pair
    preds predicted data
    '''

    score = 0
    for i in labels:
        if np.sum(np.multiply(preds <= i[1], preds >= i[0])) > 0:
            score += 1
    return score

'''算一种综合了区间存在奖励和重叠奖励的召回率。
首先找到所有真实和预测的异常区间，然后分别计算存在奖励和重叠奖励。
最后根据给定的 alpha 参数组合这两种奖励，得到最终的召回率分数，并返回平均分值'''
def range_recall_new(labels, preds, alpha):
    p = np.where(preds == 1)[0]  # positions of predicted label==1
    range_pred = range_convers_new(preds)
    range_label = range_convers_new(labels)

    Nr = len(range_label)  # total # of real anomaly segments

    ExistenceReward = existence_reward(range_label, p)

    OverlapReward = 0
    for i in range_label:
        OverlapReward += w(i, p) * Cardinality_factor(i, range_pred)

    score = alpha * ExistenceReward + (1 - alpha) * OverlapReward
    if Nr != 0:
        return score / Nr, ExistenceReward / Nr, OverlapReward / Nr
    else:
        return 0, 0, 0

'''
定制化的F1分数（customizable_f1_score），该分数结合了区间召回率（range recall）、
存在奖励（existence reward）和重叠奖励（overlap reward），来评估预测标签与真实标签之间的匹配程度
'''
def customizable_f1_score(y_test, pred_labels,  alpha=0.2):
    label = y_test
    preds = pred_labels
    Rrecall, ExistenceReward, OverlapReward = range_recall_new(label, preds, alpha)
    Rprecision = range_recall_new(preds, label, 0)[0]

    if Rprecision + Rrecall == 0:
        Rf = 0
    else:
        Rf = 2 * Rrecall * Rprecision / (Rprecision + Rrecall)
    return Rf


def main():
    y_test = np.zeros(100)
    y_test[10:20] = 1
    y_test[50:60] = 1
    pred_labels = np.zeros(100)
    pred_labels[15:19] = 1
    pred_labels[55:62] = 1
    # pred_labels[51:55] = 1
    # true_events = get_events(y_test)
    Rf = customizable_f1_score(y_test, pred_labels)
    print("Rf: {}".format(Rf))


if __name__ == "__main__":
    main()
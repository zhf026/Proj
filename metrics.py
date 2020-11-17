"""
    机器学习评价指标
        - MAE
        - MAPE
        - RMSE
"""

import numpy as np


def re_standardize(in_x, mean_, std_):
    """
    反标准化
    :param in_x:    输入数据, numpy.ndarray, (n,) or (n,1)
    :param mean_:   in_x 均值
    :param std_:    in_x 标准差
    :return val:    numpy.ndarray, (n,1)
    """

    value = in_x.copy()
    value *= std_
    value += mean_
    return value.reshape(value.shape[0], 1)


def re_normalize(in_x, min_value, max_value):
    value = in_x.copy()
    value *= (max_value - min_value)
    value += min_value
    return value.reshape(value.shape[0], 1)


def cal_rmse(act_value, pred_value):
    m = act_value.shape[0]  # 样本数
    rmse = np.sqrt(((act_value - pred_value) ** 2).sum() / m)
    return rmse


def cal_mape(act_value, pred_value):
    m = act_value.shape[0]  # 样本数
    mape = (np.abs(act_value - pred_value) / act_value).sum() / m * 100
    return float(mape)


def cal_mae(act_value, pred_value):
    m = act_value.shape[0]  # 样本数
    mae = np.abs(act_value - pred_value).sum() / m
    return mae


def cal_r2(act_value, pred_value):
    r2 = 1 - (((act_value - pred_value) ** 2).sum()) / (((act_value - act_value.mean()) ** 2).sum())
    return r2


def evaluate_(metrics='all', act_val=None, pred_val=None,
              pre_deal=None, pre_a=None, pre_b=None, point=3):
    """

    :param act_val:    标准化的实际值, numpy.ndarray, (n,1)
    :param pred_val:    标准化的预测值, numpy.ndarray, (n,)
    :param pre_a:       均值
    :param pre_b:        标准差
    :param point:      保留小数点位数
    """

    actual, predicted = None, None
    if pre_a is not None and pre_b is not None:
        if pre_deal == 'std':
            # 反标准化
            actual = re_standardize(act_val, pre_a, pre_b)
            predicted = re_standardize(pred_val, pre_a, pre_b)
        elif pre_deal == 'min_max':
            # 反Min-Max归一化
            actual = re_normalize(act_val, pre_a, pre_b)
            predicted = re_normalize(pred_val, pre_a, pre_b)
    else:
        actual = act_val
        predicted = pred_val

    if metrics == 'all':
        rmse = cal_rmse(actual, predicted)
        mape = cal_mape(actual, predicted)
        mae = cal_mae(actual, predicted)
        r2 = cal_r2(actual, predicted)
        print('-> Metrics: RMSE=%f, MAPE=%f, MAE=%f, R2=%f' % (rmse, mape, mae, r2))
        return round(rmse, point), round(mape, point), round(mae, point), round(r2, point)
    elif metrics == 'rmse':
        rmse = cal_rmse(actual, predicted)
        print('-> Metrics: RMSE=%f' % rmse)
        return round(rmse, point)
    elif metrics == 'mape':
        mape = cal_mape(actual, predicted)
        print('-> Metrics: MAPE=%f' % mape)
        return round(mape, point)
    elif metrics == 'mae':
        mae = cal_mae(actual, predicted)
        print('-> Metrics: MAE=%f' % mae)
        return round(mae, point)
    elif metrics == 'r2':
        r2 = cal_r2(actual, predicted)
        print('-> Metrics: R2=%f' % r2)
        return round(r2, point)
    else:
        print('Please input correct metrics, e.g. rmse, mape, mae, r2 or all ...')

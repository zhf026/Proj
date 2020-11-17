'''
    数据预处理 -- preprocess.py
'''

import numpy as np
import pandas as pd


def load(path, features=None, index_col=None, st_index=None, end_index=None):
    # 从CSV文件加载数据
    df = pd.read_csv(path, infer_datetime_format=True,
                     parse_dates=[index_col],
                     index_col=[index_col])
    array_data = None
    if st_index is None and end_index is None:
        array_data = df.values
    if st_index is not None and end_index is not None:
        array_data = df.values[st_index:end_index, :]
    # 将所有输入特征名称构成字典   e.g. all_feature = {'year':0, 'month':1, 'day':2, 'hour':3. ...}
    col = list(df.columns)
    all_features = dict()
    for i in range(len(col)):
        all_features[col[i]] = i
    #
    if features == 'all':
        features = list(df.columns)
    indices = [all_features[i] for i in features if i in all_features.keys()]
    indices.sort()
    #
    data = np.zeros((array_data.shape[0], len(features)), dtype='float32')
    for i in range(len(features)):
        data[:, i] = array_data[:, indices[i]]

    return data


def split_dataset(data=None, tr_rate=None, tr_num=None):
    train_num = None
    if tr_rate is not None:
        train_num = int(len(data) * tr_rate)
    elif tr_num is not None:
        train_num = tr_num
    train_data = data[0:train_num, :]  # 训练数据
    test_data = data[train_num:, :]    # 测试数据

    return train_data, test_data


def standardize(data, tr_rate, tr_num):
    temp_data = data.copy()
    # 划分训练和测试数据
    train_data, test_data = split_dataset(data=temp_data, tr_rate=tr_rate, tr_num=tr_num)
    # 标准化训练数据
    mean_ = train_data.mean(axis=0)
    train_data -= mean_
    std_ = train_data.std(axis=0)
    train_data /= std_
    # 标准化测试数据
    test_data -= mean_
    test_data /= std_

    return train_data, test_data, (mean_, std_)


def normalize(data, tr_rate, tr_num):
    """
    Min-Max Normalization
    """
    temp_data = data.copy()
    # 划分训练和测试数据
    train_data, test_data = split_dataset(data=temp_data, tr_rate=tr_rate, tr_num=tr_num)
    #
    min_value = train_data.min(axis=0)
    max_value = train_data.max(axis=0)
    train_data = (train_data - min_value) / (max_value - min_value)
    test_data = (test_data - min_value) / (max_value - min_value)

    return train_data, test_data, (min_value, max_value)


def form_3D_samples(data, lookback, delay, span=1,
                    mode=None, target_index=None):
    '''
    将训练/测试数据组成样本集合，该集合为 3-D, 用于循环神经网络输入
    :param data:            输入数据, numpy.ndarray, (m, n)
    :param lookback:        过去时间步
    :param delay:           未来时间步
    :param span:            相邻样本间时间步跨度
    :param mode:            train/test
    :param target_index:    y 索引值
    :return X, y:           X, y
    '''

    X, y = list(), list()
    # 输入序列开端
    input_st = 0
    for i in range(len(data)):
        # 输入序列末端
        input_end = input_st + lookback  # 24 = 0 + 24
        # 输出序列末端
        output_end = input_end + delay  # 25 = 24 + 1
        if output_end <= len(data):
            X.append(data[input_st:input_end, :])
            y.append(data[input_end:output_end, target_index])
            if mode == 'train':
                input_st += span
            elif mode == 'test':
                input_st += span
            else:
                print('error in form_3D_sample-mode')
    X, y = np.array(X), np.array(y)
    return X, y


def hold_out_vali(tr_x, tr_y, v_rate=None, v_num=None):
    tr_num = None
    if v_rate is not None:
        tr_num = int(len(tr_x) * (1 - v_rate))
    elif v_num is not None:
        tr_num = len(tr_x) - v_num
    # Train set
    partial_tr_x = tr_x[0:tr_num, :, :]
    partial_tr_y = tr_y[0:tr_num, :]
    # Validation set
    v_x = tr_x[tr_num:, :, :]
    v_y = tr_y[tr_num:, :]

    return (partial_tr_x, partial_tr_y), (v_x, v_y)


def processing(file_path=None, index_col=0, st_index=None, end_index=None,
               features=None, lookback=None, delay=None, target_index=0,
               tr_rate=None, tr_num=None, pre_deal=None,
               hold_out=False, v_rate=None, v_num=None,
               model_type=None,
               ):

    # Step1: 加载数据
    data = load(file_path, features, index_col, st_index, end_index)

    # Step2: 标准化 / 归一化
    train_data, test_data, (pre_a, pre_b) = None, None, (None, None)
    if pre_deal == 'std':
        train_data, test_data, (pre_a, pre_b) = standardize(data, tr_rate=tr_rate, tr_num=tr_num)
    elif pre_deal == 'min_max':
        train_data, test_data, (pre_a, pre_b) = normalize(data, tr_rate=tr_rate, tr_num=tr_num)
    # 由于逐点预测，补充训练集后lookback点数据至测试集
    train_num = None
    if tr_rate is not None:
        train_num = int(len(data) * tr_rate)
    if tr_num is not None:
        train_num = tr_num
    test_data = np.concatenate([train_data[train_num - lookback:, :], test_data], axis=0)

    # Step3: 使用训练/测试数据，构建训练/测试集
    tr_x, tr_y = form_3D_samples(train_data, lookback, delay, span=1,
                                 mode='train', target_index=target_index)
    te_x, te_y = form_3D_samples(test_data, lookback, delay, span=1,
                                 mode='test', target_index=target_index)

    # Step4:
    if hold_out is True:
        (p_tr_x, p_tr_y), (v_x, v_y) = hold_out_vali(tr_x=tr_x, tr_y=tr_y, v_rate=v_rate, v_num=v_num)
        if model_type in ['GRU', 'LSTM', 'Stacking', 'Blending', 'Ensemble', 'Keras']:
            return (p_tr_x, p_tr_y), (v_x, v_y), (te_x, te_y), (pre_a, pre_b)
        elif model_type in ['LightGBM', 'lightgbm', 'lgb']:
            # 3-D to 2-D
            p_tr_x = p_tr_x.reshape(p_tr_x.shape[0], p_tr_x.shape[1] * p_tr_x.shape[2])
            v_x = v_x.reshape(v_x.shape[0], v_x.shape[1] * v_x.shape[2])
            te_x = te_x.reshape(te_x.shape[0], te_x.shape[1] * te_x.shape[2])
            # 2-D to 1-D
            p_tr_y = p_tr_y.reshape(p_tr_y.shape[0])
            v_y = v_y.reshape(v_y.shape[0])
            te_y = te_y.reshape(te_y.shape[0])
            return (p_tr_x, p_tr_y), (v_x, v_y), (te_x, te_y), (pre_a, pre_b)
        elif model_type in ['sklearn', 'XGBoost', 'xgboost', 'xgb']:
            # 3-D to 2-D
            p_tr_x = p_tr_x.reshape(p_tr_x.shape[0], p_tr_x.shape[1] * p_tr_x.shape[2])
            v_x = v_x.reshape(v_x.shape[0], v_x.shape[1] * v_x.shape[2])
            te_x = te_x.reshape(te_x.shape[0], te_x.shape[1] * te_x.shape[2])
            return (p_tr_x, p_tr_y), (v_x, v_y), (te_x, te_y), (pre_a, pre_b)
    elif hold_out is False:
        if model_type in ['GRU', 'LSTM', 'Stacking', 'Blending', 'Ensemble', 'Keras']:
            return (tr_x, tr_y), (te_x, te_y), (pre_a, pre_b)
        elif model_type in ['LightGBM', 'lightgbm', 'lgb']:
            # 3-D to 2-D
            tr_x = tr_x.reshape(tr_x.shape[0], tr_x.shape[1] * tr_x.shape[2])
            te_x = te_x.reshape(te_x.shape[0], te_x.shape[1] * te_x.shape[2])
            # 2-D to 1-D
            tr_y = tr_y.reshape(tr_y.shape[0])
            te_y = te_y.reshape(te_y.shape[0])
            return (tr_x, tr_y), (te_x, te_y), (pre_a, pre_b)
        elif model_type in ['sklearn', 'XGBoost', 'xgboost', 'xgb']:
            # 3-D to 2-D
            tr_x = tr_x.reshape(tr_x.shape[0], tr_x.shape[1] * tr_x.shape[2])
            te_x = te_x.reshape(te_x.shape[0], te_x.shape[1] * te_x.shape[2])
            return (tr_x, tr_y), (te_x, te_y), (pre_a, pre_b)
        else:
            print('请输入正确的模型...')
    else:
        print('error in preprocess.processing')

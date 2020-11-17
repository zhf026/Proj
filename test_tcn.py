
from pytcn.tcn import TCN
from pytcn.preprocess import *
from pytcn.metrics import *


def main():
    #
    lookback, delay = 24, 1
    target_index = 0
    features = ['f_load', 'month', 'day', 'hour', 'weekday', 'holiday', 'LUZ0']
    (tr_x, tr_y), (te_x, te_y), (pre_a, pre_b) = processing(file_path='data/data_natural.csv',
                                                            index_col=0,
                                                            features=features,
                                                            lookback=lookback,
                                                            delay=delay,
                                                            target_index=target_index,
                                                            model_type='LSTM',
                                                            tr_rate=0.7,
                                                            pre_deal='std')

    # --------------------------- TCN ---------------------------
    n_timesteps, n_features = lookback, tr_x.shape[2]
    model = TCN(input_shape=(n_timesteps, n_features),
                n_blocks=3, filters=[16, 32, 64],
                kernel_size=[24, 24, 24],
                dropout_rate=[0.1, 0.1, 0.1],
                n_outputs=delay)
    # Fit
    model.fit(tr_x, tr_y, epochs=10, batch_size=256, verbose=1)
    # Predict
    tcn_pred = model.predict(te_x)
    # Metrics
    tcn_rmse, tcn_mape, tcn_mae, tcn_r2 = evaluate_(metrics='all', act_val=te_y, pred_val=tcn_pred,
                                                    pre_deal='std', pre_a=pre_a[target_index],
                                                    pre_b=pre_b[target_index])
    # 反标准化
    tcn_pred_val = re_standardize(tcn_pred, mean_=pre_a[target_index], std_=pre_b[target_index])
    print('TCN: RMSE=%f  MAPE=%F  MAE=%F'%(tcn_rmse, tcn_mape, tcn_mae))


if __name__ == '__main__':
    main()

import pandas as pd
from data import Data
from ML import ML_Engine


def get_data(data):
    # data.to_csv_and_sql_data()
    # df = data.fetch_data_from_csv("BINANCE_ETHUSDT_1d.csv")
    df = data.fetch_data_from_table("BINANCE_BTCUSDT_4h")
    df.set_index('datetime', inplace=True)
    df.index = pd.to_datetime(df.index)
    print(df.head())
    print(df.tail())
    return df


def main():
    data = Data()
    df = get_data(data)
    # shift time series data before dividing data
    df_shifted = data.shift(df.copy(), 1)
    # divide data for feeding models
    df_scaled = data.minmax_scale(df_shifted.copy())
    input_test, input_train, output_test, output_train = data.get_data_partitions(df_scaled.copy(), shifted=True)
    # scale data
    # input_test_scaled, input_train_scaled, output_test_scaled, output_train_scaled = data.scale_data_partitions(
    #   input_test, input_train, output_test, output_train)

    # train models and  collect result metrics

    # fit regression models
    ml_engine = ML_Engine()
    models = [model for model in ml_engine.train_models(input_train.to_numpy(), output_train.to_numpy())]

    # write scaled scores
    scores = [score for model in models for score in
              ML_Engine.write_scores(model, model.predict(input_test.to_numpy()).reshape(-1, 1), output_test.to_numpy())]
    df_scores_scaled = pd.DataFrame(scores, columns=["Model", "MAE", "MSE", "RMSE", "R^2"])
    print (df_scores_scaled)

    scores=[]
    for model in models:
        y_hat = model.predict(input_test.to_numpy()).reshape(-1, 1)
        y_hat_not_scaled = data.minmax_scale_inverse(pd.DataFrame(y_hat), df_shifted["shifted"].min(),
                                                     df_shifted["shifted"].max()).to_numpy()
        y_true_not_scaled = data.minmax_scale_inverse(output_test, df_shifted["shifted"].min(),
                                                      df_shifted["shifted"].max()).to_numpy()
        for score in  ML_Engine.write_scores(model, y_hat_not_scaled, y_true_not_scaled):
            scores.append(score)

    df_scores = pd.DataFrame(scores, columns=["Model", "MAE", "MSE", "RMSE", "R^2"])
    print(df_scores)




    # plot results
    for model in models:
        y_hat = model.predict(input_test.to_numpy()).reshape(-1, 1)
        y_hat_not_scaled= data.minmax_scale_inverse(pd.DataFrame(y_hat),df_shifted["shifted"].min(),
                                                    df_shifted["shifted"].max()).to_numpy()
        y_true_not_scaled = data.minmax_scale_inverse(output_test, df_shifted["shifted"].min(),
                                                    df_shifted["shifted"].max()).to_numpy()

        ML_Engine.plot_scores(y_hat_not_scaled,y_true_not_scaled)


if __name__ == '__main__':
    main()

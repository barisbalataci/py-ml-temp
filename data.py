from datetime import datetime

import ccxt
import numpy as np
import pandas as pd
from sqlalchemy import create_engine
from sklearn.preprocessing import MinMaxScaler

from config import Config


class Data:


    def __init__(self) -> None:
        self.cfg = Config()
        scaler = MinMaxScaler(feature_range=(0, 1))

    def get_timestamp_ms(self, start_date):
        startDate = datetime.strptime(start_date, "%Y-%m-%d %H:%M:%S")
        startDate = datetime.timestamp(startDate)
        startDate = int(startDate) * 1000
        return startDate

    def get_data_ccxt_func(self, exchange_id, symbol, time_frame):
        exchange = getattr(ccxt, exchange_id)()
        # exchange=ccxt.binance()
        start_date = self.cfg.start_date
        while True:
            try:
                data = np.array(
                    exchange.fetch_ohlcv(symbol=symbol, timeframe=time_frame,
                                         since=self.get_timestamp_ms(start_date), limit=1000))
                if 'h' in time_frame:
                    hour = next(int(x) for x in time_frame if x.isdigit())
                    start_date = datetime.strftime(datetime.fromtimestamp(data[-1, 0] / 1000)
                                                   + pd.Timedelta(hours=hour), "%Y-%m-%d %H:%M:%S")
                elif 'd' in time_frame:
                    day = next(int(x) for x in time_frame if x.isdigit())
                    start_date = datetime.strftime(datetime.fromtimestamp(data[-1, 0] / 1000)
                                                   + pd.Timedelta(days=day), "%Y-%m-%d %H:%M:%S")
                elif 'm' in time_frame:
                    minute = next(int(x) for x in time_frame if x.isdigit())
                    start_date = datetime.strftime(datetime.fromtimestamp(data[-1, 0] / 1000)
                                                   + pd.Timedelta(minutes=minute), "%Y-%m-%d %H:%M:%S")
                yield (data)
            except Exception as ex:
                print(ex)
                break

    def fetch_data(self, exchange, symbol, time_frame):
        df = pd.DataFrame([y for x in self.get_data_ccxt_func(exchange, symbol, time_frame)
                           for y in x], columns=self.cfg.columns)
        df.set_index('datetime', inplace=True)
        df.index = pd.to_datetime(df.index, unit='ms')
        return df

    def to_csv_and_sql_data(self):
        sql_engine = create_engine('sqlite:///data/tosbik.db')
        for time_frame in self.cfg.time_frames:
            for exchange, symbols in self.cfg.exchanges.items():
                for symbol in symbols:
                    df = self.fetch_data(exchange, symbol, time_frame)
                    symbol_name = symbol.replace(' ', '').replace(
                        '/', '').replace('-', '').upper()
                    df.to_csv(f"./data/{str.upper(exchange)}_{symbol_name}_{time_frame}.csv")
                    df.to_sql(f"{str.upper(exchange)}_{symbol_name}_{time_frame}",
                              sql_engine, if_exists='replace')

    def fetch_data_from_sql(self, sql):
        db = create_engine('sqlite:///data/tosbik.db')
        """data = pd.read_sql( 'SELECT B.*,C.close AS EURUSD FROM BINANCE_ETHUSDT_1d AS B left join 
        CURRENCYCOM_EURUSD_1d AS C on C.datetime=B.datetime', engine) """
        data = pd.read_sql(sql, db)
        return data

    def fetch_data_from_table(self, table):
        db = create_engine('sqlite:///data/tosbik.db')
        data = pd.read_sql_table(table, db)
        return data

    def fetch_data_from_csv(self, file):
        path = f"./data/{file}"
        return pd.read_csv(path)

    def scale(self, values):
        scaled = self.scaler.fit_transform(values)
        return scaled

    def scale_inverse(self, values):
        return self.scaler.inverse_transform(values)


    def shift(self, data, n):
        data[self.cfg.shifted] = data[self.cfg.output].shift(n)
        data.dropna(inplace=True)
        return data

    def get_data_partitions(self, data, shifted=False):
        y = self.cfg.output
        if shifted:
            y = self.cfg.shifted
        size = len(data[y])
        size_train = int(size * 0.7)
        input_train = data[self.cfg.inputs][:size_train]
        input_test = data[self.cfg.inputs][size_train:]
        output_train = data[y][:size_train]
        output_test = data[y][size_train:]
        return input_test, input_train, output_test, output_train

    def normalize(self,df:pd.DataFrame):
        return (df - df.mean()) / df.std()


    def denormalize(self, df_norm:pd.DataFrame, mean, std):
        return (df_norm*std)+mean

    def minmax_scale(self, df: pd.DataFrame):
        return (df - df.min()) / (df.max() - df.min())

    def minmax_scale_inverse(self, df_scaled: pd.DataFrame,min,max):
        return ((df_scaled *(max-min))+ min )

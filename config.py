class Config:
    start_date = "2015-05-01 00:00:00"
    time_frames = ['1d', '4h']
    exchanges = {"binance": ["BTCUSDT", "ETHUSDT"],
                 "currencycom": ["US100", "US30", "US500", "DE40", "EU50", "EUR/USD", "Oil - Brent", "Oil - Crude",
                                 "Natural Gas", "Gold"]}
    columns = ['datetime', 'open', 'high', 'low', 'close', 'volume']
    inputs = ['open', 'high', 'low', 'close', 'volume']
    output = 'close'
    shifted='shifted'
    # regressors=["linear","ridge","svm","var","multi_layer_perceptron","naive_bayes","k_nearest_neigbor","random_forest","decision_tree","logistic"]
    regressors = ["linear", "ridge","multi_layer_perceptron","svm"]

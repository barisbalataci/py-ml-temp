from sklearn import linear_model, svm, tree
from sklearn.neural_network import MLPRegressor, MLPClassifier
from sklearn.ensemble import RandomForestRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.naive_bayes import GaussianNB
from statsmodels.tsa.vector_ar import var_model
import sklearn.metrics as metric
from config import Config
from data import Data
from matplotlib import pyplot


class Regressor:
    def __init__(self):
        pass

    def linear(self, input, output):
        model = linear_model.LinearRegression()
        model.fit(input, output)
        return model

    def ridge(self, input, output):
        model = linear_model.Ridge(alpha=10)
        model.fit(input, output)
        return model

    def logistic(self, input, output):
        model = linear_model.LogisticRegression(C=0.01, solver='liblinear')
        model.fit(input, output)
        return model

    def svm(self, input, output):
        model = svm.SVR(C=50, gamma=0.001, epsilon=0.2)
        model.fit(input, output)
        return model

    def decision_tree(self, input, output):
        model = tree.DecisionTreeRegressor(max_features='sqrt', max_depth=5)
        model.fit(input, output)
        return model

    def random_forest(self, input, output):
        model = RandomForestRegressor(200, max_features='sqrt', max_depth=11)
        model.fit(input, output)
        return model

    def k_nearest_neigbor(self, input, output):
        model = KNeighborsRegressor(n_neighbors=2, p=1)
        model.fit(input, output)
        return model

    def naive_bayes(self, input, output):
        model = GaussianNB()
        model.fit(input, output)
        return model

    def var(self, input, output):
        model = var_model.VAR(input, output)
        model.fit()
        return model

    def multi_layer_perceptron(self, input, output):
        model = MLPRegressor(hidden_layer_sizes=(5), alpha=.001, activation='tanh', solver='lbfgs')
        model.fit(input, output)
        return model


class ML_Engine:

    def __init__(self):
        self.cfg = Config()

    def train_models(self,input, output):
        regressor = Regressor()
        for ml_algo in self.cfg.regressors:
            if ml_algo not in ["VAR"]:
                model = getattr(regressor, ml_algo)
                yield model(input, output)

    @staticmethod
    def write_scores( model, y_hat, y_true):
        if model.__class__.__name__ not in ["VAR"]:
            mae = metric.mean_absolute_error(y_true, y_hat)
            mse = metric.mean_squared_error(y_true, y_hat)
            rmse = metric.mean_squared_error(y_true, y_hat, squared=False)
            r2 = metric.r2_score(y_true, y_hat)
            yield (model.__class__.__name__, mae, mse, rmse, r2)

    @staticmethod
    def plot_scores(yhat, output):
        pyplot.plot(yhat, label='predict')
        pyplot.plot(output, label='true')
        pyplot.legend()
        pyplot.show()

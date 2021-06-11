import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import smogn
import numpy as np
from pandas.core.common import SettingWithCopyWarning
import warnings
from constants import *
warnings.simplefilter(action="ignore", category=SettingWithCopyWarning)
# Splitting data into training/testing
from sklearn.model_selection import train_test_split

# Metrics
from sklearn.metrics import mean_squared_error, mean_absolute_error, median_absolute_error, r2_score
# Standard ML Models for comparison
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import ElasticNet
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.svm import SVR
# evaluate an xgboost regression model
import xgboost as xgb
from bayes_opt import BayesianOptimization
import os
import pickle
from keras.models import Sequential
from keras.layers import Dense, Activation
import keras.backend as K


def save_dict_pickle(path, dict, file):
    """
    save dictionary to pickle file
    :param path:
    :param dict:
    :param file:
    :return:
    """

    with open(os.path.join(path, file), 'wb') as handle:
        pickle.dump(dict, handle, protocol=pickle.HIGHEST_PROTOCOL)
    return

def load_dicc(path, file):
    """
    save dictionary to pickle file
    :param path:
    :param dict:
    :param file:
    :return:
    """

    with open(os.path.join(path, file), 'rb') as handle:
        dict= pickle.load(handle)
    return dict

def tilted_loss(q,y,f):
    """
    percentile loss function
    :param q:
    :param y:
    :param f:
    :return:
    """
    e = (y-f)
    return K.mean(K.maximum(q*e, (q-1)*e), axis=-1)

def priceModel(dim):
    """
    create model
    :param dim:
    :return:
    """
    model = Sequential()
    model.add(Dense(units=1024, input_dim=dim, activation='relu'))
    model.add(Dense(units=128, activation='relu'))
    model.add(Dense(1))

    return model

def generate_models_keras(dim,qs,X_train, X_test, y_train, y_test):
    """
    Generate n model accirding to vector qs
    exacmple qs = [ 0.01, 0.9, 0.95, 0.99, 1.0]
    :param dim:
    :param qs:
    :param X_train:
    :param X_test:
    :param y_train:
    :param y_test:
    :return:
    """
    models={}
    outs = {}
    ny_test = np.array(y_test)
    outs['true'] = ny_test

    # for q, color in zip(qs, colors):
    for q in qs:
        model = priceModel(dim)
        model.compile(loss=lambda y, f: tilted_loss(q, y, f), optimizer='adam')
        model.fit(X_train.values, y_train.values, validation_split=0.15, epochs=2000, batch_size=64, verbose=1)

        # Predict the quantile
        outs[str(q)] = model.predict(X_test)
        models[str(q)] = model
        # plt.plot(yhat,linewidth=1, label=q, color=color) # plot out this quantile
        print(f"Quantile {q}")

    return outs, models

def open_xls(file):
    """
    Open the excel and concat all sheet into one
    pass as paramether name of sheets
    :param file:
    :return: Dataframe
    """
    df2018 = pd.read_excel(file, index_col=0, sheet_name='FY18',
                           engine='openpyxl')
    df2018.dropna(inplace=True)
    df2019 = pd.read_excel(file, index_col=0, sheet_name='FY19',
                           engine='openpyxl')
    df2019.dropna(inplace=True)
    df2020 = pd.read_excel(file, index_col=0, sheet_name='FY20',
                           engine='openpyxl')
    df2020.dropna(inplace=True)
    df2021 = pd.read_excel(file, index_col=0, sheet_name='FY21',
                           engine='openpyxl')
    df2021.dropna(inplace=True)

    df = pd.concat([df2018, df2019, df2020, df2021])
    return df

def heatMap(df):
    """
    Create Correlation heatmap seaborn
    :param df:
    :return:
    """
#Create Correlation df
    corr = df.corr()
    #Plot figsize
    fig, ax = plt.subplots(figsize=(18, 18))
    #Generate Color Map
    colormap = sns.diverging_palette(220, 10, as_cmap=True)
    #Generate Heat Map, allow annotations and place floats in map
    sns.heatmap(corr, cmap=colormap, annot=True, fmt=".2f")
    #Apply xticks
    plt.xticks(range(len(corr.columns)), corr.columns);
    #Apply yticks
    plt.yticks(range(len(corr.columns)), corr.columns)
    #show plot
    plt.show()
    return ax
# standarize
def standarize(df, symbols):
    """
    Data Standarization x - mean / Std
    :param df:
    :param symbols:
    :return:
    """
    result = df.copy()
    values = {}
    for symbol in symbols:
        max_value = df[symbol].max()
        min_value = df[symbol].min()
        mean_value = df[symbol].mean()
        std_value = df[symbol].std()
        result[symbol] = (df[symbol] - mean_value) / std_value
        data = {}
        data.update({'max': max_value, 'min': min_value, 'mean': mean_value, 'std': std_value})

        values.update({symbol: data})
    return result, values

def standarize_pred(df, dicc, target):
    """
    Data Standarization x - mean / Std
    :param df:
    :param symbols:
    :return:
    """
    df=df[features]# only use the features we have defined for training
    def standard(row,col,dicc):
        mean_value=dicc[col]['mean']
        std_value = dicc[col]['std']
        return (row[col] -mean_value) / std_value

    for col in df.columns:
        if col != target:
            df[col]=df.apply(lambda row: standard(row,col,dicc), axis=1)

    return df

def apply_smogn(data, y,rg_mtrx):
    """
    Conduct Smogn. Some of this parameters can ba adapted to other scenarios
    read documentations https://github.com/nickkunz/smogn
    :param data:
    :param y:
    :param rg_mtrx:
    :return:
    """
    df_reduced_smogn = smogn.smoter(

        ## main arguments
        data=data,  ## pandas dataframe
        y=y,  ## string ('header name')
        k=3,  ## positive integer (k < n)
        pert=0.04,  ## real number (0 < R < 1)
        samp_method='extreme',  ## string ('balance' or 'extreme')
        drop_na_col=True,  ## boolean (True or False)
        drop_na_row=True,  ## boolean (True or False)
        replace=False,  ## boolean (True or False)

        ## phi relevance arguments
        rel_thres=0.10,  ## real number (0 < R < 1)
        rel_method='manual',  ## string ('auto' or 'manual')
        # rel_xtrm_type = 'both', ## unused (rel_method = 'manual')
        # rel_coef = 1.50,        ## unused (rel_method = 'manual')
        rel_ctrl_pts_rg=rg_mtrx  ## 2d array (format: [x, y])
    )
    return df_reduced_smogn

def apply_split_dataset(df_reduced_smogn,y, test_size=0.25, drop_target=False):
    """

    :param df_reduced_smogn:
    :return:
    """
    labels = df_reduced_smogn[y]
    df_reduced_smogn.columns = columns3
    cols = list(df_reduced_smogn.columns)
    df_normalized, dict_values = standarize(df_reduced_smogn, cols)

    path = "config"
    file = "standarize.pkl"
    save_dict_pickle(path, dict_values, file)


    # Split into training/testing sets with 25% split.
    if drop_target==False:
        X_train, X_test, y_train, y_test = train_test_split(df_normalized, np.log(labels),

                                                        test_size=test_size,
                                                        random_state=42)
    else:
        try:
            df_normalized= df_normalized[features].drop(columns=y)
        except:
            df_normalized = df_normalized[features]
        X_train, X_test, y_train, y_test = train_test_split(df_normalized[features], np.log(labels),

                                                            test_size=test_size,
                                                            random_state=12)

    return X_train, X_test, y_train, y_test

# Calculate mae and rmse
def evaluate_predictions(predictions, true):
    """

    :param predictions:
    :param true:
    :return:
    """
    mae = np.mean(abs(predictions - true))
    rmse = np.sqrt(np.mean((predictions - true) ** 2))
    r2 = r2_score(true, predictions)

    return mae, rmse, r2

# Evaluate several ml models by training on training set and testing on testing set
def evaluate(X_train, X_test, y_train, y_test):
    # Naive baseline is the median
    median_pred = X_train['main_price'].mean()
    median_preds = [median_pred for _ in range(len(X_test))]
    true = X_test['main_price']

    # Display the naive baseline metrics
    mb_mae, mb_rmse, r2_baseline = evaluate_predictions(median_preds, true)
    # Names of models
    model_name_list = ['Linear Regression', 'ElasticNet Regression',
                       'Random Forest', 'Extra Trees', 'SVM',
                       'Gradient Boosted', 'Baseline']
    X_train = X_train.drop(columns='main_price')
    X_test = X_test.drop(columns='main_price')

    # Instantiate the models
    model1 = LinearRegression()
    model2 = ElasticNet(alpha=1.0, l1_ratio=0.5)
    model3 = RandomForestRegressor(n_estimators=50)
    model4 = ExtraTreesRegressor(n_estimators=50)
    model5 = SVR(kernel='rbf', degree=3, C=1.0, gamma='auto')
    model6 = GradientBoostingRegressor(n_estimators=20)

    # Dataframe for results
    results = pd.DataFrame(columns=['mae', 'rmse', 'r2'], index=model_name_list)

    # Train and predict with each model
    for i, model in enumerate([model1, model2, model3, model4, model5, model6]):
        model.fit(X_train, y_train)
        predictions = model.predict(X_test)

        # Metrics
        mae = np.mean(abs(predictions - y_test))
        rmse = np.sqrt(np.mean((predictions - y_test) ** 2))
        r2 = r2_score(y_test, predictions)

        # Insert results into the dataframe
        model_name = model_name_list[i]
        results.loc[model_name, :] = [mae, rmse, r2]

    # Median Value Baseline Metrics
    baseline = np.mean(y_train)
    baseline_mae = np.mean(abs(baseline - y_test))
    baseline_rmse = np.sqrt(np.mean((baseline - y_test) ** 2))
    baseline_r2 = r2_baseline

    results.loc['Baseline', :] = [baseline_mae, baseline_rmse, baseline_r2]

    return results

def load_xgboost_model(path, file, params):
    """

    :param path:
    :param file:
    :return:
    """
    model_file=os.path.join(path, file)
    model=xgb.XGBRegressor(**params)
    model.load_model(model_file)
    return model

def prediction_xgboost(data, model):
    """
    prediction Xgboost
    :param data:
    :param model:
    :return:
    """
    yhat=model.predict(data)
    return yhat

def prediction_keras(Xtest, models,qs):
    preds={}
    for q in qs:
        preds[str(q)]= models[str(q)].predict(Xtest)
    return preds


def tranform_log(predictions):
    new_list = []
    for ele in list(predictions):
        new_list.append(np.exp(ele))
    return pd.Series(new_list)

def hp_tuning_bayesian_xg(X_train, X_test, y_train, y_test, n_iter, init_points,acq='ucb'):
    """

    :param X_train:
    :param X_test:
    :param y_train:
    :param y_test:
    :return:
    """

    def xgb_evaluate(max_depth, gamma, colsample_bytree, learning_rate, n_estimators, eta, min_child_weight):
        params = {'eval_metric': 'rmse',
                  'max_depth': int(max_depth),
                  'subsample': 0.8,
                  'learning_rate': learning_rate,
                  'n_estimators': int(n_estimators),
                  'eta': eta,
                  'gamma': gamma,
                  'min_child_weight': int(min_child_weight),
                  'colsample_bytree': colsample_bytree}
        # Used around 1000 boosting rounds in the full model
        cv_result = xgb.cv(params, dtrain, num_boost_round=999, nfold=5)

        # Bayesian optimization only knows how to maximize, not minimize, so return the negative RMSE
        return -1.0 * cv_result['test-rmse-mean'].iloc[-1]

    dtrain = xgb.DMatrix(X_train, label=y_train)
    dtest = xgb.DMatrix(X_test, label=y_test)

    xgb_bo = BayesianOptimization(xgb_evaluate, params_xg)
    # Use the expected improvement acquisition function to handle negative numbers
    # Optimally needs quite a few more initiation points and number of iterations
    xgb_bo.maximize(init_points=init_points, n_iter=n_iter, acq=acq)

    return xgb_bo, dtrain, dtest

def get_params(xgb_bo):
    """

    :param xgb_bo:
    :return:
    """
    params = xgb_bo.max['params']
    params['max_depth'] = int(params['max_depth'])
    params['n_estimators'] = int(params['n_estimators'])

    return params
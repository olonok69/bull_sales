import numpy as np
from pandas.core.common import SettingWithCopyWarning
import warnings
warnings.simplefilter(action="ignore", category=SettingWithCopyWarning)
from sklearn.metrics import  r2_score
from constants import *
from utils import *
import argparse
from tensorflow.keras.models import load_model
from IPython import embed



def main():
    # parse parameters
    # move this to conf / yaml file
    qs = [0.01, 0.5, 0.9, 0.95, 0.99, 1.0]
    num_boost_round = 999
    parser = argparse.ArgumentParser("Bull_sales")

    parser.add_argument("--input_data", type=str, help="input data" ,required=False)
    parser.add_argument("--algo", type=str, help="xgboost or keras", required=True)
    parser.add_argument("--mode", type=str, help="training or prediction", required=True)
    parser.add_argument("--prediction_file", type=str, help="csv file with features for prediction", required=False)
    args = parser.parse_args()
    if args.mode=='training':

        df =open_xls(args.input_data)#'data/AAA_4yr_SalesReport_Bulls_May2021.xlsx'
        df.to_csv("data/Bull_sales_report.csv", index=False)
        # rename columns
        df.columns=columns1

        df_reduced = df[columns2]
        df_reduced.reset_index(inplace=True)
        try:
            df_reduced.drop('FY', axis=1, inplace=True)
        except:
            pass

        assert df_reduced.isnull().sum().sum() == 0
        df_reduced_smogn = apply_smogn(df_reduced, "main_price", rg_mtrx)

    if args.algo =="xgboost":

        if args.mode=='training':
        # Split into training/testing sets with 25% split.
            X_train, X_test, y_train, y_test = apply_split_dataset(df_reduced_smogn,'main_price',0.25, False)
            #evaluate
            results = evaluate(X_train, X_test, y_train, y_test)
            print(results)

            # HP tuning
            # Split into training/testing sets with 25% split.
            X_train, X_test, y_train, y_test = apply_split_dataset(df_reduced_smogn,'main_price',0.25, True)
            xgb_bo, dtrain, dtest =hp_tuning_bayesian_xg(X_train, X_test, y_train, y_test, 7, 500,'ucb')
            print(xgb_bo.max)

            params= get_params(xgb_bo)
            model = xgb.train(
                params,
                dtrain,
                num_boost_round=num_boost_round,
                evals=[(dtest, "Test")],
                early_stopping_rounds=10
            )
            num_boost_round = model.best_iteration + 1
            best_model = xgb.train(
                params,
                dtrain,
                num_boost_round=num_boost_round,
                evals=[(dtest, "Test")]
            )
            r2_score(best_model.predict(dtest), y_test)
            # save model
            best_model.save_model('models/xg_model.pkl')
            # save dicc parameters
            path = "config"
            file = "paramns_xgboost.pkl"
            save_dict_pickle(path, params, file)
            return

        elif args.mode == 'prediction':
            #load files
            configuration = load_dicc("config","standarize.pkl")
            params_xg = load_dicc("config", "paramns_xgboost.pkl")
            # load prediction File
            pred_file = pd.read_csv(args.prediction_file)
            pred_file.columns= columns3
            # standarize values
            standard_pred=standarize_pred(pred_file, configuration, 'main_price')
            # load xgboost model
            params = load_dicc("config", "paramns_xgboost.pkl")
            model=load_xgboost_model("models", "xg_model.pkl", params)
            # remove target value if exists
            try:
                Xtest=standard_pred.drop(columns='main_price')
            except:
                Xtest=standard_pred.copy()
            # Do predictions
            predictions=prediction_xgboost(Xtest, model)
            # transform log prediction
            predictions=tranform_log(predictions)
            # metrics
            r2 = r2_score(pred_file['main_price'], predictions)

            # add column with predictions to the original file
            pred_file['predictions']=predictions
            # serialize
            pred_file.to_csv('data/xg_predictions.csv', index=False)
            return

    # Keras Predictor
    elif args.algo == "keras":
        # training
        if args.mode == 'training':
            # HP tuning
            # Split into training/testing sets with 25% split.
            X_train, X_test, y_train, y_test = apply_split_dataset(df_reduced_smogn, 'main_price', 0.1, True)
            # dimiension input layer
            dim=X_train.shape[1]
            # Move this to yaml / json file

            # generate keras models one per percentile
            # train models and serialize
            outs, models = generate_models_keras(dim, qs, X_train, X_test, y_train, y_test)
            for key in models.keys():
                models[key].save(f"models/keras_percentile{key}.h5")
        elif args.mode == 'prediction':
            # load files
            configuration = load_dicc("config", "standarize.pkl")

            # load prediction File
            pred_file = pd.read_csv(args.prediction_file)
            pred_file.columns = columns3
            # standarize values
            standard_pred = standarize_pred(pred_file, configuration, 'main_price')
            # load models
            models={}
            path="models"

            for q in qs:
                # load models and not compile.
                models[str(q)]=load_model(os.path.join(path, f"keras_percentile{str(q)}.h5"), compile=False)

            # remove target value if exists
            try:
                Xtest=standard_pred.drop(columns='main_price')
            except:
                Xtest=standard_pred.copy()
            # do predictions with all saved models
            predictions=prediction_keras(Xtest, models,qs)
            final_preds={}
            for q in qs:
                final_preds[str(q)]={}
                final_preds[str(q)]['preds'] = tranform_log(predictions[str(q)])
                final_preds[str(q)]['r2']= r2_score(pred_file['main_price'], final_preds[str(q)]['preds'])
                pred_file[f'preds_{q}']=final_preds[str(q)]['preds']

            # serialize
            pred_file.to_csv('data/keras_predictions.csv', index=False)


if __name__ == '__main__':

    main()
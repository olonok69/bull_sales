# bull_sales
Quantile Regression and Synthetic Minority Over-Sampling Technique for Regression with Gaussian Noise

This application use SMOGN technique to do oversampling in a regression problem. We use two predictors to predict sales values. Log transformation and Standardization is applied before training process, and we use Xg-boost Regressor and a Neural Network with a quantile loss function

For Reference:


SMOGN:
https://github.com/nickkunz/smogn#synthetic-minority-over-sampling-technique-for-regression-with-gaussian-noise


Deep Quantile Regression:
https://towardsdatascience.com/deep-quantile-regression-c85481548b5a


Bayesian Optimization (only for XGboost)
https://github.com/fmfn/BayesianOptimization


training mode
python main.py --input_data data/AAA_4yr_SalesReport_Bulls_May2021.xlsx –algo keras or xgboost –mode training

prediction mode
python main.py  –algo keras or xgboost –mode prediction --prediction_file data/predictions.csv

All data input output is expected in an internal folder data. Models are saved in models folder. In config we save a copy of dictionaries use to standardize the data and HP tuning params for xgboost. in Xgboost training we do a Bayesian optimization of hyper-parameters, and the process may take a couple of hours.

For keras by defauls we use the vector qs with values [0.01, 0.5, 0.9, 0.95, 0.99, 1.0] to train the Neural network on those quantiles of the main price distribution. Later at perdition time,  each percentile is use to do a prediction. Recommend to take predictions at .5 percentile unless you want to predictions on the boundaries of the pdb 

For prediction is expected the same format as input


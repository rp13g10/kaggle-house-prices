#%% Module imports
from eda import load_validated_data
from etl import transform_data, derive_fields

from tqdm import tqdm
import sklearn as sk
from sklearn.model_selection import train_test_split, cross_validate, RandomizedSearchCV
from sklearn import linear_model, svm, neighbors, gaussian_process, tree, ensemble
import xgboost
import pandas as pd
from copy import deepcopy

# #%% Load and validate data
# train, metadata = load_validated_data('train.csv')
# val, _ = load_validated_data('test.csv')

# #%% Fill in/replace values to cleanse dataset
# train = transform_data(train, metadata)
# val = transform_data(val, metadata)

# #%% Derive extra fields to assist model fitting
# train, val = derive_fields(train, val, metadata)

# #%% Split into train/test/validation subsets
# features = val.columns.tolist()

# X_full = train[features]
# y_full = train['SalePrice']
# X_val = val[features]

# X_train, X_test, y_train, y_test = train_test_split(X_full, y_full, test_size=0.3, random_state=42)

#%% Pick which algorithm to use
# Notes
# - Small sample size, not much point using a neural network
# - Regression model required
# - Always worth trying XGBoost

def evaluate_model_performance(X_full, y_full):
    k = {'random_state': 42}
    it = {'max_iter': 100000}

    # To start off, we'll try a wide range of regressors using the default parameters
    regressors = [
        # Linear Models
        linear_model.LinearRegression(),
        linear_model.Ridge(**k, **it),
        linear_model.Lasso(**k, **it),
        linear_model.ElasticNet(**k, **it),
        linear_model.BayesianRidge(),
        linear_model.LogisticRegression(**k, **it),
        linear_model.SGDRegressor(**k, **it),
        # Support Vector Machine
        svm.SVR(**it),
        # Nearest Neighbours
        neighbors.KNeighborsRegressor(),
        # Gaussian Process
        gaussian_process.GaussianProcessRegressor(**k),
        # Decision Trees
        tree.DecisionTreeRegressor(**k),
        # Ensemble Methods
        ensemble.RandomForestRegressor(**k),
        ensemble.ExtraTreesRegressor(**k),
        ensemble.AdaBoostRegressor(**k),
        ensemble.GradientBoostingRegressor(**k),
        xgboost.XGBRegressor(**k)
    ]

    results = []
    reg_iter = tqdm(regressors)
    for regressor in reg_iter:
        model_name = regressor.__class__.__name__
        reg_iter.set_description(model_name)

        # Cross-validate to pick the best candidate algorithm
        # RMSE selected as that's what the Kaggle competition is evaluated on
        cv_results = cross_validate(
            estimator=regressor,
            X=X_full,
            y=y_full,
            cv=5,
            n_jobs=4,
            return_train_score=True,
            scoring='neg_root_mean_squared_error'
        )

        # Spit results out for consumption
        out = {
            'model_name': model_name,
            'fit_time': cv_results['fit_time'].mean(),
            'train_score': cv_results['train_score'].mean(),
            'test_score': cv_results['test_score'].mean(),
        }

        results.append(out)

    results = pd.DataFrame(results).sort_values(by='test_score', ascending=False)
    # results['train_score'] = results['train_score'].map('{:,.2f}'.format)
    # results['test_score'] = results['test_score'].map('{:,.2f}'.format)

    return results

#%% Take a look at the output
# results

# Looks like the ensemble methods are by-and-large the best performing
def tune_selected_models(X_full, y_full, X_val):
    k = {'random_state': 42}

    # %% Try to improve results through hyperparameter optimization
    candidates = [
        {
            'model': ensemble.GradientBoostingRegressor(**k),
            'params': {
                'n_estimators': [50, 100, 250, 1000],
                'subsample': [0.1, 0.25, 0.5, 1.0],
                'min_samples_split': [2, 5],
                'min_samples_leaf': [1, 2],
                'max_features': ['auto', 'sqrt', 'log2']
            },
        },
        {
            'model': ensemble.RandomForestRegressor(**k),
            'params': {
                'n_estimators': [50, 100, 250, 1000],
                'min_samples_split': [2, 5],
                'min_samples_leaf': [1, 2],
                'max_features': ['auto', 'sqrt', 'log2']
            },
        },
        {
            'model': ensemble.ExtraTreesRegressor(**k),
            'params': {
                'n_estimators': [50, 100, 250, 1000],
                'min_samples_split': [2, 5],
                'min_samples_leaf': [1, 2],
                'max_features': ['auto', 'sqrt', 'log2']
            },
        },
        {
            'model': xgboost.XGBRegressor(**k),
            'params': {
                'n_estimators': [50, 100, 250, 1000],
                'learning_rate': [0.01, 0.02, 0.1, 0.2],
                'colsample_by_tree': [0.1, 0.25, 0.5, 1.0],
                'gamma': [0,1,2,5], #min loss reduction required for new branch split
                'reg_lambda': [0,1,2] #l2 regularization strength
            },
        },
    ]

    results = []
    models = []
    cand_iter = tqdm(candidates)
    for candidate in cand_iter:
        model = candidate['model']
        params = candidate['params']

        optimizer = RandomizedSearchCV(
            estimator=model,
            param_distributions=params,
            scoring='neg_root_mean_squared_error',
            cv=5,
            n_jobs=4,
            return_train_score=True
        )

        optimizer.fit(X_full, y_full)

        out = {
            'model_name': model.__class__.__name__,
            'best_params': optimizer.best_params_,
            'best_score': optimizer.best_score_,
            'mean_train_score': optimizer.cv_results_['mean_train_score'][optimizer.best_index_],
            'mean_test_score': optimizer.cv_results_['mean_test_score'][optimizer.best_index_],
            'std_train_score': optimizer.cv_results_['std_train_score'][optimizer.best_index_],
            'std_test_score': optimizer.cv_results_['std_test_score'][optimizer.best_index_]
        }

        preds = optimizer.predict(X_val)
        pred_df = pd.DataFrame(X_val.index, columns=['Id'])
        pred_df.loc[:,'SalePrice'] = preds
        pred_df.to_csv(f'{model.__class__.__name__}.csv', index=False, encoding='utf8', mode='w')

        results.append(out)

        model = (model.__class__.__name__, deepcopy(optimizer.best_estimator_))
        models.append(model)

    results = pd.DataFrame(results)

    return results, models

def make_ensemble_prediction(X_full, y_full, X_val, models):

    parliament = ensemble.VotingRegressor(
        estimators=models)

    parliament.fit(X_full, y_full)
    preds = parliament.predict(X_val)
    pred_df = pd.DataFrame(X_val.index, columns=['Id'])
    pred_df.loc[:,'SalePrice'] = preds
    pred_df.to_csv('ensemble.csv', index=False, encoding='utf8', mode='w')

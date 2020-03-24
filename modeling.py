'''This file contains the functions required to evaluate the performance of a
variety of regression models, and to select the optimal parameters for 4
models which were experimentally determined to perform the best on the
data provided.


Folder structure
- eda.py - General exploration of the dataset
- etl.py - Tidying the datset in preparation for modeling
- modeling.py - Evaluate performance of various models
- Summary of Results.ipynb - Primary script, coordinates execution of all functions'''

#%% Module imports
from copy import deepcopy
from sklearn.model_selection import cross_validate, RandomizedSearchCV
from sklearn import linear_model, svm, neighbors, gaussian_process, tree, ensemble
from tqdm import tqdm
import pandas as pd
import xgboost


def evaluate_model_performance(X_full, y_full):
    '''This function will evaluate the performance of a variety of different regression
    models on the data, returning a dataframe summarising their performance when
    fitted using the default parameters'''

    # Arguments which will be applied to the majority of models
    k = {'random_state': 42}    # Ensure reproducible results
    it = {'max_iter': 100000}   # Give models adequate time to converge

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

    # For each model in the above list
    results = []
    reg_iter = tqdm(regressors) # Displays a progress bar
    for regressor in reg_iter:
        # Get the name of the current model, display it in the progress bar
        model_name = regressor.__class__.__name__
        reg_iter.set_description(model_name)

        # Cross-validate to pick the best candidate algorithm
        # RMSE selected as that's what the Kaggle competition is evaluated on
        # This implements the train/test split for me!
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

    # Convert the final results into a dataframe for easier viewing
    results = pd.DataFrame(results).sort_values(by='test_score', ascending=False)

    return results


# Looks like the ensemble methods are by-and-large the best performing
def tune_selected_models(X_full, y_full, X_val):
    '''For the best performing models, this function now uses a randomized search to
    determine the optimal parameters for them. It returns the best performing configuration
    for each of the candidate models'''

    # Applied to all models
    k = {'random_state': 42}

    # Try to improve results through hyperparameter optimization
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
                'gamma': [0, 1, 2, 5], #min loss reduction required for new branch split
                'reg_lambda': [0, 1, 2] #l2 regularization strength
            },
        },
    ]

    # Optimize one model at a time, with a progress bar
    results = []
    models = []
    cand_iter = tqdm(candidates)
    for candidate in cand_iter:

        # Extract necessary variables from dictionary
        model = candidate['model']
        params = candidate['params']

        # Fit multiple models, determine the optimal parameters using cross validation
        optimizer = RandomizedSearchCV(
            estimator=model,
            param_distributions=params,
            scoring='neg_root_mean_squared_error',
            cv=5,
            n_jobs=4,
            return_train_score=True
        )
        optimizer.fit(X_full, y_full)

        # Preserve relevant metrics for inspection
        out = {
            'model_name': model.__class__.__name__,
            'best_params': optimizer.best_params_,
            'best_score': optimizer.best_score_,
            'mean_train_score': optimizer.cv_results_['mean_train_score'][optimizer.best_index_],
            'mean_test_score': optimizer.cv_results_['mean_test_score'][optimizer.best_index_],
            'std_train_score': optimizer.cv_results_['std_train_score'][optimizer.best_index_],
            'std_test_score': optimizer.cv_results_['std_test_score'][optimizer.best_index_]
        }
        results.append(out)

        # Make predictions using the best-performing model, write them to a csv file
        preds = optimizer.predict(X_val)
        pred_df = pd.DataFrame(X_val.index, columns=['Id'])
        pred_df.loc[:, 'SalePrice'] = preds
        pred_df.to_csv(
            f'predictions/{model.__class__.__name__}.csv',
            index=False,
            encoding='utf8',
            mode='w')

        # Save the model to memory for use in the VotingRegressor
        model = (model.__class__.__name__, deepcopy(optimizer.best_estimator_))
        models.append(model)

    # Convert model performance metrics to dataframe for viewing
    results = pd.DataFrame(results)

    return results, models

def make_ensemble_prediction(X_full, y_full, X_val, models):
    '''Take the final models from the previous stage, use each of them to
    make a prediction and take the average.'''

    # Create an ensemble containing all of the selected models
    parliament = ensemble.VotingRegressor(
        estimators=models)

    # Make predictions using this final model
    parliament.fit(X_full, y_full)
    preds = parliament.predict(X_val)

    # Write the output to a csv file
    pred_df = pd.DataFrame(X_val.index, columns=['Id'])
    pred_df.loc[:, 'SalePrice'] = preds
    pred_df.to_csv('predictions/VotingRegressor.csv', index=False, encoding='utf8', mode='w')

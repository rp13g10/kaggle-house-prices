# %%
import pandas as pd
import plotly.express as px
from IPython.display import clear_output


# %%
from eda import load_validated_data
from etl import transform_data, derive_fields
from modeling import evaluate_model_performance, tune_selected_models, make_ensemble_prediction


# %%
# Load in the two datasets, validating everything against the provided metadata
train, metadata = load_validated_data('data/train.csv')
val, _ = load_validated_data('data/test.csv')


# %%
# Visually inspect the loaded data
train.head()

# %% [markdown]
# ## How much are the properties worth?

# %%
# Generate a plot to see what the distribution of sale prices looks like
fig = px.histogram(train.reset_index(),
                   x='SalePrice',
                   y='Id',
                   marginal='box',
                   title='Distribution of Sales Prices',
                   labels={'SalePrice': 'Sale Price ($)', 'Id': 'Properties'},
                   width=1280,
                   height=720)
fig.write_image('figures/SalePriceDistribution.png', scale=4)
fig.show()

# %% [markdown]
# ## How can we get the data into a usable format?

# %%
# Work out what % of each column is populated
tot_records = len(train.index)
counts = train.count(axis='index')
counts /= tot_records
counts = counts.sort_values(ascending=True)
counts = pd.DataFrame(counts, columns=['populated']).reset_index()
counts = counts.rename({'index': 'field'})
counts.loc[:, 'nulls'] = 1 - counts['populated']
counts = counts.loc[counts['populated'] < 1, :]

# Plot the output for all columns which have blank values
fig = px.bar(counts,
             x='index',
             y='populated',
             title='Availability of Data',
             labels={'populated': 'Populated (%)', 'index': 'Field'},
             height=720,
             width=1280)
fig.write_image('figures/DataAvailability.png', scale=4)
fig.show()

# %% [markdown]
# <b>Recommendations for missing values</b>
# * PoolArea - This will be dropped in favour of PoolQC which is fully populated
# * 3SsnPorch, ScreenPorch, EnclosedPorch, OpenPorchSF - These will be merged into a single Porch column
# * LowQualFinSF - Would expect this to be mostly empty, fill blanks with 0
# * BsmtFinSF2 - Will be dropped in favour of BsmtFinType2 which is fully populated
# * SaleType - Dropped as I can't see any good way to fill in the gaps
# * MasVnrArea - Drop in favour of MasVnrType
# * 2ndFlrSF - Fill with mean if the building isn't a 1-storey construction
# * WoodDeckSF - Fill gaps with 0, no decking
# * BsmtFinSF1 - Fill blanks with 0 if BsmtFinType1 is NA, otherwise use TotalBsmtSF - BsmtUnfSF
# * LotFrontage - Fill blanks with the mean
#     * Outliers in this field, leaving as-is for now but model could be improved by using median
# * BsmtUnfSF - Fill blanks with 0
# * GarageArea - Fill blanks with 0
# * GarageYrBlt - Fill blanks mean
#     * Hopefully less impact on overall model than filling with 0s as this is a numerical field
#     * Could add a GaragePresent field to mitigate this further in future revisions
# * TotalBsmtSF - fill blanks with 0
# * Electrical - Fill blanks with mode

# %%
# Visually check the distribution of values for each field
for col in train.columns:
    clear_output()
    plot_df = train.dropna(subset=[col])
    fig = px.histogram(plot_df.reset_index(), x=col, y='Id', title=col, width=1280, height=720)
    fig.show()
    check = input('Enter to continue, q to quit, p to save plot to image')
    if check == 'q':
        clear_output()
        break
    elif check == 'p':
        fig.write_image(f'figures/{col}.png', scale=4)
clear_output()

# %% [markdown]
# <b> Notes on field contents </b>
# * LotFrontage has an outlier at 310
# * LotArea highly skewed
# * Street almost entirely Pave
# * Alley might as well be switched over to binary format
# * Utilities could probably be dropped due to low variance
# * Condition2 could probably be dropped
# * Heating could probably be dropped
# * Full/Half bath columns can be merged, little data in half bath columns
# * MiscFeature is almost entirely sheds, just keep the numerical equivalent MiscVal
# %% [markdown]
# ## Which type of model performs the best?
# ### Prepare the data

# %%
# Fill in/replace values to cleanse the dataset
train = transform_data(train, metadata)
val = transform_data(val, metadata)


# %%
# Derive extra fields to assist with model fitting
train, val = derive_fields(train, val, metadata)


# %%
# Split into X, y datasets
# Note that train/test split is not called explicitly as it's built into the
#     cross validation process
features = val.columns.tolist()

X_full = train[features]
y_full = train['SalePrice']
X_val = val[features]


# %%
# Test the various candidate algorithms
algo_results = evaluate_model_performance(X_full, y_full)


# %%
# Evaluate model performance on train/test datasets
pretty_results = algo_results.copy()
pretty_results['train_score'] = pretty_results['train_score'].map('{:,.0f}'.format)
pretty_results['test_score'] = pretty_results['test_score'].map('{:,.0f}'.format)
pretty_results


# %%
# Use RandomSearchCV to find the best set of parameters for each model
cand_results, models = tune_selected_models(X_full, y_full, X_val)


# %%
# Inspect final results for the 4 candidate models
cand_results


# %%
# Finally, use an ensemble regressor to try and eke out a bit more performance
make_ensemble_prediction(X_full, y_full, X_val, models)

# %% [markdown]
# ## Which features have the greatest importance in determining price?

# %%
# Extract the fitted GradientBoostingRegressor object
GBoost = models[0][1]

# Get a list of columns and the corresponding feature importances
records = list(zip(X_full.columns, GBoost.feature_importances_))

# Parse list into a dataframe, bring through field descriptions from metadata
features = pd.DataFrame.from_records(records, columns=['field', 'importance'])
features = features.sort_values(by='importance', ascending=False)
features.loc[:, 'base_field'] = features['field'].map(lambda x: x.split('_')[0])
features.loc[:, 'field_desc'] = features['base_field'].map(
    lambda x: metadata[x]['description'] if x in metadata else pd.NA)
features = features[['field', 'field_desc', 'importance']]

# Visually inspect feature importances
features.head(20)


# %%
# Generate a plot showing feature importance above a threshold
plot_features = features.loc[features['importance'] >= 0.005, :]
fig = px.bar(plot_features,
             x='field',
             y='importance',
             hover_data=['field_desc'],
             width=1280,
             height=720,
             title='Feature Importance')
fig.write_image('figures/FeatureImportance.png', scale=4)
fig.show()

# %% [markdown]
# ## How close did we really get?

# %%
# Make predictions using the GradientBoostingRegressor
y_preds = GBoost.predict(X_full)


# %%
# Add predictions to a dataframe along with the actual values
results = train[['SalePrice']].copy()
results.loc[:, 'Predicted'] = y_preds

# Calculate prediction variance from true values
results.loc[:, 'Variance'] = results['SalePrice'] - results['Predicted']


# %%
# Generate a plot showing the distribution of the variance
fig = px.histogram(
    results.reset_index(),
    x='Variance',
    y='Id',
    marginal='box',
    labels={'Variance': 'Error ($)', 'Id': 'Properties'},
    title='Prediction Error on Training Dataset',
    width=1280,
    height=720)
fig.write_image('figures/PredictionErrors.png', scale=4)
fig.show()


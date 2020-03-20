'''This file implements the recommendations made following the work done in eda.py
Initial validation of the data provided was performed in eda.py

Folder structure
- eda.py - General exploration of the dataset
- etl.py - Tidying the datset in preparation for modeling
- modeling.py - Evaluate performance of various models
- generate_predictions.py - Apply final model to generate predictions'''

#%% Import modules
import pandas as pd
from eda import load_validated_data

#%% Load in data
train, metadata = load_validated_data('train.csv')

def find_cols(search):
    '''Convenience function to find metadata for columns containing a
    key word'''
    keys = [x for x in metadata if search.lower() in x.lower()]
    return {k: v for k, v in metadata.items() if k in keys}

#%% Merge all porch columns into one
porch_cols = [x for x in find_cols('porch')]
def coalesce(row):
    '''Analagous to coalesce in MSSQL, takes the first non-null value in the row
    provided'''
    try:
        return next(x for x in row if pd.notna(x) and x != 0)
    except StopIteration:
        return pd.NA

# Combine all porch fields, drop the originals
train.loc[:, 'PorchSF'] = train.loc[:, porch_cols].apply(coalesce, axis=1)
train = train.drop(columns=porch_cols)

#%% Drop redundant columns
train = train.drop(columns=[
    'PoolArea',     # Using PoolQC instead
    'BsmtFinSF2',   # Using BsmtFinType2 instead
    'SaleType',     # No clear way to fill the gaps
    'MasVnrArea',   # Using MasVnrType instead
    'MiscFeature',  # Almost exclusively sheds, using MiscVal instead
    'Street',       # Only 6 Grvl values, everything else is Pave
    'Utilities',    # Only 1 NoSeWa value, everything else is AllPub
])

#%% Remove any 0s from numerical fields, deal with them using logic defined in eda.py
num_cols = [x for x in metadata if metadata[x]['type']=='number' and x in train.columns]
train.loc[:, num_cols] = train.loc[:, num_cols].replace(0, pd.NA)

#%% Fill numerical fields with 0 (if it makes sense)
zero_cols = [
    'LowQualFinSF',
    'WoodDeckSF',
    'BsmtUnfSF',
    'TotalBsmtSF',
    'MiscVal',
    'PorchSF',
    'BsmtHalfBath',
    'HalfBath',
    'BsmtFullBath',
    'FullBath',
    'Fireplaces',
    'GarageCars',
    'GarageArea',
    'BedroomAbvGr',
    'KitchenAbvGr'
]

train.loc[:, zero_cols] = train.loc[:, zero_cols].fillna(0)
assert pd.isna(train[zero_cols]).sum().sum() == 0


#%% Fill numerical fields with the mean (if 0s don't make sense)
mean_cols = [
    'LotFrontage',
    'GarageYrBlt'
]

train.loc[:, mean_cols] = train.loc[:, mean_cols].apply(
    lambda col: col.fillna(col.mean()),
    axis=0
)
assert pd.isna(train[mean_cols]).sum().sum() == 0

#%% Fill categorical fields with the mode
cat_cols = [
    'Neighborhood',
    'BldgType',
    'MSZoning',
    'MasVnrType',
    'Electrical'
]

train.loc[:, cat_cols] = train.loc[:, cat_cols].apply(
    lambda col: col.fillna(col.mode().values[0]),
    axis=0
)
assert pd.isna(train[cat_cols]).sum().sum() == 0


#%% Convert fields with low variance to binary
low_var_cols = [
    'Condition2',
    'Heating'
]

# Convert to binary flag for the most common value
train.loc[:, low_var_cols] = train.loc[:, low_var_cols].apply(
    lambda col: (col == col.mode().values[0]).astype(int),
    axis=0
)
assert pd.isna(train[low_var_cols]).sum().sum() == 0

#%% Deal with fields which require more complex processing
# Convert alley to binary indicator showing if one is present
train.loc[:, 'Alley'] = (train.loc[:, 'Alley'] != 'NA').astype(int)
assert pd.isna(train['Alley']).sum() == 0

#%% Fill 2ndFlrSF with mean where building type isn't a 1-storey structure
# Fill with zeros where it is
def fill_2ndFlrSF(row, meansf):
    style = row['HouseStyle']
    sf = row['2ndFlrSF']
    if style == '1Story':
        return 0
    elif style in {'1.5Fin', '1.5Unf', '2Story', '2.5Fin', '2.5Unf', 'SFoyer', 'SLvl'}:
        return sf if pd.notna(sf) else meansf

train.loc[:, '2ndFlrSF'] = train.apply(
    lambda row: fill_2ndFlrSF(row, meansf=train['2ndFlrSF'].mean()),
    axis=1
)
assert pd.isna(train['2ndFlrSF']).sum() == 0

#%% Conditionally fill in BsmtFinSF1
def fill_BsmtFinSF1(row):
    finsf = row['BsmtFinSF1']
    fintype = row['BsmtFinType1']
    totsf = row['TotalBsmtSF']
    unfsf = row['BsmtUnfSF']
    gfsf = row['1stFlrSF']

    # Retain value if provided
    if pd.notna(finsf):
        return finsf
    # If basement isn't finished, finished area must be 0
    elif fintype in {'NA', 'Unf'}:
        return 0
    # Infer from total area and unfinished area if possible
    elif pd.notna(totsf) and pd.notna(unfsf):
        return totsf - unfsf
    # Otherwise assume area is the same as the ground floor
    else:
        return gfsf


train.loc[:, 'BsmtFinSF1'] = train.apply(
    fill_BsmtFinSF1,
    axis=1
)
assert pd.isna(train['BsmtFinSF1']).sum() == 0

# %% Derive some aggregate statistics on SalePrice across categories
cat_cols = [x for x in metadata if metadata[x]['type']=='category' and x in train.columns]

# Only consider columns which are fully populated
cat_cols = [x for x in cat_cols if 'NA' not in train[x].value_counts()]

# No cheating, figures need to be aggregates not just the SalePrice for the given record
cat_cols = [x for x in cat_cols if min(train[x].value_counts()) > 1]

for cat_col in cat_cols:
    cat_stats = train.groupby(cat_col).agg(**{
        f"{cat_col}_min": ('SalePrice', 'min'),
        f"{cat_col}_max": ('SalePrice', 'max'),
        f"{cat_col}_mean": ('SalePrice', 'mean')
    }).reset_index()
    train = pd.merge(left=train, right=cat_stats, on=cat_col, how='left')

# %% At this point, no NA objects should exist (but 'NA' as a string is OK)
assert pd.isna(train).sum().sum() == 0

# %%

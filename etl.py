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
from sklearn.preprocessing import StandardScaler, QuantileTransformer

#%% Load in data
# train, metadata = load_validated_data('train.csv')

def transform_data(df, metadata):
    '''Transform validated dataset into a state which can be used for modelling'''

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
    df.loc[:, 'PorchSF'] = df.loc[:, porch_cols].apply(coalesce, axis=1)
    df = df.drop(columns=porch_cols)

    #%% Drop redundant columns
    df = df.drop(columns=[
        'PoolArea',     # Using PoolQC instead
        'BsmtFinSF2',   # Using BsmtFinType2 instead
        'SaleType',     # No clear way to fill the gaps
        'MasVnrArea',   # Using MasVnrType instead
        'MiscFeature',  # Almost exclusively sheds, using MiscVal instead
        'Street',       # Only 6 Grvl values, everything else is Pave
        'Utilities',    # Only 1 NoSeWa value, everything else is AllPub
    ])

    #%% Remove any 0s from numerical fields, deal with them using logic defined in eda.py
    num_cols = [x for x in metadata if metadata[x]['type']=='number' and x in df.columns]
    df.loc[:, num_cols] = df.loc[:, num_cols].replace(0, pd.NA)

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

    df.loc[:, zero_cols] = df.loc[:, zero_cols].fillna(0)
    assert pd.isna(df[zero_cols]).sum().sum() == 0


    #%% Fill numerical fields with the mean (if 0s don't make sense)
    mean_cols = [
        'LotFrontage',
        'GarageYrBlt'
    ]

    df.loc[:, mean_cols] = df.loc[:, mean_cols].apply(
        lambda col: col.fillna(col.mean()),
        axis=0
    )
    assert pd.isna(df[mean_cols]).sum().sum() == 0

    #%% Fill categorical fields with the mode
    cat_cols = [
        'Neighborhood',
        'BldgType',
        'MSZoning',
        'MasVnrType',
        'Electrical',
        'Exterior1st',
        'Exterior2nd',
        'KitchenQual',
        'Functional'
    ]

    df.loc[:, cat_cols] = df.loc[:, cat_cols].apply(
        lambda col: col.fillna(col.mode().values[0]),
        axis=0
    )
    assert pd.isna(df[cat_cols]).sum().sum() == 0


    #%% Convert fields with low variance to binary  
    low_var_cols = [
        'Condition2',
        'Heating'
    ]

    # Convert to binary flag for the most common value
    df.loc[:, low_var_cols] = df.loc[:, low_var_cols].apply(
        lambda col: (col == col.mode().values[0]).astype(int),
        axis=0
    )
    assert pd.isna(df[low_var_cols]).sum().sum() == 0

    #%% Deal with fields which require more complex processing
    # Convert alley to binary indicator showing if one is present
    df.loc[:, 'Alley'] = (df.loc[:, 'Alley'] != 'NA').astype(int)
    assert pd.isna(df['Alley']).sum() == 0

    #%% Fill 2ndFlrSF with mean where building type isn't a 1-storey structure
    # Fill with zeros where it is
    def fill_2ndFlrSF(row, meansf):
        style = row['HouseStyle']
        sf = row['2ndFlrSF']
        if style == '1Story':
            return 0
        elif style in {'1.5Fin', '1.5Unf', '2Story', '2.5Fin', '2.5Unf', 'SFoyer', 'SLvl'}:
            return sf if pd.notna(sf) else meansf

    df.loc[:, '2ndFlrSF'] = df.apply(
        lambda row: fill_2ndFlrSF(row, meansf=df['2ndFlrSF'].mean()),
        axis=1
    )
    assert pd.isna(df['2ndFlrSF']).sum() == 0

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


    df.loc[:, 'BsmtFinSF1'] = df.apply(
        fill_BsmtFinSF1,
        axis=1
    )
    assert pd.isna(df['BsmtFinSF1']).sum() == 0

    # At this point, no NA objects should exist ('NA' as a string is OK)
    assert pd.isna(df).sum().sum() == 0

    return df


def scale_numerical(train, val):
    # Scale numerical values, use column list from validation dataset
    # as we don't want to include SalePrice in the scaling
    # Assuming that 32-bit precision is sufficient for target values of this magnitude...
    num_cols = val.select_dtypes(['number']).columns.tolist()
    for col in num_cols:
        # Create a new scaler object
        # scaler = StandardScaler()
        scaler = QuantileTransformer(random_state=42)

        # Extract array of values
        train_vals = train[col].values
        val_vals = val[col].values

        # Calculate no. samples in each array
        train_samples = len(train_vals)
        val_samples = len(val_vals)

        # Reshape for scaling
        train_vals = train_vals.reshape(train_samples, -1)
        val_vals = val_vals.reshape(val_samples, -1)

        # Perform scaling operation
        train_vals = scaler.fit_transform(train_vals)
        val_vals = scaler.transform(val_vals)

        # Flatten the output
        train_vals = train_vals.flatten()
        val_vals = val_vals.flatten()

        # Assign scaled values back to dataframe
        train.loc[:, col] = train_vals
        val.loc[:, col] = val_vals

    # Scale the copied SalePrice values
    # scaler = StandardScaler()
    scaler = QuantileTransformer(random_state=42)
    train_vals = train['SalePriceTemp'].values
    train_vals = train_vals.reshape(len(train_vals), -1)
    train_vals = scaler.fit_transform(train_vals).flatten()
    train.loc[:, 'SalePriceTemp'] = train_vals

    return train, val


def get_aggregates(train, val, metadata):
    # %% Derive some aggregate statistics on SalePrice across categories
    overall_min = train['SalePriceTemp'].min()
    overall_max = train['SalePriceTemp'].max()
    overall_mean = train['SalePriceTemp'].mean()

    cat_cols = [x for x in metadata if metadata[x]['type']=='category' and x in train.columns]

    # Only consider columns which are fully populated
    cat_cols = [x for x in cat_cols if 'NA' not in train[x].value_counts()]

    # No cheating, figures need to be aggregates not just the SalePrice for the given record
    cat_cols = [x for x in cat_cols if min(train[x].value_counts()) > 1]

    for cat_col in cat_cols:
        cat_stats = train.groupby(cat_col).agg(**{
            f"{cat_col}_min": ('SalePriceTemp', 'min'),
            f"{cat_col}_max": ('SalePriceTemp', 'max'),
            f"{cat_col}_mean": ('SalePriceTemp', 'mean')
        }).reset_index()
        train = pd.merge(left=train, right=cat_stats, on=cat_col, how='left')
        val = pd.merge(left=val, right=cat_stats, on=cat_col, how='left')

        val.loc[:, f"{cat_col}_min"] = val.loc[:, f"{cat_col}_min"].fillna(overall_min)
        val.loc[:, f"{cat_col}_max"] = val.loc[:, f"{cat_col}_max"].fillna(overall_max)
        val.loc[:, f"{cat_col}_mean"] = val.loc[:, f"{cat_col}_mean"].fillna(overall_mean)

    # Drop the copy of the SalePrice column
    train = train.drop(columns=['SalePriceTemp'])

    return train, val

def get_dummies(train, val):
    # Generate dummy variables for all categorical columns
    cat_cols = train.select_dtypes(['object']).columns.tolist()

    train_dummies = pd.get_dummies(train.loc[:, cat_cols], drop_first=True)
    val_dummies = pd.get_dummies(val.loc[:, cat_cols], drop_first=True)

    dummy_cols = set(train_dummies.columns).union(set(val_dummies.columns))
    dummy_cols = list(dummy_cols)
    train_dummies = train_dummies.reindex(columns=dummy_cols).fillna(0)
    val_dummies = val_dummies.reindex(columns=dummy_cols).fillna(0)

    train = train.drop(columns=cat_cols)
    val = val.drop(columns=cat_cols)

    train = pd.concat([train, train_dummies], axis=1)
    val = pd.concat([val, val_dummies], axis=1)

    return train, val


def derive_fields(train, val, metadata):

    train_inx = train.index.copy()
    val_inx = val.index.copy()

    # Take a copy of SalePrice for scaling, used to calculate aggregate statistics
    train.loc[:, 'SalePriceTemp'] = train['SalePrice'].copy(deep=True)

    # Scale all numerical values to mean 0 and stdev 1
    train, val = scale_numerical(train, val)

    # Generate aggregate statistics
    train, val = get_aggregates(train, val, metadata)

    # Create dummy variables for categorical fields
    train, val = get_dummies(train, val)

    train.index = train_inx
    val.index = val_inx

    return train, val

# %% At this point, no NA objects should exist (but 'NA' as a string is OK)
# assert pd.isna(train).sum().sum() == 0

# %%

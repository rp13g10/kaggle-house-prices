'''This file implements the recommendations made following the work done in eda.py
Initial validation of the data provided was performed in eda.py

Folder structure
- eda.py - General exploration of the dataset
- etl.py - Tidying the datset in preparation for modeling
- modeling.py - Evaluate performance of various models
- Summary of Results.ipynb - Primary script, coordinates execution of all functions'''

#%% Import modules
import pandas as pd
from sklearn.preprocessing import QuantileTransformer


def transform_data(df, metadata):
    '''Transform validated dataset into a state which can be used for modelling. No
    fields are derived at this point, but all null values are filled in and any
    columns deemed to be unhelpful are dropped. After this function has been called,
    the output dataframe will not contain any empty values.'''

    def find_cols(search):
        '''Convenience function to find metadata for columns containing a
        key word'''
        keys = [x for x in metadata if search.lower() in x.lower()]
        return {k: v for k, v in metadata.items() if k in keys}

    # Merge all porch columns into one
    porch_cols = list(find_cols('porch'))
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

    # Drop redundant columns
    df = df.drop(columns=[
        'PoolArea',     # Using PoolQC instead
        'BsmtFinSF2',   # Using BsmtFinType2 instead
        'SaleType',     # No clear way to fill the gaps
        'MasVnrArea',   # Using MasVnrType instead
        'MiscFeature',  # Almost exclusively sheds, using MiscVal instead
        'Street',       # Only 6 Grvl values, everything else is Pave
        'Utilities',    # Only 1 NoSeWa value, everything else is AllPub
    ])

    # Remove any 0s from numerical fields, deal with them using logic defined in eda.py
    num_cols = [x for x in metadata if metadata[x]['type'] == 'number' and x in df.columns]
    df.loc[:, num_cols] = df.loc[:, num_cols].replace(0, pd.NA)

    # Fill numerical fields with 0 (if it makes sense to do so)
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


    # Fill numerical fields with the median (if 0s don't make sense)
    # - The median is used as LotFrontage contains some outliers which distort the mean
    # - Might make more sense to fill GarageYrBlt with 0s and add a separate flag for
    #   missing values.
    avg_cols = [
        'LotFrontage',
        'GarageYrBlt'
    ]

    df.loc[:, avg_cols] = df.loc[:, avg_cols].apply(
        lambda col: col.fillna(col.median()),
        axis=0
    )
    assert pd.isna(df[avg_cols]).sum().sum() == 0

    # Fill categorical fields with the mode
    # - mode() returns a pandas series with a single value, accessing the
    #   .values[0] attribute extracts the final figure as a float
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


    # Convert fields with low variance to binary form
    # - Converts to a binary flag for the most common value
    low_var_cols = [
        'Condition2',
        'Heating'
    ]

    df.loc[:, low_var_cols] = df.loc[:, low_var_cols].apply(
        lambda col: (col == col.mode().values[0]).astype(int),
        axis=0
    )
    assert pd.isna(df[low_var_cols]).sum().sum() == 0


    # Convert alley to binary indicator showing if one is present
    df.loc[:, 'Alley'] = (df.loc[:, 'Alley'] != 'NA').astype(int)
    assert pd.isna(df['Alley']).sum() == 0


    # Fill 2ndFlrSF with mean where building type isn't a 1-storey structure
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

    # Conditionally fill in BsmtFinSF1
    def fill_BsmtFinSF1(row):
        finsf = row['BsmtFinSF1']
        fintype = row['BsmtFinType1']
        totsf = row['TotalBsmtSF']
        unfsf = row['BsmtUnfSF']
        gfsf = row['1stFlrSF']

        # Retain original value if provided
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
    '''Scales all numerical values, using the column list from validation dataset
    as we don't want to include SalePrice in the scaling.

    See here for documentation on the scaling algorithm:
    https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.QuantileTransformer.html

    This is necessary to ensure that no single field ends up unduly impacting the output of
    the model simply because it contains the biggest numbers.
    '''

    # Columns which have only integer values from 0 to 10
    scale_cols = {'OverallQual', 'OverallCond'}

    # Get a list of all numerical columns
    num_cols = val.select_dtypes(['number']).columns.tolist()

    # Scale them one column at a time. QuantileTransformer selected as it is more robust
    # to outliers.
    for col in num_cols:
        # Create a new scaler object
        if col in scale_cols:
            scaler = QuantileTransformer(random_state=42, output_distribution='uniform')
        else:
            scaler = QuantileTransformer(random_state=42, output_distribution='normal')

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
    scaler = QuantileTransformer(random_state=42, output_distribution='normal')
    train_vals = train['SalePriceTemp'].values
    train_vals = train_vals.reshape(len(train_vals), -1)
    train_vals = scaler.fit_transform(train_vals).flatten()
    train.loc[:, 'SalePriceTemp'] = train_vals

    return train, val


def get_aggregates(train, val, metadata):
    '''While training a model on data that includes the target variable is generally
    a bad idea, it's too valuable a column to get rid of it entirely. This function
    calculates the min/max/mean sale price in each category, for each categorical
    field in the dataset.'''

    # Derive overall min/max/mean as a backup for categories which appear
    # in the test dataset but not the training dataset
    overall_min = train['SalePriceTemp'].min()
    overall_max = train['SalePriceTemp'].max()
    overall_mean = train['SalePriceTemp'].mean()

    # Get a list of all categorical columns
    cat_cols = [x for x in metadata if metadata[x]['type'] == 'category' and x in train.columns]

    # Only consider columns which are fully populated
    cat_cols = [x for x in cat_cols if 'NA' not in train[x].value_counts()]

    # No cheating, figures need to be aggregates not just the SalePrice for the given record
    cat_cols = [x for x in cat_cols if min(train[x].value_counts()) > 1]

    # Generate aggregate statistics one column at a time
    for cat_col in cat_cols:
        # Calculate required figures
        cat_stats = train.groupby(cat_col).agg(**{
            f"{cat_col}_min": ('SalePriceTemp', 'min'),
            f"{cat_col}_max": ('SalePriceTemp', 'max'),
            f"{cat_col}_mean": ('SalePriceTemp', 'mean')
        }).reset_index()

        # Merge back on to main datasets
        train = pd.merge(left=train, right=cat_stats, on=cat_col, how='left')
        val = pd.merge(left=val, right=cat_stats, on=cat_col, how='left')

        # Fill in any gaps with overall figures
        val.loc[:, f"{cat_col}_min"] = val.loc[:, f"{cat_col}_min"].fillna(overall_min)
        val.loc[:, f"{cat_col}_max"] = val.loc[:, f"{cat_col}_max"].fillna(overall_max)
        val.loc[:, f"{cat_col}_mean"] = val.loc[:, f"{cat_col}_mean"].fillna(overall_mean)

    # Drop the copy of the SalePrice column
    train = train.drop(columns=['SalePriceTemp'])

    return train, val

def get_dummies(train, val):
    '''Generate dummy variables for all categorical columns, dropping
    the originals.'''

    # Get a list of all categorical fields
    cat_cols = train.select_dtypes(['object']).columns.tolist()

    # Generate dummy variables for these fields
    train_dummies = pd.get_dummies(train.loc[:, cat_cols], drop_first=True)
    val_dummies = pd.get_dummies(val.loc[:, cat_cols], drop_first=True)

    # Get a complete list of all fields in the data
    dummy_cols = set(train_dummies.columns).union(set(val_dummies.columns))
    dummy_cols = list(dummy_cols)

    # Make sure both datasets have the same fields in the same order
    train_dummies = train_dummies.reindex(columns=dummy_cols).fillna(0)
    val_dummies = val_dummies.reindex(columns=dummy_cols).fillna(0)

    # Drop the original fields
    train = train.drop(columns=cat_cols)
    val = val.drop(columns=cat_cols)

    # Join dummy variables back on to the original datasets
    train = pd.concat([train, train_dummies], axis=1)
    val = pd.concat([val, val_dummies], axis=1)

    return train, val

def get_flags(train, val, metadata):
    '''For numerical fields which have 0 in place of empty values, generate a new field
    which shows whether or not the field is null. This should help the model to distinguish
    between genuine low values, and 'zeros' which are there in place of missing data'''

    # Get a list of numerical fields
    num_cols = [x for x in metadata if metadata[x]['type'] == 'number' and x in train.columns]
    scale_cols = {'OverallQual', 'OverallCond'}

    # Derive a new field for each numerical column
    for col in num_cols:
        # Unless it's a scale from 0-10
        if col in scale_cols:
            continue

        # 1 if the value is 0, 0 if it isn't
        train.loc[:, f'{col}_Flag'] = train[col] == 0
        val.loc[:, f'{col}_Flag'] = val[col] == 0

    return train, val


def derive_fields(train, val, metadata):
    '''Perform all of the ETL stages as defined in the functions above.'''

    # Take a copy of the index (the Id variable) as it gets reset during the
    # process of creating dummy variables (val started at 0 rather than 1461)
    train_inx = train.index.copy()
    val_inx = val.index.copy()

    # Take a copy of SalePrice for scaling, used to calculate aggregate statistics
    train.loc[:, 'SalePriceTemp'] = train['SalePrice'].copy(deep=True)

    # Create a separate flag column for numerical values which have been filled with 0
    train, val = get_flags(train, val, metadata)

    # Scale all numerical values to mean 0 and stdev 1
    train, val = scale_numerical(train, val)

    # Generate aggregate statistics
    train, val = get_aggregates(train, val, metadata)

    # Create dummy variables for categorical fields
    train, val = get_dummies(train, val)

    # Set the index of each dataframe back to its initial value
    train.index = train_inx
    val.index = val_inx

    return train, val

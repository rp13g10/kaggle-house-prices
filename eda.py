'''This file will be used to determine which fields are likely to be useful in making final
predictions using the provided dataset. After this has been completed, etl.py will parse the
data, modeling.py will be used to determine the optimal algorithm and generate_predictions.py
will 

Folder structure
- eda.py - General exploration of the dataset
- etl.py - Tidying the datset in preparation for modeling
- modeling.py - Evaluate performance of various models
- generate_predictions.py - Apply final model to generate predictions'''

#%% Import modules
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

####################################################################################################
# Load data                                                                                        #
####################################################################################################


#%% Load in metadata

def get_metadata():
    '''This will load in the provided metadata and parse the results into a more useable dictionary
    format. This will allow us to substitute human-readable categories back into the dataframe
    to assist with the eda process'''

    # Load in the data dictionary provided by Kaggle
    with open('data_description.txt', 'r') as f:
        raw_meta = f.readlines()

    def create_record(cur_key, cur_desc, cur_values):
        '''Create a dictionary containing all of the data for the current field,
        taking the form:

        {cur_key: {'description': cur_desc, 'values': cur_values', 'type': str}}
        
        Inputs:
        cur_key - The field name (str)
        cur_desc - Description of the field (str)
        cur_values - All possible values for the field (list)
        
        Outputs:
        record - A dictionary containing all of the inputs in a standard format (dict)'''

        # If there's no field name, don't return anything
        if not cur_key:
            return {}
        
        # Initialize a nested dictionary for the current field
        record = {cur_key: {}}

        # Store the field description & values
        record[cur_key]['description'] = cur_desc
        record[cur_key]['values'] = cur_values

        # Determine whether the field is numerical or categorical
        if not cur_values:
            record[cur_key]['type'] = 'number'
        elif all([x in cur_values for x in '1 2 3 4 5 6 7 8 9 10'.split()]):
            record[cur_key]['type'] = 'number'
        else:
            record[cur_key]['type'] = 'category'

        return record
            
    # Initialize empty variables
    metadata = {}
    cur_key = None
    cur_desc = None
    cur_values = {}

    # Read through the file one line at a time
    for line in raw_meta:
        # Skip empty lines
        if not line.strip():
            continue

        # Check if line is indented
        indented = line[0] == ' '

        # No indent - move on to the next field
        if not indented:
            # Write data from previous field to metadata
            metadata.update(create_record(cur_key, cur_desc, cur_values))

            # Get data for current field
            cur_key, cur_desc = line.split(': ')
            cur_desc = cur_desc.strip()
            cur_values = {}
        # Indented - will be an id/value pair
        else:
            id_, value = line.strip().split('\t')
            cur_values[id_] = value

    # Write final field to metadata
    metadata.update(create_record(cur_key, cur_desc, cur_values))
    
    return metadata


def validate_data(df, metadata):
    '''Ensure that every value in the dataframe is consistent with the schema defined in
    the table metadata'''

    # Prevent any changes from leaking out into the global scope unintentionally
    df = df.copy(deep=True)

    #%% Take a closer look at numerical columns
    num_cols = [x for x in metadata if metadata[x]['type']=='number']

    # Remove any non-numerical data from the numerical columns
    for num_col in num_cols:
        # df.loc[:, num_col] = df.loc[:, num_col].replace('NA', pd.NA)
        df.loc[:, num_col] = pd.to_numeric(df.loc[:, num_col], errors='coerce')

    # Note that many numerical columns contain 0s in place of nulls
    # Treat an area of 0 as a null
    area_cols = [
        x for x in num_cols
        if any(
            phrase in metadata[x]['description']
            for phrase in ['square foot', 'area', 'square feet']
        )
    ]

    df.loc[:, area_cols] = df.loc[:, area_cols].replace(0, pd.NA)

    # Make some substitutions where values don't match the metadata
    df.loc[:, 'Exterior2nd'] = df.loc[:, 'Exterior2nd'].replace({
        'Wd Shng': 'WdShing',
        'CmentBd': 'CemntBd',
        'Brk Cmn': 'BrkComm'
    })

    df.loc[:, 'MSZoning'] = df.loc[:, 'MSZoning'].replace({
        'C (all)': 'C'
    })

    df.loc[:, 'BldgType'] = df.loc[:, 'BldgType'].replace({
        'Duplex': 'Duplx',
        '2fmCon': '2FmCon',
        'Twnhs': 'TwnhsI'
    })

    df.loc[:, 'MasVnrType'] = df.loc[:, 'MasVnrType'].replace({
        'None': 'NA'
    })

    df.loc[:, 'SaleType'] = df.loc[:, 'SaleType'].replace({
        'NA': 'Oth'
    })

    for col in ['Electrical', 'MSZoning', 'Utilities',
                'Exterior1st', 'Exterior2nd', 'KitchenQual',
                'Functional']:
        df.loc[:, col] = df.loc[:, col].replace({
            'NA': pd.NA
        })

    # Validate that everything in the data should be there
    # Any unknown values will throw an error and will need to be
    # built in to the logic above
    cat_cols = [x for x in metadata if metadata[x]['type'] == 'category']
    for cat_col in cat_cols:
        meta_vals = set(metadata[cat_col]['values'])
        df_vals = set(df[cat_col].dropna().astype(str))
        diff = df_vals.difference(meta_vals)
        assert not diff, f"{cat_col}: {diff.__repr__()}"

    return df


def load_validated_data(fname):
    '''This function loads in the metadata table and uses it to ensure that all
    values brought in conform to the provided specification. Non-conforming values are simply
    dropped, to be dealt with at a later time'''

    #%% Bring in the data
    metadata = get_metadata()
    df = pd.read_csv(fname, index_col='Id', keep_default_na=False)

    #%% Check everything matches up
    meta_cols = set(metadata.keys())
    df_cols = set(df.columns)

    # Update metadata to reflect the dataframe
    # Field names don't match up in the original files
    metadata['BedroomAbvGr'] = metadata['Bedroom']
    metadata['KitchenAbvGr'] = metadata['Kitchen']
    del metadata['Bedroom']
    del metadata['Kitchen']

    # Check that the only difference is the column being predicted
    meta_cols = set(metadata.keys())
    if fname == 'train.csv':
        assert df_cols.difference(meta_cols) == {'SalePrice'}
    else:
        # test.csv doesn't include the target column so columns should match
        assert not df_cols.difference(meta_cols)

    # Now it's safe to validate against the metadata
    df = validate_data(df, metadata)

    return df, metadata


if __name__ == '__main__':
    '''Leftover code from the initial EDA process. Mostly this has been moved across to the
    Jupyter notebook, but it's left here for reference.'''

    # Load data into the kernel
    train, metadata = load_validated_data('train.csv')

    def find_cols(search):
        '''Convenience function to find metadata for columns containing a
        key word'''
        keys = [x for x in metadata if search.lower() in x.lower()]
        return {k: v for k, v in metadata.items() if k in keys}

    # Get a list of all numerical fields
    num_cols = [x for x in metadata if metadata[x]['type']=='number']


    ####################################################################################################
    # Check for missing values                                                                         #
    ####################################################################################################

    #%% Check field completion
    tot_records = len(train.index)
    counts = train.count(axis='index')
    counts /= tot_records
    counts = counts.sort_values(ascending=True)
    counts[counts < 1]

    # PoolArea (0.005) - Drop in favour of PoolQC which is 100% populated
    # Porch columns will be merged, remaining NAs to be filled with 0s
    # LowQualFinSF (0.018) - Fill blanks with 0
    # Alley looks acceptable (0.06), but convert non-nas to boolean
    # Drop BsmtFinSF2 (0.11) in favour of BsmtFinType2
    # Drop SaleType (0.13), can't see any reliable way to fill in the gaps
    # Drop MasVnrArea (0.4) in favour of MasVnrType
    # Fill 2ndFlrSF (0.43) with mean where building type isn't a 1-storey variant
    # WoodDeckSF (0.47) - Fill blanks with 0
    # BsmtFinSF1 (0.68) - Fill blanks with 0 if BsmtFinType1 is 'NA' else use TotalBsmtSF - BsmtUnfSF
    # LotFrontage (0.82) - Fill blanks with mean
    # Neighborhood (0.84) - Fill blanks with mode
    # BldgType (0.91) - Fill blanks with mode
    # BsmtUnfSF (0.92) - Fill blanks with 0
    # Exterior2nd (0.93) - Leave NAs in place
    # GarageArea (0.94) - Fill blanks with 0
    # GarageYrBlt (0.04) - Fill blanks with mean
    # TotalBsmtSF (0.97) - Fill blanks with 0
    # MSZoning (0.99) - Fill blanks with mode
    # MasVnrType (0.99) - Fill blanks with mode
    # Electrical (0.99) - Fill blanks with mode


    #%% Check for outliers/variance
    for col in train.columns:
        print(col)
        fig = train[col].hist()
        plt.show()
        input('Press enter to continue')

    # Possible anomalies in LotArea
    # Low variance in Street, Utilities, Condition2, Heating, 
    # Very few BsmtHalfBaht entries, merge with BsmtFullBath
    # Drop MiscFeature in favour of MiscVal, almost all are sheds



    #%% Check for covariance
    plt.figure(figsize=(18,16))
    scaled = train[num_cols] / train[num_cols].std()
    cov = scaled.cov()
    fig = sns.heatmap(cov, vmin=0, vmax=1)
    plt.show()

    covs = {}
    for row in num_cols:
        for col in num_cols:
            if row != col and (col, row) not in covs:
                covs[(row, col)] = cov.loc[row, col]

    covs = [(x, covs[x]) for x in sorted(covs, key=lambda x: covs[x])]

    # Based on tail of covs, could swap TotalBsmtSF to BsmtPresent


    # %%

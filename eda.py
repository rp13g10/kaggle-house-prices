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

    with open('data_description.txt', 'r') as f:
        raw_meta = f.readlines()

    def create_record(cur_key, cur_desc, cur_values):
        if not cur_key:
            return {}
        record = {cur_key: {}}
        record[cur_key]['description'] = cur_desc
        record[cur_key]['values'] = cur_values
        if not cur_values:
            record[cur_key]['type'] = 'number'
        elif all([x in cur_values for x in '1 2 3 4 5 6 7 8 9 10'.split()]):
            record[cur_key]['type'] = 'number'
        else:
            record[cur_key]['type'] = 'category'
        return record
            

    metadata = {}
    cur_key = None
    cur_desc = None
    cur_values = {}
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
        # indented - will be an id/value pair
        else:
            id_, value = line.strip().split('\t')
            cur_values[id_] = value

    # Write final field to metadata
    metadata.update(create_record(cur_key, cur_desc, cur_values))
    
    return metadata


#%% Bring in the data
metadata = get_metadata()
train = pd.read_csv('train.csv', index_col='Id', keep_default_na=False)

#%% Check everything matches up
meta_cols = set(metadata.keys())
df_cols = set(train.columns)

# Not a 100% match just yet
assert len(meta_cols) == 79
assert len(df_cols) == 80
assert len(meta_cols.intersection(df_cols)) == 77
assert meta_cols - df_cols == {'Bedroom', 'Kitchen'}
assert df_cols - meta_cols == {'BedroomAbvGr', 'KitchenAbvGr', 'SalePrice'}

# Update metadata to reflect the dataframe
metadata['BedroomAbvGr'] = metadata['Bedroom']
metadata['KitchenAbvGr'] = metadata['Kitchen']
del metadata['Bedroom']
del metadata['Kitchen']

# Check that the only difference is the column being predicted
meta_cols = set(metadata.keys())
assert df_cols.difference(meta_cols) == {'SalePrice'}



####################################################################################################
# Check for missing values                                                                         #
####################################################################################################

#%% Take a closer look at numerical columns
num_cols = [x for x in metadata if metadata[x]['type']=='number']
train[num_cols]


#%% Validate all data ##############################################################################

# Remove any non-numerical data from the numerical columns
for num_col in num_cols:
    train.loc[:, num_col] = pd.to_numeric(train.loc[:, num_col], errors='coerce')

# Note that many numerical columns contain 0s in place of nulls
# Treat an area of 0 as a null
area_cols = [x for x in num_cols
             if any(
                 phrase in metadata[x]['description']
                 for phrase in ['square foot', 'area', 'square feet'])
            ]

train.loc[:, area_cols] = train.loc[:, area_cols].replace(0, pd.NA)

# Remove any categorical values which aren't specified in the metadata
cat_cols = [x for x in metadata if metadata[x]['type'] == 'category']

for cat_col in cat_cols:
    cat_vals = metadata[cat_col]['values']
    train.loc[:, cat_col] = train[cat_col].where(train[cat_col].isin(cat_vals))

#%% Check field completion
tot_records = len(train.index)
counts = train.count(axis='index')
counts /= tot_records
counts = counts.sort_values(ascending=True)
counts.head(20)

def find_cols(search):
    '''Convenience function to find metadata for columns containing a
    key word'''
    keys = [x for x in metadata if search.lower() in x.lower()]
    return {k: v for k, v in metadata.items() if k in keys}

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


# Lots of porch columns with limited data, merge them
porch_cols = find_cols('porch').keys()


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

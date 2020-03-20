from eda import load_validated_data
from etl import transform_data, derive_fields

train, metadata = load_validated_data('train.csv')
test, _ = load_validated_data('test.csv')

train = transform_data(train, metadata)
test = transform_data(test, metadata)

train, test = derive_fields(train, test, metadata)
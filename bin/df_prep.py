import logging

import keras
import numpy
import pandas
from keras.layers import Embedding, Flatten, Concatenate, Dense
from sklearn.preprocessing import Imputer, StandardScaler, LabelEncoder
from sklearn_pandas import DataFrameMapper


def create_mapper(df, cat_vars=list(), cont_vars=list(), date_vars=list(), no_transform_vars=list(), response_vars=list()):
    logging.info('Creating mapper')

    # TODO Add support for datetime variables

    # Reference variables
    transformation_list = list()

    # Copy df, to avoid 'inplace' transformation
    df = df.copy(deep=True)

    # TODO Check if any df variables are not listed in cat_vars or cont_vars. If so, remove them.

    # Check if any df variables are listed in cat_vars and cont_vars. If so, raise an error.
    intersection = filter(lambda x: x in cat_vars, cont_vars)
    if len(intersection) > 0:
        raise AssertionError('Columns appear in both cat_vars and cont_vars: {}'.format(intersection))

    # Convert continuous variables to float32
    for cont_var in cont_vars + response_vars:
        logging.debug('Converting cont_var data type: {}'.format(cont_var))
        df[cont_var] = df[cont_var].astype(numpy.float32)

    for date_var in date_vars:
        logging.info('Enriching for datetime var: {}'.format(date_var))
        df, date_cat_vars, date_cont_vars = add_datetime_vars(df, date_var)
        cat_vars.extend(date_cat_vars)
        cont_vars.extend(date_cont_vars)

    # Add continuous variable transformations for cont_vars
    for cont_var in cont_vars + response_vars:
        logging.debug('Creating transformation list for cont_var: {}'.format(cont_var))
        transformations = [Imputer(strategy='mean'), StandardScaler()]
        var_tuple = ([cont_var], transformations)
        transformation_list.append(var_tuple)

    # Add categorical variable transformations for cat_vars
    for cat_var in cat_vars:
        logging.debug('Creating transformation list for cat_var: {}'.format(cat_var))
        # TODO Replace LabelEncoder with CategoricalEncoder, to better handle unseen cases
        transformations = [LabelEncoder()]
        var_tuple = (cat_var, transformations)
        transformation_list.append(var_tuple)

    for no_transform_var in no_transform_vars:
        logging.debug('Creating transformation list for cont_var: {}'.format(no_transform_var))
        transformations = [Imputer(strategy='most_frequent')]
        var_tuple = ([no_transform_var], transformations)
        transformation_list.append(var_tuple)

    # Create mapper
    logging.info('Creating mapper')
    mapper = DataFrameMapper(features=transformation_list, df_out=True)

    # Train mapper
    logging.info('Training newly created mapper')
    mapper.fit(df)

    # Throw away transformation, to set up mapper
    logging.info('Transforming data set with newly created mapper, to initialize mapper internals')
    mapper.transform(df.sample(1000))

    return mapper


def create_model_layers(df, cat_vars, cont_vars, response_var, date_vars, no_transform_vars, mapper):



    # Reference variables
    Xs = list()
    x_labels = list()
    x_inputs = list()
    x_nub_last_layers = list()

    # Create X inputs for categorical variables
    for cat_var in cat_vars:

        # Add current variable to list of variables to process
        x_labels.append(cat_var)

        # Pull transformed data
        transformed = df[cat_var].as_matrix()

        # TODO Datatype converstion

        # Add transformed data to inputs
        Xs.append(transformed)

        # Set up dimensions for input and embedding layers
        if len(transformed.shape) >= 2:
            embedding_input_length = int(transformed.shape[1])
        else:
            embedding_input_length = 1
        embedding_input_dim = int(max(transformed)) + 1
        embedding_output_dim = int(min((embedding_input_dim + 1) / 2, 50))
        logging.info('Creating embedding for cat_var: {}, withemb edding_input_length: {}, embedding_input_dim: {}, '
                     'embedding_output_dim: {}'.format(cat_var, embedding_input_length, embedding_input_dim,
                                                       embedding_output_dim))

        # Create input and embedding layer
        sequence_input = keras.Input(shape=(embedding_input_length,), dtype='int32')

        embedding_layer = Embedding(input_dim=embedding_input_dim,
                                    output_dim=embedding_output_dim,
                                    input_length=embedding_input_length,
                                    trainable=True)

        embedded_sequences = embedding_layer(sequence_input)
        x = Flatten()(embedded_sequences)


        # Add input to inputs
        x_inputs.append(sequence_input)

        # Add last layer to layers
        x_nub_last_layers.append(x)

    # Create X inputs for continuous variables
    for cont_var in cont_vars:

        # Add current variable to list of variables
        x_labels.append(cont_var)

        # Pull transformed data
        transformed = df[cont_var].as_matrix()

        # TODO Datatype conversion

        # Add transformed data to inputs
        Xs.append(transformed)

        # Set up dimensions for input layer
        if len(transformed.shape) >= 2:
            embedding_input_length = int(transformed.shape[1])
        else:
            embedding_input_length = 1

        # Create input layer
        sequence_input = keras.Input(shape=(embedding_input_length,), dtype='float32')

        # Add input to inputs
        x_inputs.append(sequence_input)

        # Add last layer to layers
        x_nub_last_layers.append(sequence_input)

    # Concatenate all inputs
    input_nub = Concatenate()(x_nub_last_layers)

    # Create y matrix
    # TODO Data type conversion
    if response_var is not None:
        y = df[response_var].values

        # Create output nub

        output_shape = 1
        output_nub = Dense(units=output_shape, kernel_initializer='normal')
    else:
        y = None
        output_nub = None
    return Xs, y, x_inputs, input_nub, output_nub

def add_datetime_vars(df, date_var):
    cat_vars = list()
    cont_vars = list()

    # Create datetime related variables

    df[date_var] = pandas.to_datetime(df[date_var])
    date_cat_vars = ['dayofweek']
    date_cont_vars = []

    for date_new_var in date_cat_vars:

        local_var = '_'.join([date_var, date_new_var])
        logging.debug('Creating datetime sub var: {}'.format(local_var))
        df[local_var] = df[date_var].apply(lambda x: getattr(x, date_new_var, None))
        cat_vars.append(local_var)

    for date_new_var in date_cont_vars:
        local_var = '_'.join([date_var, date_new_var])
        logging.debug('Creating datetime sub var: {}'.format(local_var))
        df[local_var] = df[date_var].apply(lambda x: getattr(x, date_new_var, None))
        cont_vars.append(local_var)

    # Convert to epoch
    epoch_var = date_var + '_epoch'
    df[epoch_var] = df[date_var].astype(numpy.int64) // 10 ** 9
    cont_vars.append(epoch_var)
    return df, cat_vars, cont_vars
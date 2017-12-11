import logging

import keras
import numpy
from keras.layers import Embedding, Flatten, Concatenate, Dense
from sklearn.preprocessing import Imputer, StandardScaler, LabelEncoder
from sklearn_pandas import DataFrameMapper


def create_mapper(df, cat_vars, cont_vars, mapper=None):

    # TODO Add support for datetime variables

    # Copy df, to avoid 'inplace' transformation
    df = df.copy(deep=True)

    # TODO Check if any df variables are not listed in cat_vars or cont_vars. If so, remove them.

    # Check if any df variables are listed in cat_vars and cont_vars. If so, raise an error.
    intersection = filter(lambda x: x in cat_vars, cont_vars)
    if len(intersection) > 0:
        raise AssertionError('Columns appear in both cat_vars and cont_vars: {}'.format(intersection))

    # Convert continuous variables to float32
    for cont_var in cont_vars:
        df[cont_var] = df[cont_var].astype(numpy.float32)

    # If no mapper provided, create, populate and train mapper,
    if mapper is None:
        transformation_list = list()

        # Add continuous variable transformations for cont_vars
        for cont_var in cont_vars:
            transformations = [Imputer(strategy='mean'), StandardScaler()]
            var_tuple = ([cont_var], transformations)
            transformation_list.append(var_tuple)

        # Add categorical variable transformations for cat_vars
        for cat_var in cat_vars:

            # TODO Replace LabelEncoder with CategoricalEncoder, to better handle unseen cases
            transformations = [LabelEncoder()]
            var_tuple = (cat_var, transformations)
            transformation_list.append(var_tuple)

        # Create mapper
        mapper = DataFrameMapper(features=transformation_list, df_out=True)

        # Train mapper
        mapper.fit(df)

        # Throw away transformation, to set up mapper
        mapper.transform(df)

    return mapper


def create_model_layers(df, cat_vars, cont_vars, response_var, mapper):

    # Reference variables
    Xs = list()
    x_labels = list()
    x_inputs = list()
    x_layers = list()

    mapper_transformed = mapper.transform(df)

    # Create X inputs for categorical variables
    for cat_var in cat_vars:

        # Add current variable to list of variables to process
        x_labels.append(cat_var)

        # Pull transformed data
        transformed = mapper_transformed[cat_var].as_matrix()

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
        logging.info('Creating embedding for cat_var: {}, withembedding_input_length: {}, embedding_input_dim: {}, '
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
        x_layers.append(x)

    # Create X inputs for continuous variables
    for cont_var in cont_vars:

        # Add current variable to list of variables
        x_labels.append(cont_var)

        # Pull transformed data
        transformed = mapper_transformed[cont_var].as_matrix()

        # TODO Datatype conversion

        # Add transformed data to inputs
        Xs.append(transformed)

        # Set up dimensions for input layer
        if len(transformed.shape) >= 2:
            embedding_input_length = int(transformed.shape[1])
        else:
            embedding_input_length = 1

        # Create input layer
        sequence_input = keras.Input(shape=(embedding_input_length,), dtype='int32')

        # Add input to inputs
        x_inputs.append(sequence_input)

        # Add last layer to layers
        x_layers.append(x)

    # Concatenate all inputs
    input_nub = Concatenate()(x_layers)

    # Create y matrix
    # TODO Data type conversion
    if response_var is not None:
        y = mapper_transformed[response_var].values

        # Create output nub
        if len(y.shape) >= 2:
            output_shape = int(transformed.shape[1])
        else:
            output_shape = 1
        output_nub = Dense(units=output_shape, kernel_initializer='normal')
    else:
        y = None
        output_nub = None
    return Xs, y, x_inputs, input_nub, output_nub

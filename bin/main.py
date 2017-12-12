#!/usr/bin/env python
"""
coding=utf-8

Code Template

"""
import logging

import pandas
from keras import Model, optimizers

import lib
import df_prep


def main():
    """
    Main function documentation template
    :return: None
    :rtype: None
    """
    logging.basicConfig(level=logging.DEBUG)

    observations = extract()
    observations, mapper = transform(observations)
    model(observations, mapper)
    load()
    pass


def extract():
    logging.info('Begin extract')

    reservations = pandas.read_csv('../data/input/air_reserve.csv')
    visits = pandas.read_csv('../data/input/air_visit_data.csv')

    observations = pandas.merge(reservations, visits)
    observations = observations.head(10000)
    lib.archive_dataset_schemas('extract', locals(), globals())
    logging.info('End extract')
    return observations


def transform(observations):
    logging.info('Begin transform')

    cat_vars = []
    cont_vars = ['reserve_visitors', 'visitors']
    date_vars = ['visit_datetime', 'reserve_datetime']

    # Convert datetime vars
    for date_var in date_vars:
        logging.info('Converting date_var: {}'.format(date_var))
        observations[date_var] = pandas.to_datetime(observations[date_var], format='%Y-%m-%d %H:%M:%S')

    mapper = df_prep.create_mapper(observations, cat_vars=cat_vars, cont_vars=cont_vars, date_vars=date_vars)

    lib.archive_dataset_schemas('transform', locals(), globals())
    logging.info('End transform')
    return observations, mapper


def model(observations, mapper):
    logging.info('Begin model')

    cat_vars = []
    cont_vars = ['reserve_visitors']
    date_vars = ['visit_datetime', 'reserve_datetime']
    response_var = 'visitors'

    Xs, y, x_inputs, input_nub, output_nub = df_prep.create_model_layers(observations, mapper, cat_vars, cont_vars,
                                                                         date_vars, response_var)

    # Create model
    x = input_nub
    preds = output_nub(x)

    regression_model = Model(x_inputs, preds)
    opt = optimizers.Adam()
    regression_model.compile(loss=lib.root_mean_squared_log_error,
                             optimizer=opt)

    regression_model.fit(Xs, y, batch_size=2 ** 12, validation_split=.2)

    lib.archive_dataset_schemas('model', locals(), globals())
    logging.info('End model')
    pass


def load():
    logging.info('Begin load')

    lib.archive_dataset_schemas('load', locals(), globals())
    logging.info('End load')
    pass


# Main section
if __name__ == '__main__':
    main()

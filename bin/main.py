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
    lib.archive_dataset_schemas('extract', locals(), globals())
    logging.info('End extract')
    return observations


def transform(observations):
    logging.info('Begin transform')

    cat_vars = ['air_store_id']
    cont_vars = ['reserve_visitors', 'visitors']
    date_vars = ['visit_datetime', 'reserve_datetime', 'visit_date']
    mapper = df_prep.create_mapper(observations, cat_vars=cat_vars, cont_vars=cont_vars)

    lib.archive_dataset_schemas('transform', locals(), globals())
    logging.info('End transform')
    return observations, mapper


def model(observations, mapper):
    logging.info('Begin model')

    cat_vars = ['air_store_id']
    cont_vars = ['reserve_visitors']
    date_vars = ['visit_datetime', 'reserve_datetime', 'visit_date']
    response_var = 'visitors'

    Xs, y, x_inputs, input_nub, output_nub = df_prep.create_model_layers(observations, cat_vars, cont_vars, response_var, mapper)

    # Create model
    x = input_nub
    preds = output_nub(x)

    regression_model = Model(x_inputs, preds)
    opt = optimizers.Adam()
    regression_model.compile(loss='mean_squared_error',
                             optimizer=opt)

    regression_model.fit(Xs, y)

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

import itertools
import numpy as np
import pandas as pd
import sys

from keras.models import load_model
from preprocess import read_and_preprocess_data
from datetime import datetime
from keras.models import Sequential, load_model
from keras.layers import Dense, Activation, Dropout
from keras.layers import LSTM
from keras import objectives
from keras import backend as K
from keras import optimizers
import keras.losses
import json


def dummy_weighted_loss(y_true, y_pred):
    """
    If we use a custom loss, keras needs that for inference as well. Here we don't weight, so we don't need the global
    variable, but we don't use the loss anyways, so it doesn't matter.
    :param y_true:
    :param y_pred:
    :return:
    """
    out = -(y_true * K.log(y_pred + 1e-5) + (1.0 - y_true) * K.log(1.0 - y_pred + 1e-5))
    return K.mean(out, axis=-1)


def create_model(params):
    """
    Creates an LSTM-based model given the parameters: 'lr', 'do', 'reg', 'lstm_units', 'n_outputs', 'n_features', 'use_weighted_loss'.
    :param params: model parameters as dict
    :return: Keras LSTM model
    """

    adam = optimizers.Adam(lr=params['lr'])

    model = Sequential()
    model.add(LSTM(params['lstm_units'], batch_input_shape=(params['batch_size'], None, params['n_features']), return_sequences=True, stateful=True))
    model.add(Dropout(params['dropout']))
    model.add(Dense(params['n_outputs'], activation='sigmoid'))
    model.compile(loss=dummy_weighted_loss if params['use_weighted_loss'] else 'binary_crossentropy', optimizer=adam)
    return model


def model_predict_future_activation(in_file, model, params, current_time, mean_occupancies):
    """
    Predict the next 24h for a given model and in_file (up to current_time) using the specified params
    :param in_file: input file
    :param model: model to predict with
    :param params: parameter
    :param current_time: time to read up to
    :return: Predicted future activations as numpy array of shape [24, params['n_outputs']]
    """

    current_time = datetime.strptime(current_time, "%Y-%m-%d %H:%M:%S")
    print(current_time)
    features, labels, device_list, _ = read_and_preprocess_data(in_file, current_time, batch_size=1)

    print("Features times: ", features[0, :, :2])
    print("Feature batch: ", features.shape)
    print("label batch: ", labels.shape)
    keras.losses.weighted_loss = dummy_weighted_loss

    predictions = np.squeeze(model.predict(features, batch_size=1))  # (n_timesteps, n_outputs)
    print(predictions.shape)

    all_predictions = []

    last_features = np.squeeze(features)[-1]
    last_predictions = tmp_prediction = predictions[-1]

    tmp_features = np.array(last_features)

    tmp_mean_occupancies = [mean_occupancies.loc[(tmp_features[0] * 24 + tmp_features[1], 'device_' + str(i + 1)),
                                                 'mean_occupancy'] for i in range(len(device_list))]

    print("Last features: ", last_features)
    tmp_features = np.concatenate([tmp_features[:2], last_predictions, tmp_mean_occupancies])
    for i in range(24):

        tmp_mean_occupancies = [mean_occupancies.loc[(tmp_features[0] * 24 + tmp_features[1], 'device_' + str(j + 1)),
                                                     'mean_occupancy'] for j in range(len(device_list))]

        tmp_features = np.concatenate([tmp_features[:2], tmp_prediction[0, 0], tmp_mean_occupancies])

        print(tmp_features)

        # Increment time features
        if tmp_features[1] == 23:
            tmp_features[0] = (tmp_features[0] + 1) % 7
        tmp_features[1] = (tmp_features[1] + 1) % 24

        tmp_prediction = model.predict(np.reshape(tmp_features, [1, 1, len(tmp_features)]))

        all_predictions += [tmp_prediction]

    return np.round(np.concatenate(all_predictions))


def predict(in_file, model, params, current_time=None, mean_occupancies=None):
    """
    Wraps the 24h prediction and returns it in a pandas dataframe
    :param in_file:
    :param model:
    :param params:
    :param current_time:
    :return: dataframe containing the predicted activations
    """
    if mean_occupancies is None:
        mean_occupancies = pd.read_pickle('mean_occupancy.pkl')

    if current_time is None:
        # Use last element of data as current time
        previous_readings = pd.read_csv(in_file)
        current_time = previous_readings['time'].iloc[-1]
    result = model_predict_future_activation(in_file, model, params, current_time, mean_occupancies)

    # Make 24 predictions for each hour starting at the next full hour
    next_24_hours = pd.date_range(current_time, periods=24, freq='H').ceil('H')

    # produce 24 hourly slots per device:
    xproduct = list(itertools.product(next_24_hours, params['devices']))
    predictions = pd.DataFrame(xproduct, columns=['time', 'device'])
    predictions.set_index(['time', 'device'], inplace=True)

    predictions['activation_predicted'] = np.ravel(result).astype(np.int32)
    print(predictions['activation_predicted'])

    return predictions


if __name__ == '__main__':

    current_time, in_file, out_file = sys.argv[1:]

    with open('params.json', 'r') as fp:
        params = json.load(fp)
        params['batch_size'] = 1

    model = create_model(params)
    model.load_weights('model.h5')


    predictions = predict(in_file, model, params, current_time)
    predictions.to_csv(out_file)


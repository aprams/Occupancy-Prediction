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
import keras.losses
import json


def dummy_weighted_loss(y_true, y_pred):
    out = -(y_true * K.log(y_pred + 1e-5) + (1.0 - y_true) * K.log(1.0 - y_pred + 1e-5))
    return K.mean(out, axis=-1)


def create_model(params):
    model = Sequential()
    model.add(LSTM(params['lstm_units'], batch_input_shape=(params['batch_size'], None, params['n_features']), return_sequences=True, stateful=True))
    model.add(Dropout(params['dropout']))
    model.add(Dense(params['n_outputs'], activation='sigmoid'))
    model.compile(loss=dummy_weighted_loss if params['use_weighted_loss'] else 'binary_crossentropy', optimizer=params['optimizer'])
    return model


def predict_next_24h(model, in_file):
    feature_batch, label_batch, device_list = read_and_preprocess_data(in_file, batch_size=1)
    print(feature_batch.shape)
    print(label_batch.shape)


def model_predict_future_activation(in_file, model, params, current_time):
    current_time = datetime.strptime(current_time, "%Y-%m-%d %H:%M:%S")
    features, labels, device_list = read_and_preprocess_data(in_file, current_time, batch_size=1)

    print("Feature batch: ", features.shape)
    print("label batch: ", labels.shape)
    keras.losses.weighted_loss = dummy_weighted_loss

    predictions = np.squeeze(model.predict(features, batch_size=1))  # (n_timesteps, n_outputs)
    print(predictions.shape)

    all_predictions = []

    last_features = np.squeeze(features)[-1]
    last_predictions = tmp_prediction = predictions[-1]

    tmp_features = np.array(last_features)
    tmp_features = np.concatenate([tmp_features[:2], last_predictions])
    for i in range(24):
        # print(tmp_prediction)
        tmp_prediction = model.predict(np.reshape(tmp_features, [1, 1, len(tmp_features)]))
        tmp_features = np.concatenate([tmp_features[:2], tmp_prediction[0, 0]])

        # Increment time features
        if tmp_features[1] == 23:
            tmp_features[0] = (tmp_features[0] + 1) % 7
        tmp_features[1] = (tmp_features[1] + 1) % 24
        all_predictions += [tmp_prediction]

    return np.round(np.concatenate(all_predictions))


def predict_future_activation(current_time, previous_readings):
    """
    This function predicts future hourly activation given previous sensings.
    """
    # make predictable
    np.random.seed(len(previous_readings))

    # Make 24 predictions for each hour starting at the next full hour
    next_24_hours = pd.date_range(current_time, periods=24, freq='H').ceil('H')

    device_names = sorted(previous_readings.device.unique())

    # produce 24 hourly slots per device:
    xproduct = list(itertools.product(next_24_hours, device_names))
    predictions = pd.DataFrame(xproduct, columns=['time', 'device'])
    predictions.set_index('time', inplace=True)

    # Random guess!
    predictions['activation_predicted'] = np.random.randint(2, size=len(predictions))
    return predictions


def predict(in_file, model, params, current_time=None):
    if current_time is None:
        # Use last element of data as current time
        previous_readings = pd.read_csv(in_file)
        current_time = previous_readings['time'].iloc[-1]
    result = model_predict_future_activation(in_file, model, params, current_time)

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


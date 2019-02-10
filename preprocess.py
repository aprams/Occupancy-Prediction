import itertools
import numpy as np
import pandas as pd
import sys


def preallocate_features(previous_readings):
    device_list = sorted(previous_readings.device.unique())

    first_time_stamp = pd.to_datetime(previous_readings['time'][0])

    hour_interval_start_end = pd.date_range(first_time_stamp.replace(minute=0, second=0),
                                            previous_readings['time'][len(previous_readings) - 1].replace(minute=0,
                                                                                                          second=0),
                                            freq='H')

    features = pd.DataFrame(0, index=np.arange(len(hour_interval_start_end)),
                            columns=['time', 'weekday', 'hour'] + device_list)
    features['time'] = hour_interval_start_end
    return features


def preprocess_features_and_labels(previous_readings):
    """
    Generate features so that we have an array with dimensions [T, n_devices]
    T: time
    n_devices: number of devices
    """

    device_list = sorted(previous_readings.device.unique())

    # We preallocate to avoid many appends (append copies according to pandas docs, might become an issue/slow for large data)
    features = preallocate_features(previous_readings)
    labels = pd.DataFrame(0, index=np.arange(len(features) - 1),
                          columns=device_list)

    for index, row in previous_readings.iterrows():
        if index == 0:
            continue

        dt = row['time'].replace(minute=0, second=0)
        feature_idx = features.index[features['time'] == dt]
        # Increment device's counter at time
        features.loc[feature_idx, row['device']] = 1
        labels.loc[feature_idx - 1, row['device']] = 1

    # Second loop, can we improve here?
    for index, row in features.iterrows():
        features.loc[index, 'weekday'] = row['time'].weekday()
        features.loc[index, 'hour'] = row['time'].hour

    features.drop('time', axis=1, inplace=True)

    return (features, labels)


def create_timeseries_batches(features, labels, length=20):
    np_features = features.to_numpy()[:-1]
    np_labels = labels.to_numpy()

    np_features = np_features[:length * (len(np_features) // length)]
    np_labels = np_labels[:length * (len(np_labels) // length)]

    feature_batch = np.reshape(np_features, [-1, length, np_features.shape[1]])
    label_batch = np.reshape(np_labels, [-1, length, np_labels.shape[1]])

    return feature_batch, label_batch


def read_and_preprocess_data(in_file, batch=True):
    previous_readings = pd.read_csv(in_file)
    previous_readings['time'] = pd.to_datetime(previous_readings['time'])

    features, labels = preprocess_features_and_labels(previous_readings)
    print("File {0} has {1} timesteps (hours)".format(in_file, labels.shape[0]))

    if batch:
        features, labels = create_timeseries_batches(features, labels)
    else:
        features = np.expand_dims(features.to_numpy()[:-1], 0)
        labels = np.expand_dims(labels.to_numpy(), 0)

    device_list = sorted(previous_readings.device.unique())

    return features, labels, device_list

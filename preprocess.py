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


def preprocess_features_and_labels(previous_readings, current_time=None):
    """
    Generate features so that we have an array with dimensions [T, n_devices]
    T: time
    n_devices: number of devices
    """

    device_list = sorted(previous_readings.device.unique())
    current_time = None
    if current_time is not None:
        n_elements_before_now = len(previous_readings[previous_readings['time'] < current_time])
        previous_readings_truncated = previous_readings[:n_elements_before_now]
    else:
        previous_readings_truncated = previous_readings

    # We preallocate to avoid many appends (append copies according to pandas docs, might become an issue/slow for large data)
    features = preallocate_features(previous_readings_truncated)
    labels = pd.DataFrame(0, index=np.arange(len(features) - 1),
                          columns=device_list)

    for index, row in previous_readings_truncated.iterrows():
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

    np_features = features.to_numpy()[:-1]
    np_labels = labels.to_numpy()

    return (np_features, np_labels)


def create_timeseries_batches(features, labels, sequence_length=100, sequence_start_shift=30,
                              n_sequences=16):
    """
    Creates n_sequences shifted "stateful" sequences with each sequence having sequence_length elements

    :param features:
    :param labels:
    :param sequence_length:
    :param sequence_start_shift:
    :param n_sequences:
    :return:
    """
    print("initial features shape: ", features.shape)

    full_sequence_length = sequence_length * ((len(features) - n_sequences * sequence_start_shift) // sequence_length)
    n_minibatches = full_sequence_length // sequence_length
    print("Full sequence length: ", full_sequence_length)
    # Trim initial sequence (alternative: padding)
    trimmed_length = sequence_length * (len(features) // sequence_length)
    trimmed_features = features[:trimmed_length]
    trimmed_labels = labels[:trimmed_length]
    features = trimmed_features
    labels = trimmed_labels

    sequences_features = []
    sequences_labels = []

    target_features_shape = [n_minibatches, sequence_length, features.shape[-1]]
    target_labels_shape = [n_minibatches, sequence_length, labels.shape[-1]]

    for i in range(n_sequences):
        start_idx = i * sequence_start_shift
        end_idx = trimmed_length - (n_sequences - i - 1) * sequence_start_shift
        print("Sequence {0} has start index {1} and end index {2}".format(i, start_idx, end_idx))

        print(np.array(features[start_idx: end_idx]).shape)
        sequences_features += [np.array(features[start_idx: end_idx]).reshape(target_features_shape)]
        sequences_labels += [np.array(labels[start_idx: end_idx]).reshape(target_labels_shape)]

    sequences_features = np.array(sequences_features)
    sequences_labels = np.array(sequences_labels)

    print("Features sequences shape: ", sequences_features.shape)
    print("Labels sequences shape: ", sequences_labels.shape)

    mini_batch_features_arr_shape = [n_minibatches * n_sequences, sequence_length, features.shape[-1]]
    mini_batch_features = np.zeros(mini_batch_features_arr_shape)
    mini_batch_labels_arr_shape = [n_minibatches * n_sequences, sequence_length, labels.shape[-1]]
    mini_batch_labels = np.zeros(mini_batch_labels_arr_shape)
    for i in range(n_minibatches):
        for j in range(n_sequences):
            mini_batch_features[i * n_sequences + j] = sequences_features[j, i]
            mini_batch_labels[i * n_sequences + j] = sequences_labels[j, i]


    #feature_batch = np.transpose(np.array(sequences_features),[1, 0, 2 ,3]).reshape([-1, target_features_shape[1], target_features_shape[2]])
    #label_batch = np.transpose(np.array(sequences_labels), [1, 0, 2, 3]).reshape([-1, target_labels_shape[1], target_labels_shape[2]])

    return mini_batch_features, mini_batch_labels


def read_and_preprocess_data(in_file, current_time=None, batch_size=32):
    previous_readings = pd.read_csv(in_file)

    previous_readings['time'] = pd.to_datetime(previous_readings['time'])


    features, labels = preprocess_features_and_labels(previous_readings, current_time)
    print("File {0} has {1} timesteps (hours) until {2}".format(in_file, labels.shape[0],
                                                                current_time if current_time is not None else "now"))

    if batch_size > 1:
        features, labels = create_timeseries_batches(features, labels, n_sequences=batch_size)
    else:
        features = np.expand_dims(features, 0)
        labels = np.expand_dims(labels, 0)

    device_list = sorted(previous_readings.device.unique())

    return features, labels, device_list

import itertools
import numpy as np
import pandas as pd
import sys
import datetime


def preallocate_features(previous_readings, device_list, current_time=None):
    """
    Preallocate a dataframe for the features, so we do not append the single rows
    :param previous_readings: input from file as dataframe
    :param device_list: list of device names
    :return: preallocated features data frame with columns 'time' (datetime), 'weekday' (0-7), 'hour' (0-23),
    'device_i' (for each device) (0 or 1), 'device_i_mean_occ' mean occupancy per device (0.0-1.0)
    """

    first_time_stamp = pd.to_datetime(previous_readings['time'][0])

    end_time = previous_readings['time'][len(previous_readings) - 1].replace(minute=0, second=0) if current_time is None \
        else current_time

    print("PREALLOCATE END TIME: ", current_time)

    hour_interval_start_end = pd.date_range(first_time_stamp.replace(minute=0, second=0), end_time, freq='H')


    print("PREALLOCATE hour_interval_start_end: ", hour_interval_start_end)

    features = pd.DataFrame(0, index=np.arange(len(hour_interval_start_end)),
                            columns=['time', 'weekday', 'hour'] + device_list + [x + "_mean_occ" for x in device_list])
    features['time'] = hour_interval_start_end
    return features


def preprocess_features_and_labels(previous_readings, end_time=None, device_list=None):
    """
    Generate features so that we have an array with dimensions [timesteps, n_devices]
    :param previous_readings: inputs from file
    :param end_time: up to which to read the inputs
    :param device_list: list of device names
    :return: - numpy array of features with columns: 'weekday' (0-7), 'hour' (0-23),
    'device_i' (for each device) (0 or 1), 'device_i_mean_occ' mean occupancy per device (0.0-1.0)
    - numpy array of labels
    - calculated mean occupancies to be reused for inference (to be fed as inputs)
    """

    device_list = sorted(previous_readings.device.unique()) if device_list is None else device_list
    if end_time is not None:
        n_elements_before_now = len(previous_readings[previous_readings['time'] < end_time])
        previous_readings_truncated = previous_readings[:n_elements_before_now]
    else:
        previous_readings_truncated = previous_readings

    # We preallocate to avoid many appends (append copies according to pandas docs, might become an issue/slow for large data)
    features = preallocate_features(previous_readings_truncated, device_list, current_time=end_time)
    labels = pd.DataFrame(0, index=np.arange(len(features) - 1),
                          columns=device_list)

    # Calculate mean occupancies per hour of the day per device as feature
    days_hour_range = pd.RangeIndex(start=0, stop=168)
    initial_occupancy = [0]
    hours_x_devices = list(itertools.product(days_hour_range, device_list, initial_occupancy))

    mean_occupancy_per_hour = pd.DataFrame(hours_x_devices, columns=['time', 'device', 'mean_occupancy'])
    mean_occupancy_per_hour.set_index(['time', 'device'], inplace=True)

    n_hours_in_data = len(features)

    print("Hours in data: ", n_hours_in_data)

    for index, row in previous_readings_truncated.iterrows():
        dt = row['time'].replace(minute=0, second=0)
        feature_idx = features.index[features['time'] == dt]
        # Set device's activation at time to 1
        features.loc[feature_idx, row['device']] = 1
        if feature_idx >= 2:
            labels.loc[feature_idx - 2, row['device']] = 1
        cur_hour_of_week = row['time'].weekday() * 24 + row['time'].hour
        mean_occupancy_per_hour.loc[(cur_hour_of_week, row['device']), 'mean_occupancy'] += 1 / (n_hours_in_data / 24)

    # Second loop, can we improve here?
    for index, row in features.iterrows():
        features.loc[index, 'weekday'] = row['time'].weekday()
        features.loc[index, 'hour'] = row['time'].hour
        cur_hour_of_week = row['time'].weekday() * 24 + row['time'].hour
        for i in range(len(device_list)):
            mean_occ_for_device = mean_occupancy_per_hour.loc[(cur_hour_of_week, 'device_' + str(i + 1)), 'mean_occupancy']
            features.loc[index, 'device_' + str(i + 1) + '_mean_occ'] = mean_occ_for_device

    features.drop('time', axis=1, inplace=True)

    np_features = features.to_numpy()[1:]
    np_labels = labels.to_numpy()
    print(np_features[:10])
    print(np_labels[:10])

    return np_features, np_labels, mean_occupancy_per_hour


def create_timeseries_batches(features, labels, sequence_length, sequence_start_shift=10,
                              n_sequences=16):
    """
    Creates n_sequences shifted "stateful" sequences with each sequence having sequence_length elements. Sequences need
    to be arranged precisely not to mess up with Keras batch training keeping states.
    :param features: numpy array of features
    :param labels: numpy array of features
    :param sequence_length: subsequence length per batch
    :param sequence_start_shift: shift per sequence
    :param n_sequences: number of sequences to generate
    :return: batched timeseries batches for stateful LSTM training with shape [n_minibatches (in code), sequence_length,
    features.shape[-1]]

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
        end_idx = start_idx + full_sequence_length#trimmed_length - (n_sequences - i - 1) * sequence_start_shift
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

    return mini_batch_features, mini_batch_labels


def read_and_preprocess_data(in_file, current_time=None, batch_size=32, sequence_start_shift=20, sequence_length=100,
                             device_list=None):
    """
    Reads in :param in_file: up to :param current_time:, generates the features (and batches if specified) and returns
    the preprocessed features and labels.
    :param in_file: Input file
    :param current_time: End time to read to
    :param batch_size: batch size for stateful sequence batches
    :param sequence_start_shift: Shift per batched sequence
    :param sequence_length: Length of each subsequence in the stateful batches
    :param device_list: list of device names
    :return: Preprocessed features, labels and mean occupancies
    """
    previous_readings = pd.read_csv(in_file)

    previous_readings['time'] = pd.to_datetime(previous_readings['time'])


    features, labels, mean_occupancies = preprocess_features_and_labels(previous_readings, end_time=current_time, device_list=device_list)
    print("File {0} has {1} timesteps (hours) until {2}".format(in_file, labels.shape[0],
                                                                current_time if current_time is not None else "now"))

    if batch_size > 1:
        features, labels = create_timeseries_batches(features, labels, n_sequences=batch_size,
                                                     sequence_start_shift=sequence_start_shift,
                                                     sequence_length=sequence_length)
    else:
        features = np.expand_dims(features, 0)
        labels = np.expand_dims(labels, 0)

    device_list = sorted(previous_readings.device.unique())

    return features, labels, device_list, mean_occupancies

import itertools
import numpy as np
import pandas as pd
import sys

from keras.models import load_model

def model_predict_future_activation():
    feature_batch, label_batch, device_list = read_and_preprocess_data(in_file)
    model = load_model('model.h5')

    print(test_feature_batch_expanded.shape)
    predictions = model.predict(test_feature_batch_expanded, batch_size=1)[:, 0, :]
    # print(np.round(predictions, 1))

    print(predictions.shape)
    print(test_label_batch_flattened.shape)


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


if __name__ == '__main__':

    current_time, in_file, out_file = sys.argv[1:]

    previous_readings = pd.read_csv(in_file)
    result = predict_future_activation(current_time, previous_readings)
    result.to_csv(out_file)

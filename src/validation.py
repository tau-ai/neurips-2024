import tensorflow as tf
from tensorflow.python.client import device_lib
import keras
from keras.models import Sequential, load_model
from keras.layers import ConvLSTM2D, LayerNormalization, BatchNormalization, Conv3D, Conv2D, TimeDistributed, Conv3DTranspose, Reshape, Input
import numpy as np
from datetime import datetime, timedelta
import os
import pandas as pd
import numpy as np

# Load datasets #
FLDAS_data_tensor = np.load('/mnt/disks/data/FLDAS/dataset.npy', allow_pickle=True)
FLDAS_times_tensor = np.load('/mnt/disks/data/FLDAS/times.npy', allow_pickle=True)
FEWS_data_tensor = np.load('/mnt/disks/data/FEWS/food_security_index_tensor.npy', allow_pickle=True)
FEWS_times_tensor = np.load('/mnt/disks/data/FEWS/food_security_index_times.npy', allow_pickle=True)

print(FLDAS_data_tensor.shape, FLDAS_times_tensor.shape, FEWS_data_tensor.shape, FEWS_times_tensor.shape)

# Normalize the datasets #

FEWS_data_tensor = np.nan_to_num(FEWS_data_tensor, copy=False)
FEWS_data_tensor_max = np.nanmax(FEWS_data_tensor)
FEWS_data_tensor_min = np.nanmin(FEWS_data_tensor)
print(f"FEWS data tensor max: {FEWS_data_tensor_max}, FEWS data tensor min: {FEWS_data_tensor_min}")
if FEWS_data_tensor_max > FEWS_data_tensor_min:
    FEWS_data_tensor = (FEWS_data_tensor - FEWS_data_tensor_min) / (FEWS_data_tensor_max - FEWS_data_tensor_min)
else:
    FEWS_data_tensor = np.zeros_like(FEWS_data_tensor)
FEWS_data_tensor_max = np.max(FEWS_data_tensor)
FEWS_data_tensor_min = np.min(FEWS_data_tensor)
print(f"Normalised FEWS data tensor max: {FEWS_data_tensor_max}, FEWS data tensor min: {FEWS_data_tensor_min}")

FLDAS_data_tensor = np.nan_to_num(FLDAS_data_tensor, copy=False)
FLDAS_data_tensor_max = np.nanmax(FLDAS_data_tensor)
FLDAS_data_tensor_min = np.nanmin(FLDAS_data_tensor)
print(f"FLDAS data tensor max: {FLDAS_data_tensor_max}, FEWS data tensor min: {FLDAS_data_tensor_min}")
def normalize_FLDAS_tensor(tensor):
    assert tensor.shape[-1] == 28, "The last dimension of the tensor must be 28."

    normalized_tensor = np.empty_like(tensor)

    for i in range(28):
        channel = tensor[..., i]

        channel_min = np.min(channel)
        channel_max = np.max(channel)

        normalized_channel = (channel - channel_min) / (channel_max - channel_min)

        normalized_tensor[..., i] = normalized_channel

    return normalized_tensor

FLDAS_data_tensor = normalize_FLDAS_tensor(FLDAS_data_tensor)
FLDAS_data_tensor_max = np.max(FLDAS_data_tensor)
FLDAS_data_tensor_min = np.min(FLDAS_data_tensor)
print(f"Normalised FLDAS data tensor max: {FLDAS_data_tensor_max}, FEWS data tensor min: {FLDAS_data_tensor_min}")

print("Normalized tensors' shapes:", FLDAS_data_tensor.shape, FEWS_data_tensor.shape)

print(FLDAS_data_tensor_max, FLDAS_data_tensor_min, FEWS_data_tensor_max, FEWS_data_tensor_min)

#Temporal alignment #
# FLDAS_times_tensor = FLDAS_times_tensor[:166]
# FLDAS_data_tensor = FLDAS_data_tensor[:166]

FLDAS_times_tensor = FLDAS_times_tensor[:166]
FLDAS_data_tensor = FLDAS_data_tensor[:166]

fldas_dates = [pd.to_datetime(t[0]).to_pydatetime() for t in FLDAS_times_tensor]
fews_dates = [pd.to_datetime(t).to_pydatetime() for t in FEWS_times_tensor]
extrapolated_fews_tensor = np.zeros((FLDAS_data_tensor.shape[0], 256, 411))
layers = []
for i in range(extrapolated_fews_tensor.shape[0]):
    blank_matrix = extrapolated_fews_tensor[i]
    associated_time = fldas_dates[i]
    latest_date = None
    for date in fews_dates:
        if date <= associated_time:
            latest_date = date
        else:
            if date == datetime(2009, 7, 1) and associated_time < datetime(2009, 7, 1):
                latest_date = date
            else:
                continue
    if latest_date is None:
        raise ValueError("No associated date found for the given FLDAS date.")
    print(i, associated_time, latest_date)
    layers.append(FEWS_data_tensor[fews_dates.index(latest_date)])

extrapolated_fews_tensor = np.stack(layers)
FEWS_data_tensor = extrapolated_fews_tensor
print(f"Extrapolated FEWS tensor shape: {FEWS_data_tensor.shape} FLDAS tensor shape: {FLDAS_data_tensor.shape}")
FEWS_data_tensor = np.expand_dims(FEWS_data_tensor, axis=-1)
completed_data_tensor = np.concatenate((FLDAS_data_tensor, FEWS_data_tensor), axis=-1)
print(f"Combined data tensor shape: {completed_data_tensor.shape}")

completed_data_tensor = np.nan_to_num(completed_data_tensor, copy=False)
if np.isnan(completed_data_tensor).any():
    print("There are still nan values in combined data tensor.")
else:
    print("No nan values in combined data tensor.")

np.save('/mnt/disks/data/complete_data_tensor.npy', completed_data_tensor)

##########################################################################

CHUNK_SIZE = 12
BATCH_SIZE = 8

completed_data_tensor = np.load('/mnt/disks/data/complete_data_tensor.npy')

def safe_mse_single_channel(y_true, y_pred): #The name is an artifact of a previous version of the safe_mse function. cba to refactor
    channel_weights = tf.constant([1.0] * 28 + [8.0], dtype=tf.float32)
    channel_weights = tf.convert_to_tensor(channel_weights, dtype=tf.float32)
    squared_difference = tf.square(y_true - y_pred)
    weights_shape = [1] * len(squared_difference.shape)
    weights_shape[-1] = channel_weights.shape[0]
    expanded_weights = tf.reshape(channel_weights, weights_shape)
    weighted_squared_difference = squared_difference * expanded_weights
    weighted_mse = tf.reduce_mean(weighted_squared_difference, axis=[-2, -3, -4])
    epsilon = 1e-7
    safe_mse = tf.where(tf.math.is_finite(weighted_mse), weighted_mse, epsilon)
    final_mse = tf.reduce_mean(safe_mse)
    return final_mse

# Initialize TPU
# tpu_devices = tf.config.list_logical_devices('TPU')
# tpu_names = [device.name for device in tpu_devices]
# print("TPUs available in TensorFlow:", tpu_names)

# try:
#     resolver = tf.distribute.cluster_resolver.TPUClusterResolver(tpu='local')
#     tf.tpu.experimental.initialize_tpu_system(resolver)
#     strategy = tf.distribute.TPUStrategy(resolver)
#     print("TPU initialized successfully")
# except Exception as e:
#     print(f"Error initializing TPU: {e}")
#     raise RuntimeError("Fatal error occurred. Halting the process. (TPU initialization)")
#NOTE: TPU inference was throwing errors, but due to the small number of paramaters, it was decided to run the inference on the CPU instead since it ran in 5-10mins and backing up the predictions just made more sense
# with strategy.scope():
model_path = '/home/duckb/neurips/src/saved_models/Final_pretrained.keras'
model = load_model(model_path, custom_objects={'safe_mse_single_channel': safe_mse_single_channel})

test_data_start_index = 120  # 12 months before January 2020
test_data_end_index = 165  # Up to the end of 2022
test_data = completed_data_tensor[test_data_start_index:test_data_end_index]

input_data = []
for i in range(test_data.shape[0] - CHUNK_SIZE):
    input_chunk = test_data[i:i + CHUNK_SIZE]
    if input_chunk.shape[0] == CHUNK_SIZE:
        input_data.append(input_chunk)

input_data = np.array(input_data)

np.save('/mnt/disks/data/input_data.npy', input_data)

if input_data.shape[0] == 0:
    raise ValueError("Input data is empty. Please check the data preparation steps.")

num_samples = input_data.shape[0]
num_padding = (BATCH_SIZE - (num_samples % BATCH_SIZE)) % BATCH_SIZE
if num_padding > 0:
    padding_shape = ((num_padding,) + input_data.shape[1:])
    input_data_padded = np.concatenate([input_data, np.zeros(padding_shape)], axis=0)
else:
    input_data_padded = input_data

num_samples = input_data.shape[0]
num_padding = 64 - num_samples
if num_padding > 0:
    padding_shape = ((num_padding,) + input_data.shape[1:])
    input_data_padded = np.concatenate([input_data, np.zeros(padding_shape)], axis=0)
else:
    input_data_padded = input_data

print("input_data.shape:", input_data.shape)  # Should be (num_samples, CHUNK_SIZE, 256, 411, 29)

# Generate predictions using the TPU-accelerated model
# with strategy.scope():
    # Check if input_data_padded is not empty
if input_data.shape[0] > 0:
    predictions_padded = model.predict(input_data, batch_size=BATCH_SIZE)
else:
    raise ValueError("Padded input data is empty. This should not happen after padding.")

# Trim the Predictions to remove padding ('Tis a relic of tpus needing padding to pass into - not needed anymore)
predictions = predictions_padded#[:-num_padding] if num_padding > 0 else predictions_padded

# Ensure predictions are not empty before proceeding
if predictions.shape[0] == 0:
    raise ValueError(f"Predictions are empty. Check if the model's input requirements are met. {predictions.shape}")

print("Predictions shape after trimming:", predictions.shape)

# Extract actual outputs from indices 132 to 165
actual_outputs_start_index = 132
actual_outputs_end_index = 166  # 166 because slicing excludes the end index
actual_outputs = completed_data_tensor[actual_outputs_start_index:actual_outputs_end_index]

# Make sure that the shape of actual outputs matches the predictions
actual_outputs = actual_outputs[:predictions.shape[0], np.newaxis, ...]

print("Actual outputs shape:", actual_outputs.shape)

# Save predictions and actual outputs for further analysis (optional)
np.save('/mnt/disks/data/predictions.npy', predictions)
np.save('/mnt/disks/data/actual_outputs.npy', actual_outputs)

# Load saved predictions and actual outputs - again
predictions = np.load('/mnt/disks/data/predictions.npy')
actual_outputs = np.load('/mnt/disks/data/actual_outputs.npy')

mse = np.mean(np.square(actual_outputs - predictions))
print(f"Mean Squared Error: {mse}")
###################################################################################
###################################################################################
###################################################################################
###################################################################################
###################################################################################
###################################################################################
###################################################################################

# predictions = np.load('/mnt/disks/data/predictions-backup.npy').squeeze()
# actual_outputs = np.load('/mnt/disks/data/actual_outputs-backup.npy').squeeze()

# print(actual_outputs.shape, predictions.shape)

# np.save('/mnt/disks/data/predictions.npy', predictions)
# np.save('/mnt/disks/data/actual_outputs.npy', actual_outputs)

###################################################################################
###################################################################################
###################################################################################
###################################################################################
###################################################################################
###################################################################################
###################################################################################

# FLDAS_data_tensor = np.load('/mnt/disks/data/FLDAS/dataset.npy', allow_pickle=True)
# FLDAS_times_tensor = np.load('/mnt/disks/data/FLDAS/times.npy', allow_pickle=True)
# food_security_index_tensor_validation = np.load('/mnt/disks/data/FEWS/ALL_HFIC/food_security_index_tensor_validation.npy', allow_pickle=True)
# food_security_index_times_validation = np.load('/mnt/disks/data/FEWS/ALL_HFIC/food_security_index_times_validation.npy', allow_pickle=True)
# # FLDAS_times_tensor = FLDAS_times_tensor[:166]
# # FLDAS_data_tensor = FLDAS_data_tensor[:166]
# FLDAS_times_tensor = FLDAS_times_tensor[120:166]
# FLDAS_data_tensor = FLDAS_data_tensor[120:166]
# fldas_dates = [pd.to_datetime(t[0]).to_pydatetime() for t in FLDAS_times_tensor]
# fews_dates = [pd.to_datetime(t).to_pydatetime() for t in food_security_index_times_validation]
# extrapolated_fews_tensor = np.zeros((FLDAS_data_tensor.shape[0], 256, 411))
# layers = []
# for i in range(extrapolated_fews_tensor.shape[0]):
#     blank_matrix = extrapolated_fews_tensor[i]
#     associated_time = fldas_dates[i]
#     latest_date = None
#     for date in fews_dates:
#         print(date, associated_time)
#         if date <= associated_time:
#             latest_date = date
#         else:
#             break
#     if latest_date is None:
#         # If no latest_date is found and the associated_time is before the first date in fews_dates,
#         # set latest_date to the first date in fews_dates
#         if associated_time < fews_dates[0]:
#             latest_date = fews_dates[0]
#         else:
#             raise ValueError("No associated date found for the given FLDAS date.")
#     print(i, associated_time, latest_date)
#     layers.append(food_security_index_tensor_validation[fews_dates.index(latest_date)])
# extrapolated_fews_tensor = np.stack(layers)
# FEWS_data_tensor = extrapolated_fews_tensor
# print(f"Extrapolated FEWS tensor shape: {FEWS_data_tensor.shape} FLDAS tensor shape: {FLDAS_data_tensor.shape}")
# FEWS_data_tensor = np.expand_dims(FEWS_data_tensor, axis=-1)
# completed_data_tensor = np.concatenate((FLDAS_data_tensor, FEWS_data_tensor), axis=-1)
# print(f"Combined data tensor shape: {completed_data_tensor.shape}")
# completed_data_tensor = np.nan_to_num(completed_data_tensor, copy=False)
# if np.isnan(completed_data_tensor).any():
#     print("There are still nan values in combined data tensor.")
# else:
#     print("No nan values in combined data tensor.")
# print(np.max(food_security_index_tensor_validation), np.min(food_security_index_tensor_validation))

# predictions = np.load('/mnt/disks/data/predictions-backup.npy')
# actual_outputs = np.load('/mnt/disks/data/actual_outputs.npy')

# predictions = predictions.squeeze()
# actual_outputs = actual_outputs.squeeze()

# original_min = np.min(predictions[..., 28])
# original_max = np.max(predictions[..., 28])

# print(f"Prediction: Max: {original_max}, Min: {original_min}")
# print(f"Actual: Max: {np.max(actual_outputs[..., 28])}, Min: {np.min(actual_outputs[..., 28])}")

# predictions[..., 28] = predictions[..., 28] * 4
# actual_outputs[..., 28] = np.round(actual_outputs[..., 28] * 4)

# print(f"Prediction: Max: {np.max(predictions[..., 28])}, Min: {np.min(predictions[..., 28])}")
# print(f"Actual: Max: {np.max(actual_outputs[..., 28])}, Min: {np.min(actual_outputs[..., 28])}")

# def gaussian(x, mu, sigma):
#     return np.exp(-0.5 * ((x - mu) / sigma) ** 2)

# differences = []
# hits = {
#     0.0: 0.0, 1.0: 0.0, 2.0: 0.0, 3.0: 0.0, 4.0: 0.0, 5.0: 0.0
# }
# totals = {
#     0.0: 0.0, 1.0: 0.0, 2.0: 0.0, 3.0: 0.0, 4.0: 0.0, 5.0: 0.0
# }
# differences_map = {
#     0.0: [], 1.0: [], 2.0: [], 3.0: [], 4.0: [], 5.0: []
# }

# sigma = 0.5

# for i in range(predictions.shape[0]):
#     print("Calculated gaussians for sample ", i)
#     for j in range(predictions.shape[1]):
#         for k in range(predictions.shape[2]):
#             pred = predictions[i, j, k, 28]
#             actual = actual_outputs[i, j, k, 28]
#             difference = pred - actual
#             gaussian_score = gaussian(pred, actual, sigma) # mu == actual
#             # print(f"Pred: {pred}, Actual: {actual}, Difference: {difference}, Gaussian Score: {gaussian_score}")
#             hits[actual] += gaussian_score
#             totals[actual] += 1
#             # print(hits[actual])
#             differences.append(difference)
#             differences_map[actual].append(difference)

# print("Hits: ", hits)
# print("Totals: ", totals)

# differences = np.array(differences)

# mse = np.mean(np.square(differences))

# hits_total = sum(hits.values())
# total_elements = predictions.shape[0] * predictions.shape[1] * predictions.shape[2]
# percentage_accuracy = (hits_total / total_elements) * 100

# print(f'MSE: {mse}')
# print(f'Percentage Accuracy: {percentage_accuracy}%')
# def calculate_absolute_deviation(data):
#     mean = np.mean(data)
#     absolute_deviation = np.abs(data - mean)
#     mean_absolute_deviation = np.mean(absolute_deviation)
#     return mean_absolute_deviation

# for key, value in hits.items():
#     if totals[key] > 0:  # Check to avoid division by zero
#         print("-" * 50)
#         accuracy = (value / totals[key]) * 100
#         print(f"{key}: {accuracy}% | {key} hit value: {value} | {key} total: {totals[key]}")
#         print(f"Absolute_differences: {calculate_absolute_deviation(differences_map[key])}")
#         print("-" * 50)
#     else:
#         print(f"{key}: N/A - No samples")
# print("differences mean: ", np.mean(differences))
# print("actual outputs mean: ", np.mean(actual_outputs[..., 28]))
# print("predictions mean: ", np.mean(differences[..., 28]))
# print(f"Absolute deviation for whole dataset: {calculate_absolute_deviation(differences)}")
# # # print("mse: ", np.mean(np.square(differences)))
# # # print("Percentage accuracy: " + str((hits / (predictions.shape[0] * predictions.shape[1] * predictions.shape[2])) * 100) + "%")
# # # print(f"percentage accuracy {percentage_accuracy}")

# import matplotlib.pyplot as plt

# def bell_curve(x, mu, sigma):
#     return np.exp(-np.power(x - mu, 2) / (2 * np.power(sigma, 2))) / (sigma * np.sqrt(2 * np.pi))

# x = np.arange(np.min(predictions[..., 28]), np.max(predictions[..., 28]), 0.01)

# for i in range(1,5):
#     filtered_predictions = predictions[actual_outputs[..., 28] == i, 28]
    
#     mu = np.mean(filtered_predictions)
#     sigma = np.std(filtered_predictions)
#     y = bell_curve(x, mu, sigma)
    
#     plt.plot(x, y)
#     plt.xlabel('Predicted Value')
#     plt.ylabel('Frequency')
#     plt.title(f'Gaussian Distribution for Actual Value {i}')
#     plt.savefig(f'guassian_{i}.png')
#     plt.close()
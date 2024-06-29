import tensorflow as tf
from tensorflow.python.client import device_lib
import keras
from keras.models import Sequential
from keras.layers import ConvLSTM2D, LayerNormalization, BatchNormalization, Conv3D, Conv2D, TimeDistributed, Conv3DTranspose, Reshape, Input
import numpy as np
from datetime import datetime, timedelta
import os
import pandas as pd

# Load datasets
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

#Temporal alignment
# FLDAS_times_tensor = FLDAS_times_tensor[:166]
# FLDAS_data_tensor = FLDAS_data_tensor[:166]

FLDAS_times_tensor = FLDAS_times_tensor[:132]
FLDAS_data_tensor = FLDAS_data_tensor[:132]

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
combined_data_tensor = np.concatenate((FLDAS_data_tensor, FEWS_data_tensor), axis=-1)
print(f"Combined data tensor shape: {combined_data_tensor.shape}")

combined_data_tensor = np.nan_to_num(combined_data_tensor, copy=False)
if np.isnan(combined_data_tensor).any():
    print("There are still nan values in combined data tensor.")
else:
    print("No nan values in combined data tensor.")

np.save('/mnt/disks/data/combined_data_tensor.npy', combined_data_tensor)

os.makedirs('saved_models', exist_ok=True)

############################################################################################################
############################################################################################################
#TRAINING

import tensorflow as tf
from keras.models import Sequential
from keras.layers import Input, ConvLSTM2D, LayerNormalization, TimeDistributed, Conv2D, Reshape, Conv3DTranspose, Dropout
from keras.optimizers import Adam
from datetime import datetime
import numpy as np

CHUNK_SIZE = 12
BATCH_SIZE = 8

def safe_mse_single_channel(y_true, y_pred):
    channel_weights = tf.constant([1.0] * 28 + [8.0], dtype=tf.float32)
    channel_weights = tf.convert_to_tensor(channel_weights, dtype=tf.float32)
    squared_difference = tf.square(y_true - y_pred)
    weights_shape = [1] * len(squared_difference.shape)  # Create a shape of all 1s matching the rank of squared_difference
    weights_shape[-1] = channel_weights.shape[0]  # Set the last dimension to match the number of channels
    expanded_weights = tf.reshape(channel_weights, weights_shape)
    weighted_squared_difference = squared_difference * expanded_weights
    weighted_mse = tf.reduce_mean(weighted_squared_difference, axis=[-2, -3, -4])  # Reduce mean across all but the batch and channel dimensions
    epsilon = 1e-7
    safe_mse = tf.where(tf.math.is_finite(weighted_mse), weighted_mse, epsilon)
    final_mse = tf.reduce_mean(safe_mse)
    return final_mse


tpu_devices = tf.config.list_logical_devices('TPU')
tpu_names = [device.name for device in tpu_devices]
print("TPUs available in TensorFlow:", tpu_names)

try:
    resolver = tf.distribute.cluster_resolver.TPUClusterResolver(tpu='local')
    tf.tpu.experimental.initialize_tpu_system(resolver)
    strategy = tf.distribute.TPUStrategy(resolver)
    print("TPU initialized successfully")
except Exception as e:
    print(f"Error initializing TPU: {e}")
    raise RuntimeError("Fatal error occurred. Halting the process. (TPU initialization)")

with strategy.scope():
    model = Sequential()
    model.add(Input(shape=(CHUNK_SIZE, 256, 411, 29)))

    model.add(ConvLSTM2D(filters=32, kernel_size=(5, 5), padding='same', return_sequences=True))
    model.add(LayerNormalization())
    model.add(TimeDistributed(Conv2D(filters=32, kernel_size=(1, 1), padding='same', activation='relu')))
    model.add(Dropout(0.2))

    model.add(ConvLSTM2D(filters=64, kernel_size=(5, 5), padding='same', return_sequences=True))
    model.add(LayerNormalization())
    model.add(TimeDistributed(Conv2D(filters=64, kernel_size=(1, 1), padding='same', activation='relu')))
    model.add(Dropout(0.2))

    model.add(ConvLSTM2D(filters=64, kernel_size=(5, 5), padding='same', return_sequences=True))
    model.add(LayerNormalization())
    model.add(TimeDistributed(Conv2D(filters=64, kernel_size=(1, 1), padding='same', activation='relu')))
    model.add(Dropout(0.2))

    model.add(ConvLSTM2D(filters=128, kernel_size=(5, 5), padding='same', return_sequences=False))
    model.add(LayerNormalization())
    model.add(Conv2D(filters=128, kernel_size=(1, 1), padding='same', activation='relu'))
    model.add(Dropout(0.2))

    model.add(Reshape((1, 256, 411, 128)))
    model.add(Conv3DTranspose(filters=29, kernel_size=(3, 3, 3), activation='sigmoid', padding='same')) # filters=29

    optimizer = Adam(clipnorm=1.0, learning_rate=1e-5)

    available_metrics = ['mse', 'mae', 'mape']

    model.compile(optimizer=optimizer, loss=safe_mse_single_channel, metrics=available_metrics)

    model.summary()

inputs = []
outputs = []

for i in range(CHUNK_SIZE, len(combined_data_tensor)):
    inputs.append(combined_data_tensor[i - CHUNK_SIZE:i])  # Previous CHUNK_SIZE states
    outputs.append(combined_data_tensor[i])  # Next state (t+1 prediction)

inputs = np.array(inputs)
outputs = np.array(outputs)

print("Input shape:", inputs.shape)  # Should be (num_samples, CHUNK_SIZE, 256, 411, 29)
print("Output shape:", outputs.shape)  # Should be (num_samples, 256, 411, 29)

def batch_generator(inputs, outputs, batch_size):
    num_samples = inputs.shape[0]
    indices = np.arange(num_samples)
    np.random.shuffle(indices)
    while True:
        for start in range(0, num_samples, batch_size):
            batch_indices = indices[start:start + batch_size]
            yield inputs[batch_indices], outputs[batch_indices]

train_generator = batch_generator(inputs, outputs, BATCH_SIZE)

log_dir = "logs/fit/" + datetime.now().strftime("%Y%m%d-%H%M%S")

model_path = datetime.now().strftime("%Y%m%d-%H%M%S") + '_model.keras'

callbacks = [
    tf.keras.callbacks.EarlyStopping(monitor='loss', patience=5, restore_best_weights=True),
    tf.keras.callbacks.ModelCheckpoint(model_path, save_best_only=True, monitor='loss'),
    tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)
]

with strategy.scope():
    history = model.fit(
        train_generator,
        steps_per_epoch=len(inputs) // BATCH_SIZE,
        epochs=80,  # CHANGE FOR TESTING
        callbacks=callbacks)

cpu_devices = tf.config.list_logical_devices('CPU')
cpu_names = [device.name for device in cpu_devices]
print("CPUs available in TensorFlow:", cpu_names)

model_path = 'saved_models/' + model_path
model.save(model_path)
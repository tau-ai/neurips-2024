import os
import tensorflow as tf
from tensorflow.python.client import device_lib
print(device_lib.list_local_devices())

os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = '/home/duckb/neurips/tpu-key.json'
try:
    resolver = tf.distribute.cluster_resolver.TPUClusterResolver(tpu='local')
    print(os.environ['TPU_NAME'])
    tf.tpu.experimental.initialize_tpu_system(resolver)
    strategy = tf.distribute.TPUStrategy(resolver)
    print("TPU initialized successfully")
except Exception as e:
    print(f"Error initializing TPU: {e}")
    raise

a = 3.0
x = tf.ones([3, 3], tf.float32)
y = tf.ones([3, 3], tf.float32)

with strategy.scope():
    @tf.function
    def tpu_computation(a, x, y):
        return a * x + y

    try:
        output = tpu_computation(a, x, y)
        tf.print(output)
    except Exception as e:
        print(f"Error during TPU computation: {e}")
        raise
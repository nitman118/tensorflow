import tensorflow as tf

x = tf.random.uniform([3, 3])

print("Is there a GPU available: "),
print(tf.config.experimental.list_physical_devices("GPU"))

print("Is the Tensor on GPU #0:  "),
print(x.device.endswith('GPU:0'))

@tf.function
def sig(x):
    return 1/(1-tf.exp(-x))

print(sig(3.0))

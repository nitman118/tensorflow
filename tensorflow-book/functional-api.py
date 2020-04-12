import tensorflow as tf # type: ignore
import os

root_logdir = os.path.join(os.getcwd(), "my_logs")

def get_run_logdir():
    import time
    run_id = time.strftime("run_%Y_%m_%d-%H_%M_%S")
    return os.path.join(root_logdir, run_id)



fashion_mnist_ds = tf.keras.datasets.fashion_mnist

(X_train_full, y_train_full), (X_test, y_test) = fashion_mnist_ds.load_data()
X_valid, X_train = X_train_full[:5000]/255.0, X_train_full[5000:]/255.0
y_valid, y_train = y_train_full[:5000], y_train_full[5000:]

input_ = tf.keras.layers.Input(shape = X_train.shape[1:])
flatten = tf.keras.layers.Flatten()(input_)
hidden1 = tf.keras.layers.Dense(256, activation="relu")(flatten)
hidden2 = tf.keras.layers.Dense(256, activation="relu")(hidden1)
output = tf.keras.layers.Dense(10, activation = "softmax")(hidden2)
model = tf.keras.Model(inputs = [input_], outputs = [output])
model.summary()


model.compile(optimizer = 'adam', loss = 'sparse_categorical_crossentropy', metrics = ['accuracy'])

run_logdir = get_run_logdir()

checkpoint_cb = tf.keras.callbacks.ModelCheckpoint("my_model.h5")
early_stopping_cb = tf.keras.callbacks.EarlyStopping(patience = 3, restore_best_weights = True)
tensorboard_cb = tf.keras.callbacks.TensorBoard(run_logdir)
# to run TensorBoard afterwards, on command line, run tensorboard --logdir=./my_logs --port=6006
if tf.keras.models.load_model('my_model.h5'):
    model = tf.keras.models.load_model('my_model.h5')

history = model.fit(X_train, y_train, batch_size=32, epochs = 10,
                    validation_data = (X_valid, y_valid), callbacks=[checkpoint_cb, early_stopping_cb, tensorboard_cb])

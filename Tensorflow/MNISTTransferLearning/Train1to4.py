from tensorflow import keras
from tensorflow.keras.datasets import mnist
from functools import partial
from sklearn.model_selection import train_test_split

(x_train, y_train), (x_test, y_test) = mnist.load_data()

x_train, x_test = x_train / 255.0, x_test / 255.0

x_train_14 = x_train[y_train < 5]
y_train_14 = y_train[y_train < 5]

x_test_14 = x_test[y_test < 5]
y_test_14 = y_test[y_test < 5]

x_train_14, x_train_14_valid, y_train_14, y_train_14_valid = train_test_split(x_train_14, y_train_14)

RegularizedDense = partial(keras.layers.Dense, activation='elu', kernel_initializer='he_normal')

model = keras.models.Sequential([
    keras.layers.Flatten(input_shape=[28, 28]),
    keras.layers.Dropout(rate=0.2),
    keras.layers.BatchNormalization(),
    RegularizedDense(100),
    keras.layers.Dropout(rate=0.2),
    keras.layers.BatchNormalization(),
    RegularizedDense(100),
    keras.layers.Dropout(rate=0.2),
    keras.layers.BatchNormalization(),
    RegularizedDense(100),
    keras.layers.Dropout(rate=0.2),
    keras.layers.BatchNormalization(),
    RegularizedDense(100),
    keras.layers.Dropout(rate=0.2),
    keras.layers.BatchNormalization(),
    RegularizedDense(100),
    keras.layers.Dropout(rate=0.2),
    keras.layers.BatchNormalization(),
    RegularizedDense(5, activation='softmax'),
])

optimizer = keras.optimizers.Adam()
model.compile(loss='sparse_categorical_crossentropy', metrics=['accuracy'], optimizer=optimizer)

early_stopping_cb = keras.callbacks.EarlyStopping(patience=10, restore_best_weights=True)
checkpoint_cb = keras.callbacks.ModelCheckpoint('cwiczenie1.h5')

# training model on entire training set
history = model.fit(x_train_14, y_train_14, epochs=30, validation_data=(x_train_14_valid, y_train_14_valid),
                    callbacks=[early_stopping_cb, checkpoint_cb])

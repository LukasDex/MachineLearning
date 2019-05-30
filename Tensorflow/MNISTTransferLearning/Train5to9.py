from tensorflow import keras
from tensorflow.keras.datasets import mnist
from functools import partial
from sklearn.model_selection import train_test_split

(x_train, y_train), (x_test, y_test) = mnist.load_data()

x_train, x_test = x_train / 255.0, x_test / 255.0

x_train_59 = x_train[y_train > 4]
y_train_59 = y_train[y_train > 4]

x_test_59 = x_test[y_test > 4]
y_test_59 = y_test[y_test > 4]

x_train_59, x_train_59_valid, y_train_59, y_train_59_valid = train_test_split(x_train_59, y_train_59)

x_train_59, y_train_59 = x_train_59[:500], y_train_59[:500]
x_train_59_valid, y_train_59_valid = x_train_59_valid[:200], y_train_59_valid[:200]

y_train_59 = y_train_59 - 5
y_train_59_valid = y_train_59_valid - 5

# loading model trained on numbers from 0 to 4
model = keras.models.load_model("cwiczenie1.h5")

# freezing all layers except from two last dense layers
for layer in model.layers[:-1]:
    layer.trainable = False

model.layers[-4].trainable = True
model.layers[-1] = keras.layers.Dense(5, activation='softmax', kernel_initializer='he_normal')

# learning model on small amount of data
optimizer = keras.optimizers.Adam()
model.compile(loss='sparse_categorical_crossentropy', metrics=['accuracy'], optimizer=optimizer)

early_stopping_cb = keras.callbacks.EarlyStopping(patience=10, restore_best_weights=True)

history = model.fit(x_train_59, y_train_59, epochs=30, validation_data=(x_train_59_valid, y_train_59_valid),
                    callbacks=[early_stopping_cb])

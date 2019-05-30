import tensorflow as tf
from tensorflow import keras
import pandas as pd
from sklearn.model_selection import train_test_split
from functools import partial
import pandas as pd
import matplotlib.pyplot as plt

fashion_mnist = keras.datasets.fashion_mnist

(X_train, y_train), (X_test, y_test) = fashion_mnist.load_data()

X_train, X_test = X_train / 255., X_test / 255.

X_train, X_train_valid, y_train, y_train_valid = train_test_split(X_train, y_train)

RegularizedDense = partial(keras.layers.Dense, activation='elu')

model = keras.models.Sequential([
    keras.layers.Flatten(input_shape=[28, 28]),
    keras.layers.Dropout(rate=0.3),
    keras.layers.BatchNormalization(),
    RegularizedDense(300),
    keras.layers.Dropout(rate=0.3),
    keras.layers.BatchNormalization(),
    RegularizedDense(150),
    keras.layers.Dropout(rate=0.3),
    keras.layers.BatchNormalization(),
    RegularizedDense(150),
    keras.layers.Dropout(rate=0.3),
    keras.layers.BatchNormalization(),
    keras.layers.Dense(10, activation='softmax')
])
# optimizer = keras.optimizers.SGD(lr=0.01, decay=1e-4)
model.compile(loss='sparse_categorical_crossentropy', metrics=['accuracy'], optimizer='adam')
early_stopping_cb = keras.callbacks.EarlyStopping(patience=10, restore_best_weights=True)

history = model.fit(X_train, y_train, epochs=50, validation_data=(X_train_valid, y_train_valid),
                    callbacks=[early_stopping_cb])

pd.DataFrame(history.history).plot(figsize=(8, 5))
plt.grid(True)
plt.gca().set_ylim(0, 1)  # set the vertical range to [0-1]
plt.show()

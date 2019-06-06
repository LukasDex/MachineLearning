import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.datasets import load_boston
from sklearn.model_selection import train_test_split

data = load_boston()
boston = pd.DataFrame(data.data, columns=data.feature_names)

print(boston.columns)
boston['MEDV'] = data.target

correlation_matrix = boston.corr()
sns.heatmap(data=correlation_matrix, annot=True).set_title('Correlation matrix')
plt.show()
# matrix is showing strong correlation between target value and RM (number of rooms) and LSTAT (% of low status population)

target = 'MEDV'
features = ['RM', 'LSTAT']
# plotting house median with respect to features
for feature in features:
    ax = boston.plot.scatter(x=feature, y='MEDV', label=feature)
    plt.show()

price = np.array(boston['MEDV'], np.float32)
rooms = np.array(boston['RM'], np.float32)
low_status_pop = np.array(boston['LSTAT'], np.float32)
rooms_train, rooms_valid, low_status_pop_train, low_status_pop_valid = train_test_split(rooms, low_status_pop,
                                                                                        random_state=42)
price_train, price_valid = train_test_split(price, random_state=42)

# setting variables
intercept = tf.Variable(0.1, np.float32)
slope_1 = tf.Variable(0.1, np.float32)
slope_2 = tf.Variable(0.1, np.float32)


# creating loss function
def loss_function(intercept, slope_1, slope_2, price, low_status_pop, rooms):
    return tf.keras.losses.mae(price, intercept + low_status_pop * slope_1 + rooms * slope_2)


opt = tf.keras.optimizers.Adam()
for j in range(8000):
    opt.minimize(lambda: loss_function(intercept, slope_1, slope_2, price_train, low_status_pop_train, rooms_train),
                 var_list=[intercept, slope_1, slope_2])
    if j % 500 == 0:
        print('Loss on train set:    ' + str(loss_function(intercept, slope_1, slope_2, price_train,
                                                           low_status_pop_train, rooms_train).numpy()), end='  ')
        print('Loss on valid set:    ' + str(loss_function(intercept, slope_1, slope_2, price_valid,
                                                           low_status_pop_valid, rooms_valid).numpy()))

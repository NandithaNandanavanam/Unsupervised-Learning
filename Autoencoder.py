# Read Fashion MNIST dataset

import util_mnist_reader
X_train, y_train = util_mnist_reader.load_mnist('../data/fashion', kind='train')
X_test, y_test = util_mnist_reader.load_mnist('../data/fashion', kind='t10k')

# Your code goes here . . .
#Auto encoder
#-------------------------------------------------------------------------------------------------------------------------------
#Libraries
import util_mnist_reader
import numpy as np
import matplotlib.pyplot as plt
from sklearn import metrics, cluster
import tensorflow
from tensorflow.keras import layers, models, initializers, optimizers, utils
from tensorflow.keras.layers import Dense, Dropout, Conv2D, MaxPool2D, UpSampling2D, Activation
from tensorflow.keras.optimizers import SGD
from scipy.stats import mode
from sklearn.cluster import KMeans
from tensorflow.keras import backend as K
from sklearn.metrics import confusion_matrix
from tensorflow.keras.models import model_from_json

#Read Fashion MNIST dataset
X_train, y_train = util_mnist_reader.load_mnist('../data/fashion', kind='train')
X_test, y_test = util_mnist_reader.load_mnist('../data/fashion', kind='t10k')

#Preprocessing and normalization of data
X_train = X_train.reshape(-1,28,28,1) / 255
X_test = X_test.reshape(-1,28,28,1) / 255

batch_size = 128
epochs = 10
num_of_clusters = 10 

model = models.Sequential()
model.add(Conv2D(14, kernel_size=3, padding='same', activation='relu', input_shape=(28,28,1)))
model.add(MaxPool2D((2,2), padding='same'))
model.add(Dropout(0.2))
model.add(Conv2D(7, kernel_size=3, padding='same', activation='relu'))
model.add(MaxPool2D((2,2), padding='same'))
model.add(Dropout(0.2))
model.add(Conv2D(7, kernel_size=3, padding='same', activation='relu'))
model.add(UpSampling2D((2,2)))
model.add(Dropout(0.2))
model.add(Conv2D(14, kernel_size=3, padding='same', activation='relu'))
model.add(UpSampling2D((2,2)))
model.add(Dropout(0.2))
model.add(Conv2D(1, kernel_size=3, padding='same', activation='relu'))

model.compile(optimizer='adam', loss="mse", metrics = ['accuracy'])

model.summary()

autoencoder_train = model.fit(X_train, X_train, epochs=epochs, batch_size=batch_size, validation_data=(X_test, X_test), verbose=1)

loss = autoencoder_train.history['loss']
val_loss = autoencoder_train.history['val_loss']
epochs = range(epochs)
plt.figure()
plt.plot(epochs, loss, 'g', label='Training loss')
plt.plot(epochs, val_loss, 'b', label='Validation loss')
plt.title('Training and validation loss')
plt.legend()
plt.show()

#serialize model to JSON
model_json = model.to_json()
with open("model.json", "w") as json_file:
    json_file.write(model_json)
#serialize weights to HDF5
model.save_weights("model.h5")
print("Saved model to disk")

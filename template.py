#
#
#

import keras
from keras.models import Sequential
from keras.layers import Dense, Activation
import numpy as np
import matplotlib.pyplot as plt
import random

# Remove limits on numpy array print length
np.set_printoptions(threshold=np.inf)


# ==================================================
#               Data Preprocessing
# ==================================================

# Load images in
images = np.load('data/images.npy')

# Load labels in
labels = np.load('data/labels.npy')


# Convert image labels to "one-hot vectors"
one_hot_labels = keras.utils.to_categorical(labels)
#for l in one_hot_labels:
#    print(l)

# Preprocess images
#vectorized_images = []
x_train = []
y_train = []
x_val = []
y_val = []
x_test = []
y_test = []
index = 0

for i in images:
    vectorized_image = np.reshape(i, -1).astype('float32')
    ran_seed = random.random()
    #print(ran_seed)
    if ran_seed < 0.6:
        x_train.append(vectorized_image)
        y_train.append(one_hot_labels[index])
    elif ran_seed < 0.75:
        x_val.append(vectorized_image)
        y_val.append(one_hot_labels[index])
    else:
        x_test.append(vectorized_image)
        y_test.append(one_hot_labels[index])
    index += 1

x_train = np.array(x_train)
y_train = np.array(y_train)
x_val = np.array(x_val)
y_val = np.array(y_val)
x_test = np.array(x_test)
y_test = np.array(y_test)

x_train /= 255
x_val /= 255
x_test /= 255

#print('Image 0:')
#print(images[0])
#print(vectorized_images[0])
#print('Image 1:')
#print(images[1])
#print(vectorized_images[1])


# Determine initial two layers - the number of pixels and the number of classes
px_count = images.shape[1] * images.shape[2]
class_count = one_hot_labels.shape[1]
#print(one_hot_labels)
#print(class_count)

# ==================================================
#               Plotting sample images
# ==================================================

''' plt.subplot(221)
plt.imshow(images[0], cmap=plt.get_cmap('gray'))
print(labels[0])
plt.subplot(222)
plt.imshow(images[1], cmap=plt.get_cmap('gray'))
print(labels[1])
plt.show() '''


# ==================================================
#               
# ==================================================


# Model Template

model = Sequential() # declare model
model.add(Dense(px_count, activation='relu', kernel_initializer='he_normal', input_dim=px_count)) # simplified first layer
model.add(Dense(class_count, activation='softmax', kernel_initializer='he_normal'))

model.add(Dense(10, kernel_initializer='he_normal', activation='softmax')) # Required final layer

# Compile Model
model.compile(optimizer='sgd', loss='categorical_crossentropy', metrics=['accuracy'])

# Train Model
history = model.fit(x_train, y_train, validation_data = (x_val, y_val), epochs=125, batch_size=3, verbose=2)


# Report Results
print(history.history)
#scores = model.evaluate(x_val, y_val)
#print(scores)
#model.predict(x_test)